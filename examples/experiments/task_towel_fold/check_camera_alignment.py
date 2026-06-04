#!/usr/bin/env python3
"""Diagnose train/inference observation mismatch for the towel task.

The policy recomputes Octo embeddings from the LIVE ``side_policy_256`` and
``wrist_1`` images at every step (see ConrftCPOctoAgentSingleArm.sample_actions
-> forward_policy with action_embeddings=None). So if the live camera wiring in
``config.py::EnvConfig.CAMERAS`` does not feed the SAME physical views that the
demo pkl was built from, offline checks look perfect but live rollout produces
nonsense actions.

This tool dumps:
  1. the demo pkl's stored ``side_policy_256`` / ``wrist_1`` frames, and
  2. one live frame from each candidate ZMQ camera port,
so you can eyeball which live port matches the training primary/wrist views.

Run it ON THE ROLLOUT BOX (it needs both the demo file and the live streams):

    cd examples/experiments/task_towel_fold
    python check_camera_alignment.py \
        --demo_path ./demo_data/towel_demos_29452_s0_2026-06-03_20-07-16.pkl \
        --ports 5556,5558,5560 \
        --out_dir ./_cam_check
"""

import argparse
import os
import pickle as pkl
import struct
import time

import cv2
import numpy as np
import zmq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo_path", default=None,
                   help="Path to towel_demos_*.pkl. Omit to skip the demo dump "
                        "(e.g. on the rollout box, which only has live cameras).")
    p.add_argument("--ports", default="5556,5558,5560",
                   help="Comma-separated localhost camera ports to sample. Pass "
                        "an empty string to skip live capture (e.g. on the cloud "
                        "VM, which only has the demo pkl).")
    p.add_argument("--host", default="localhost")
    p.add_argument("--out_dir", default="./_cam_check")
    p.add_argument("--poll_seconds", type=float, default=2.0,
                   help="How long to poll each live camera for a frame.")
    return p.parse_args()


def _last_frame(stacked):
    """Demo obs images are horizon-stacked (T, H, W, 3); take the last frame."""
    arr = np.asarray(stacked)
    if arr.ndim == 4:
        return arr[-1]
    return arr


def dump_demo_frames(demo_path, out_dir):
    with open(demo_path, "rb") as f:
        demos = pkl.load(f)
    assert len(demos) > 0, "Demo file is empty"

    # Pick a mid-episode transition so the scene is representative.
    tr = demos[len(demos) // 2]
    obs = tr["observations"]

    saved = {}
    for key in ("side_policy_256", "wrist_1"):
        if key not in obs:
            print(f"[demo] WARNING: key {key!r} not in demo observations "
                  f"(have: {list(obs.keys())})")
            continue
        frame = _last_frame(obs[key]).astype(np.uint8)
        path = os.path.join(out_dir, f"DEMO_{key}.png")
        # demo frames are stored RGB; cv2 writes BGR.
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        saved[key] = (path, frame.shape)
        print(f"[demo] saved {key} frame {frame.shape} -> {path}")
    return saved


def grab_live_frame(ctx, host, port, poll_seconds, topic=b"image"):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.RCVHWM, 1)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{host}:{port}")
    s.setsockopt(zmq.SUBSCRIBE, topic)
    last_rgb = None
    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        try:
            parts = s.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.01)
            continue
        if len(parts) < 3 or parts[0] != topic or len(parts[1]) != 8:
            continue
        try:
            struct.unpack(">Q", parts[1])[0]
        except struct.error:
            continue
        jpeg = parts[2]
        if len(jpeg) < 2 or jpeg[:2] != b"\xff\xd8":
            continue
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            last_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    s.close(linger=0)
    return last_rgb


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.demo_path:
        print("=== Demo (training) frames ===")
        dump_demo_frames(args.demo_path, args.out_dir)
    else:
        print("=== Demo dump skipped (no --demo_path) ===")

    print("\n=== Live camera frames ===")
    ctx = zmq.Context.instance()
    for port in [p.strip() for p in args.ports.split(",") if p.strip()]:
        rgb = grab_live_frame(ctx, args.host, int(port), args.poll_seconds)
        if rgb is None:
            print(f"[live] port {port}: NO FRAME (stream down or wrong port)")
            continue
        path = os.path.join(args.out_dir, f"LIVE_port{port}.png")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[live] port {port}: frame {rgb.shape} -> {path}")

    print(
        "\nNext: open the PNGs in --out_dir. The live port whose image matches "
        "DEMO_side_policy_256.png MUST be wired to 'side_policy_256' in "
        "config.py::EnvConfig.CAMERAS, and the one matching DEMO_wrist_1.png to "
        "'wrist_1'. If they are swapped or point at the wrong camera, that is the "
        "cause of the bad rollout."
    )


if __name__ == "__main__":
    main()
