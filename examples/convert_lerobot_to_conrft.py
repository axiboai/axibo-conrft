"""Convert LeRobot (v2.1) datasets on the Hugging Face Hub into the pickle
formats consumed by this ConRFT codebase.

Two output modes:

  demos       -> Stage I (Cal-ConRFT) demonstration transitions, mirroring what
                 ``record_demos_octo.py`` produces. Use the clean successful
                 expert episodes (e.g. ``Ishan-Axibo/piperx_flatten_depth``).

  classifier  -> success / failure frames for ``train_reward_classifier.py``.
                 Use the labelled rollouts (e.g. ``Ishan-Axibo/piperx_towel_rollouts``)
                 whose per-episode ``is_success`` lives in ``meta/episodes.jsonl``.

This reads the raw parquet + mp4 files directly (via ``huggingface_hub`` +
``pyav``), so it does not depend on a specific ``lerobot`` package version.

Camera aliasing
---------------
The Octo encoder in this repo is hard-wired to two views named
``side_policy_256`` (Octo ``image_primary``) and ``wrist_1`` (Octo
``image_wrist``). We therefore alias the bimanual cameras onto those names:

  cam_front        -> side_policy_256   (256x256)
  cam_<wrist>_wrist -> wrist_1          (128x128)

The second wrist view is unused by Octo-small (primary + single wrist only).

Examples
--------
  # Stage I demos from the clean expert dataset (needs Octo + GPU for embeddings)
  python convert_lerobot_to_conrft.py demos \
      --repo_id Ishan-Axibo/piperx_flatten_depth \
      --out_dir experiments/task_towel_fold/demo_data \
      --octo_path /path/to/octo-small \
      --task_desc "pick towel from pile, fold and stack"

  # Reward-classifier frames from the labelled rollouts
  python convert_lerobot_to_conrft.py classifier \
      --repo_id Ishan-Axibo/piperx_towel_rollouts \
      --out_dir experiments/task_towel_fold/classifier_data
"""

import os
import json
import copy
import pickle as pkl
import datetime
from collections import deque

import numpy as np
from absl import app, flags

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required: pip install opencv-python") from e

try:
    import av  # PyAV decodes the AV1 mp4 videos
except ImportError as e:  # pragma: no cover
    raise ImportError("PyAV is required to decode videos: pip install av") from e

try:
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    raise ImportError("pyarrow is required: pip install pyarrow") from e

from huggingface_hub import hf_hub_download


FLAGS = flags.FLAGS

flags.DEFINE_string("repo_id", None, "Hugging Face dataset repo id.")
flags.DEFINE_string("out_dir", None, "Output directory for the .pkl files.")
flags.DEFINE_string("primary_cam", "observation.images.cam_front",
                    "Video key mapped to Octo image_primary (side_policy_256).")
flags.DEFINE_string("wrist_cam", "observation.images.cam_right_wrist",
                    "Video key mapped to Octo image_wrist (wrist_1).")
flags.DEFINE_integer("primary_size", 256, "Resize for the primary camera (square).")
flags.DEFINE_integer("wrist_size", 128, "Resize for the wrist camera (square).")
flags.DEFINE_integer("obs_horizon", 2, "Frame stacking horizon (must match training).")
flags.DEFINE_integer("control_hz", 10,
                     "Target control rate; frames are subsampled from the dataset fps.")
flags.DEFINE_integer("max_episodes", 0, "If >0, only process this many episodes.")

# demos-only flags
flags.DEFINE_integer("max_len", 0, "If >0, cap each demo trajectory to this many steps.")
flags.DEFINE_float("reward_pos", 1.0, "Sparse reward at the successful terminal step.")
flags.DEFINE_float("reward_neg", -0.05, "Per-step reward before success.")
flags.DEFINE_float("discount", 0.98, "Discount used for Monte-Carlo returns.")
flags.DEFINE_string("octo_path", None,
                    "Path to the Octo model. If unset, demos are written WITHOUT "
                    "embeddings (the learner requires embeddings, so set this).")
flags.DEFINE_string("task_desc", "pick towel from pile, fold and stack",
                    "Language task used to build Octo task embeddings.")
flags.DEFINE_integer("episodes_per_file", 0,
                     "If >0, shard demo output into files of this many episodes.")

# classifier-only flags
flags.DEFINE_integer("success_tail_frames", 10,
                     "Number of final frames of each successful episode used as positives.")
flags.DEFINE_integer("neg_stride", 5,
                     "Sample one negative frame every N (subsampled) frames.")


PRIMARY_KEY = "side_policy_256"
WRIST_KEY = "wrist_1"


def _download_json(repo_id, path):
    local = hf_hub_download(repo_id, path, repo_type="dataset")
    with open(local) as f:
        return json.load(f)


def _download_jsonl(repo_id, path):
    local = hf_hub_download(repo_id, path, repo_type="dataset")
    rows = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_paths(info, episode_index):
    chunk = episode_index // info["chunks_size"]
    data_path = info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index)
    return chunk, data_path


def _video_path(info, episode_index, video_key):
    chunk = episode_index // info["chunks_size"]
    return info["video_path"].format(
        episode_chunk=chunk, video_key=video_key, episode_index=episode_index)


def _decode_video(local_mp4, size):
    """Decode an mp4 into a list of RGB uint8 frames resized to (size, size)."""
    frames = []
    container = av.open(local_mp4)
    try:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            img = cv2.resize(img, (size, size))
            frames.append(img.astype(np.uint8))
    finally:
        container.close()
    return frames


def _read_episode(repo_id, info, episode_index, primary_cam, wrist_cam,
                  primary_size, wrist_size):
    """Return per-frame state, action, primary frames, wrist frames for one episode."""
    _, data_path = _episode_paths(info, episode_index)
    local_parquet = hf_hub_download(repo_id, data_path, repo_type="dataset")
    table = pq.read_table(local_parquet).to_pydict()

    states = np.asarray(table["observation.state"], dtype=np.float32)
    actions = np.asarray(table["action"], dtype=np.float32)
    n = len(states)

    primary_local = hf_hub_download(
        repo_id, _video_path(info, episode_index, primary_cam), repo_type="dataset")
    wrist_local = hf_hub_download(
        repo_id, _video_path(info, episode_index, wrist_cam), repo_type="dataset")

    primary_frames = _decode_video(primary_local, primary_size)
    wrist_frames = _decode_video(wrist_local, wrist_size)

    # Guard against off-by-a-few mismatches between parquet rows and decoded frames.
    n = min(n, len(primary_frames), len(wrist_frames))
    return states[:n], actions[:n], primary_frames[:n], wrist_frames[:n]


def _subsample_indices(n, fps, control_hz, max_len):
    stride = max(1, round(fps / control_hz))
    idxs = list(range(0, n, stride))
    if max_len and max_len > 0:
        idxs = idxs[:max_len]
    return idxs


def _make_obs(state, primary_img, wrist_img):
    return {
        "state": state.astype(np.float32),
        PRIMARY_KEY: primary_img,
        WRIST_KEY: wrist_img,
    }


def _stack(frame_deque):
    """Stack a deque of single-frame obs dicts into a horizon-stacked obs dict."""
    keys = frame_deque[0].keys()
    return {k: np.stack([f[k] for f in frame_deque], axis=0) for k in keys}


def convert_demos(_):
    info = _download_json(FLAGS.repo_id, "meta/info.json")
    fps = info["fps"]
    total = info["total_episodes"]
    if FLAGS.max_episodes:
        total = min(total, FLAGS.max_episodes)

    # Optional success filter from metadata (flatten_depth is all-success, but be safe).
    is_success = {}
    try:
        for row in _download_jsonl(FLAGS.repo_id, "meta/episodes.jsonl"):
            is_success[row["episode_index"]] = bool(row.get("is_success", True))
    except Exception:
        pass

    model, tasks = None, None
    if FLAGS.octo_path:
        from octo.model.octo_model import OctoModel
        from data_util import (add_mc_returns_to_trajectory,
                               add_embeddings_to_trajectory,
                               add_next_embeddings_to_trajectory)
        model = OctoModel.load_pretrained(FLAGS.octo_path)
        tasks = model.create_tasks(texts=[FLAGS.task_desc])
    else:
        from data_util import add_mc_returns_to_trajectory
        print("\033[93mWARNING: --octo_path not set; writing demos WITHOUT Octo "
              "embeddings. The learner needs embeddings, so add them before "
              "Stage I training.\033[00m")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    all_transitions = []
    shard_index = 0
    n_demo_episodes = 0

    for ep in range(total):
        if not is_success.get(ep, True):
            continue
        states, actions, primary_frames, wrist_frames = _read_episode(
            FLAGS.repo_id, info, ep, FLAGS.primary_cam, FLAGS.wrist_cam,
            FLAGS.primary_size, FLAGS.wrist_size)
        n = len(states)
        if n < 2:
            continue
        idxs = _subsample_indices(n, fps, FLAGS.control_hz, FLAGS.max_len)
        if len(idxs) < 2:
            continue

        obs_deque = deque(maxlen=FLAGS.obs_horizon)
        next_deque = deque(maxlen=FLAGS.obs_horizon)
        first = _make_obs(states[idxs[0]], primary_frames[idxs[0]], wrist_frames[idxs[0]])
        for _ in range(FLAGS.obs_horizon):
            obs_deque.append(copy.deepcopy(first))

        trajectory = []
        for j in range(len(idxs) - 1):
            t, t_next = idxs[j], idxs[j + 1]
            cur = _make_obs(states[t], primary_frames[t], wrist_frames[t])
            nxt = _make_obs(states[t_next], primary_frames[t_next], wrist_frames[t_next])

            obs_deque.append(cur)
            next_deque = deque(list(obs_deque)[1:] + [nxt], maxlen=FLAGS.obs_horizon)

            done = (j == len(idxs) - 2)
            reward = FLAGS.reward_pos if done else FLAGS.reward_neg
            transition = dict(
                observations=_stack(obs_deque),
                actions=actions[t].astype(np.float32),
                next_observations=_stack(next_deque),
                rewards=float(reward),
                masks=1.0 - float(done),
                dones=bool(done),
            )
            trajectory.append(transition)

        trajectory = add_mc_returns_to_trajectory(
            trajectory, FLAGS.discount, reward_scale=1.0, reward_bias=0.0,
            reward_neg=FLAGS.reward_neg, is_sparse_reward=True)
        if model is not None:
            trajectory = add_embeddings_to_trajectory(trajectory, model, tasks=tasks)
            trajectory = add_next_embeddings_to_trajectory(trajectory)

        all_transitions.extend(copy.deepcopy(trajectory))
        n_demo_episodes += 1
        print(f"[demos] episode {ep}: {len(trajectory)} transitions "
              f"(total {len(all_transitions)})")

        if FLAGS.episodes_per_file and n_demo_episodes % FLAGS.episodes_per_file == 0:
            _dump_demos(all_transitions, shard_index)
            all_transitions = []
            shard_index += 1

    if all_transitions:
        _dump_demos(all_transitions, shard_index)
    print(f"\033[92mDone. Converted {n_demo_episodes} successful episodes.\033[00m")


def _dump_demos(transitions, shard_index):
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"towel_demos_{len(transitions)}_s{shard_index}_{uuid}.pkl"
    path = os.path.join(FLAGS.out_dir, name)
    with open(path, "wb") as f:
        pkl.dump(transitions, f)
    print(f"\033[92msaved {len(transitions)} transitions to {path}\033[00m")


def convert_classifier(_):
    info = _download_json(FLAGS.repo_id, "meta/info.json")
    fps = info["fps"]
    total = info["total_episodes"]
    if FLAGS.max_episodes:
        total = min(total, FLAGS.max_episodes)

    is_success = {row["episode_index"]: bool(row["is_success"])
                  for row in _download_jsonl(FLAGS.repo_id, "meta/episodes.jsonl")}

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    successes, failures = [], []

    for ep in range(total):
        states, actions, primary_frames, wrist_frames = _read_episode(
            FLAGS.repo_id, info, ep, FLAGS.primary_cam, FLAGS.wrist_cam,
            FLAGS.primary_size, FLAGS.wrist_size)
        n = len(states)
        if n < 1:
            continue
        idxs = _subsample_indices(n, fps, FLAGS.control_hz, max_len=0)

        # The reward classifier runs on the wrapped env's stacked observations
        # (stack_obs_num=2), so repeat each frame across the horizon to match.
        # Keep full transition keys so ReplayBuffer.insert works in
        # train_reward_classifier.py.
        def transition_at(t):
            t_next = min(t + 1, n - 1)
            cur = _make_obs(states[t], primary_frames[t], wrist_frames[t])
            nxt = _make_obs(states[t_next], primary_frames[t_next], wrist_frames[t_next])
            stacked_cur = {k: np.stack([cur[k]] * FLAGS.obs_horizon, axis=0)
                           for k in cur}
            stacked_next = {k: np.stack([nxt[k]] * FLAGS.obs_horizon, axis=0)
                            for k in nxt}
            return dict(
                observations=stacked_cur,
                actions=actions[t].astype(np.float32),
                next_observations=stacked_next,
                rewards=0.0,
                masks=1.0,
                dones=False,
            )

        if is_success.get(ep, False):
            k = FLAGS.success_tail_frames
            tail = idxs[-k:] if k else idxs
            head = idxs[:-k] if k else []
            for t in tail:
                successes.append(transition_at(t))
            # earlier (not-yet-folded) frames of a success are still negatives
            for t in head[::FLAGS.neg_stride]:
                failures.append(transition_at(t))
        else:
            for t in idxs[::FLAGS.neg_stride]:
                failures.append(transition_at(t))
        print(f"[classifier] episode {ep} success={is_success.get(ep, False)} "
              f"pos={len(successes)} neg={len(failures)}")

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pos_path = os.path.join(FLAGS.out_dir, f"towel_{len(successes)}_success_images_{uuid}.pkl")
    neg_path = os.path.join(FLAGS.out_dir, f"towel_failure_images_{uuid}.pkl")
    with open(pos_path, "wb") as f:
        pkl.dump(successes, f)
    with open(neg_path, "wb") as f:
        pkl.dump(failures, f)
    print(f"\033[92msaved {len(successes)} success / {len(failures)} failure frames "
          f"to {FLAGS.out_dir}\033[00m")


def main(argv):
    assert FLAGS.repo_id is not None, "--repo_id is required"
    assert FLAGS.out_dir is not None, "--out_dir is required"
    mode = argv[1] if len(argv) > 1 else None
    if mode == "demos":
        convert_demos(argv)
    elif mode == "classifier":
        convert_classifier(argv)
    else:
        raise ValueError("First positional arg must be 'demos' or 'classifier'. "
                         f"Got: {mode!r}")


if __name__ == "__main__":
    app.run(main)
