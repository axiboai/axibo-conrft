#!/usr/bin/env python3
"""Localize the train/inference gap for the towel policy.

Cameras are verified aligned and the offline check passes, yet live rollout is
random. The remaining difference is the VALUES the policy sees at runtime. This
tool runs the SAME restored policy on (a) a real demo observation and (b) a live
observation, and prints the proprio ``state`` vector and the sampled action for
both, so we can see exactly which input drifts.

Two-step usage (the demo pkl lives on the cloud VM, the robot on the rollout
box), bridged by a small reference file:

  # 1) ON THE CLOUD VM (has the demo pkl) -- no robot needed:
  python compare_live_vs_demo.py export \
      --demo_path ./demo_data/towel_demos_29452_s0_2026-06-03_20-07-16.pkl \
      --ref_out ./demo_ref.npz --n_samples 6

  # 2) copy ./demo_ref.npz to the rollout box, then ON THE ROLLOUT BOX
  #    (has the robot + live cameras):
  python compare_live_vs_demo.py compare \
      --exp_name task_towel_fold \
      --checkpoint_path ./conrft_fixed_20k_only \
      --eval_checkpoint_step 20000 \
      --ref_in ./demo_ref.npz
"""

import argparse
import pickle as pkl

import numpy as np


def _stats(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    print(f"  {name}: shape={arr.shape} "
          f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")


# --------------------------------------------------------------------------- #
# Step 1: export a compact demo reference on the VM (no JAX / robot required).
# --------------------------------------------------------------------------- #
def do_export(args):
    with open(args.demo_path, "rb") as f:
        demos = pkl.load(f)
    assert len(demos) > 0, "Demo file is empty"

    idx = np.linspace(0, len(demos) - 1, args.n_samples).astype(int)
    states, primaries, wrists, gt_actions = [], [], [], []
    for i in idx:
        obs = demos[int(i)]["observations"]
        states.append(np.asarray(obs["state"], dtype=np.float32))
        primaries.append(np.asarray(obs["side_policy_256"], dtype=np.uint8))
        wrists.append(np.asarray(obs["wrist_1"], dtype=np.uint8))
        gt_actions.append(np.asarray(demos[int(i)]["actions"], dtype=np.float32))

    np.savez_compressed(
        args.ref_out,
        state=np.stack(states),
        side_policy_256=np.stack(primaries),
        wrist_1=np.stack(wrists),
        gt_action=np.stack(gt_actions),
    )
    print(f"[export] wrote {len(idx)} demo samples -> {args.ref_out}")
    print("[export] demo proprio state per-sample (last timestep):")
    for k, s in enumerate(states):
        last = s[-1] if s.ndim == 2 else s
        print(f"  sample {k}: {np.array2string(last, precision=3, max_line_width=200)}")
    print("[export] demo proprio range (per dim, over samples, last timestep):")
    last_states = np.stack([s[-1] if s.ndim == 2 else s for s in states])
    print("  min :", np.array2string(last_states.min(0), precision=3, max_line_width=200))
    print("  max :", np.array2string(last_states.max(0), precision=3, max_line_width=200))


# --------------------------------------------------------------------------- #
# Step 2: compare on the rollout box (needs JAX, octo, the env, the robot).
# --------------------------------------------------------------------------- #
def do_compare(args):
    import jax
    from flax.training import checkpoints
    from experiments.mappings import CONFIG_MAPPING
    from octo.model.octo_model import OctoModel
    from serl_launcher.utils.launcher import (
        make_conrft_octo_cp_pixel_agent_single_arm,
    )

    config = CONFIG_MAPPING[args.exp_name]()
    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

    # Build the env WITHOUT the classifier so we don't need the GPU twice; we
    # only use it to fetch a live observation.
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    agent = make_conrft_octo_cp_pixel_agent_single_arm(
        seed=args.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        sample_tasks=tasks,
        octo_model=octo_model,
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        fix_gripper=False,
        q_weight=1.0,
        bc_weight=0.1,
    )
    ckpt = checkpoints.restore_checkpoint(
        args.checkpoint_path, agent.state, step=args.eval_checkpoint_step)
    agent = agent.replace(state=ckpt)
    print(f"[compare] restored checkpoint step {args.eval_checkpoint_step} "
          f"from {args.checkpoint_path}")

    def policy_action(obs, n=5):
        acts = []
        rng = jax.random.PRNGKey(args.seed)
        for _ in range(n):
            rng, key = jax.random.split(rng)
            a, _ = agent.sample_actions(
                observations=jax.device_put(obs),
                tasks=jax.device_put(tasks),
                seed=key,
                argmax=False,
            )
            acts.append(np.asarray(jax.device_get(a), dtype=np.float32))
        return np.stack(acts)

    # ---- (A) DEMO observation ------------------------------------------------
    ref = np.load(args.ref_in)
    print("\n================ DEMO observation (from pkl) ================")
    demo_obs = {
        "state": ref["state"][0],
        "side_policy_256": ref["side_policy_256"][0],
        "wrist_1": ref["wrist_1"][0],
    }
    _stats("demo state", demo_obs["state"])
    _stats("demo side_policy_256", demo_obs["side_policy_256"])
    _stats("demo wrist_1", demo_obs["wrist_1"])
    print("  demo state (last timestep):",
          np.array2string(np.asarray(demo_obs["state"])[-1], precision=3, max_line_width=200))
    demo_acts = policy_action(demo_obs)
    print("  gt action :", np.array2string(ref["gt_action"][0], precision=3, max_line_width=200))
    print("  policy mean action:", np.array2string(demo_acts.mean(0), precision=3, max_line_width=200))
    print("  policy std  action:", np.array2string(demo_acts.std(0), precision=3, max_line_width=200))
    print(f"  policy action saturation (|a|>=0.95): {(np.abs(demo_acts) >= 0.95).mean():.3f}")

    # ---- (B) LIVE observation ------------------------------------------------
    print("\n================ LIVE observation (from robot) ================")
    obs, _ = env.reset()
    _stats("live state", obs["state"])
    _stats("live side_policy_256", obs["side_policy_256"])
    _stats("live wrist_1", obs["wrist_1"])
    print("  live state (last timestep):",
          np.array2string(np.asarray(obs["state"])[-1], precision=3, max_line_width=200))
    live_acts = policy_action(obs)
    print("  policy mean action:", np.array2string(live_acts.mean(0), precision=3, max_line_width=200))
    print("  policy std  action:", np.array2string(live_acts.std(0), precision=3, max_line_width=200))
    print(f"  policy action saturation (|a|>=0.95): {(np.abs(live_acts) >= 0.95).mean():.3f}")

    print("\n================ READOUT ================")
    print("Compare the two 'state (last timestep)' rows: if the live proprio is "
          "a different scale/sign/range than the demo proprio, that is the cause. "
          "Also compare image mean values: a large gap means the live scene/pose "
          "is out-of-distribution (e.g. RESET_JOINTS pose unlike the demos). High "
          "live action std or saturation vs the demo indicates the policy is "
          "uncertain because the observation is off-distribution.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("export", help="Run on the VM that holds the demo pkl.")
    pe.add_argument("--demo_path", required=True)
    pe.add_argument("--ref_out", default="./demo_ref.npz")
    pe.add_argument("--n_samples", type=int, default=6)

    pc = sub.add_parser("compare", help="Run on the rollout box (robot + cameras).")
    pc.add_argument("--exp_name", default="task_towel_fold")
    pc.add_argument("--checkpoint_path", required=True)
    pc.add_argument("--eval_checkpoint_step", type=int, default=20000)
    pc.add_argument("--ref_in", default="./demo_ref.npz")
    pc.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.mode == "export":
        do_export(args)
    else:
        do_compare(args)


if __name__ == "__main__":
    main()
