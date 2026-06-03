#!/usr/bin/env python3
"""Offline sanity check for policy outputs on converted demos.

Loads a trained checkpoint and evaluates action predictions on recorded demo
observations (no robot required). Useful to catch saturation or action-space
mismatches before live rollout.
"""

import argparse
import pickle as pkl

import jax
import numpy as np
from flax.training import checkpoints

from experiments.mappings import CONFIG_MAPPING
from octo.model.octo_model import OctoModel
from serl_launcher.utils.launcher import make_conrft_octo_cp_pixel_agent_single_arm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="task_towel_fold")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--demo_path", required=True)
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    config = CONFIG_MAPPING[args.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)

    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

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
        q_weight=0.1,
        bc_weight=1.0,
    )

    ckpt = checkpoints.restore_checkpoint(args.checkpoint_path, agent.state)
    agent = agent.replace(
        state=agent.state.replace(params=ckpt.params, target_params=ckpt.target_params)
    )

    with open(args.demo_path, "rb") as f:
        demos = pkl.load(f)
    assert len(demos) > 0, "Demo file is empty"

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(demos), size=min(args.num_samples, len(demos)), replace=False)

    pred_actions, gt_actions = [], []
    sample_rng = jax.random.PRNGKey(args.seed)
    for i in idx:
        tr = demos[int(i)]
        obs = tr["observations"]
        gt = np.asarray(tr["actions"], dtype=np.float32)
        sample_rng, key = jax.random.split(sample_rng)
        pred, _ = agent.sample_actions(
            observations=jax.device_put(obs),
            tasks=jax.device_put(tasks),
            seed=key,
            argmax=False,
        )
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)
        pred_actions.append(pred)
        gt_actions.append(gt)

    pred = np.stack(pred_actions)
    gt = np.stack(gt_actions)
    err = np.abs(pred - gt)

    print(f"samples: {len(pred)}")
    print(f"pred min/max: {pred.min():.4f} / {pred.max():.4f}")
    print(f"gt   min/max: {gt.min():.4f} / {gt.max():.4f}")
    print(f"mean |pred| per-dim: {np.mean(np.abs(pred), axis=0)}")
    print(f"mean |gt|   per-dim: {np.mean(np.abs(gt), axis=0)}")
    print(f"MAE mean: {err.mean():.4f}")
    print(f"MAE p50/p90/p95: {np.percentile(err, [50, 90, 95])}")
    print(f"pred saturation ratio (|a|>=0.95): {(np.abs(pred) >= 0.95).mean():.4f}")


if __name__ == "__main__":
    main()
