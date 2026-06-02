"""Sanity-check pickles produced by convert_lerobot_to_conrft.py.

Pure numpy + pickle (no JAX), so it runs anywhere. Validates keys, shapes and
dtypes against what the ConRFT learner / reward classifier expect, and prints
summary stats so you can catch conversion bugs before spending GPU time.

Usage
-----
  python check_converted_data.py demos       --path experiments/task_towel_fold/demo_data/towel_demos_*.pkl
  python check_converted_data.py classifier  --path experiments/task_towel_fold/classifier_data
"""

import os
import glob
import pickle as pkl
import argparse

import numpy as np


PRIMARY_KEY = "side_policy_256"
WRIST_KEY = "wrist_1"

EXPECTED_DEMO_KEYS = {
    "observations", "actions", "next_observations",
    "rewards", "masks", "dones", "mc_returns",
}
EMBED_KEYS = {"embeddings", "next_embeddings"}


def _load(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def _expand(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.pkl")))
    return sorted(glob.glob(path))


def _shape(x):
    return getattr(np.asarray(x), "shape", None)


def _check_obs(obs, label, problems, horizon=2):
    for k in ("state", PRIMARY_KEY, WRIST_KEY):
        if k not in obs:
            problems.append(f"{label}: missing observation key '{k}'")
    for k, img in ((PRIMARY_KEY, obs.get(PRIMARY_KEY)), (WRIST_KEY, obs.get(WRIST_KEY))):
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 4 or arr.shape[0] != horizon or arr.shape[-1] != 3:
            problems.append(f"{label}: '{k}' shape {arr.shape} (expected ({horizon},H,W,3))")
        if arr.dtype != np.uint8:
            problems.append(f"{label}: '{k}' dtype {arr.dtype} (expected uint8)")


def check_demos(paths):
    problems, total = [], 0
    rewards, mc, n_episodes, embed_dims, action_dims = [], [], 0, set(), set()

    for path in paths:
        data = _load(path)
        if not isinstance(data, list) or not data:
            problems.append(f"{path}: not a non-empty list of transitions")
            continue
        print(f"\n== {os.path.basename(path)}: {len(data)} transitions ==")
        t0 = data[0]
        missing = EXPECTED_DEMO_KEYS - set(t0.keys())
        if missing:
            problems.append(f"{path}: missing transition keys {sorted(missing)}")
        has_embed = EMBED_KEYS.issubset(t0.keys())
        if not has_embed:
            problems.append(f"{path}: no Octo embeddings (run convert with --octo_path; "
                            f"the learner needs them)")

        for i, tr in enumerate(data):
            total += 1
            label = f"{os.path.basename(path)}#{i}"
            _check_obs(tr.get("observations", {}), label + ".obs", problems)
            _check_obs(tr.get("next_observations", {}), label + ".next_obs", problems)
            action_dims.add(_shape(tr.get("actions")))
            rewards.append(float(np.asarray(tr.get("rewards", np.nan))))
            if "mc_returns" in tr:
                mc.append(float(np.asarray(tr["mc_returns"])))
            if bool(np.asarray(tr.get("dones", False))):
                n_episodes += 1
            if has_embed:
                embed_dims.add(_shape(tr.get("embeddings")))

    print("\n---- demo summary ----")
    print(f"transitions: {total}   episodes (dones): {n_episodes}")
    print(f"action shapes seen: {action_dims}")
    print(f"embedding shapes seen: {embed_dims or 'NONE'}")
    if rewards:
        r = np.asarray(rewards)
        print(f"reward: min={r.min():.3f} max={r.max():.3f} "
              f"frac_positive={(r > 0).mean():.3f}")
    if mc:
        m = np.asarray(mc)
        print(f"mc_returns: min={m.min():.3f} max={m.max():.3f} mean={m.mean():.3f}")
    _report(problems)


def check_classifier(paths):
    problems = []
    pos = neg = 0
    for path in paths:
        data = _load(path)
        is_pos = "success" in os.path.basename(path).lower()
        if not isinstance(data, list):
            problems.append(f"{path}: not a list")
            continue
        for i, tr in enumerate(data):
            if "observations" not in tr:
                problems.append(f"{os.path.basename(path)}#{i}: no 'observations'")
                continue
            _check_obs(tr["observations"], f"{os.path.basename(path)}#{i}", problems)
        if is_pos:
            pos += len(data)
        else:
            neg += len(data)
        print(f"{os.path.basename(path)}: {len(data)} frames "
              f"({'positive' if is_pos else 'negative'})")

    print("\n---- classifier summary ----")
    print(f"positives: {pos}   negatives: {neg}")
    if pos == 0 or neg == 0:
        problems.append("need BOTH positive and negative frames to train the classifier")
    elif neg < pos:
        print("note: fewer negatives than positives; 2-3x more negatives is recommended")
    _report(problems)


def _report(problems):
    if problems:
        print(f"\n\033[91mFOUND {len(problems)} ISSUE(S):\033[00m")
        for p in problems[:50]:
            print(f"  - {p}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        raise SystemExit(1)
    print("\n\033[92mAll checks passed.\033[00m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["demos", "classifier"])
    ap.add_argument("--path", required=True,
                    help="A .pkl file, a glob, or a directory of .pkl files.")
    args = ap.parse_args()

    paths = _expand(args.path)
    if not paths:
        raise SystemExit(f"No .pkl files matched: {args.path}")

    if args.mode == "demos":
        check_demos(paths)
    else:
        check_classifier(paths)


if __name__ == "__main__":
    main()
