# Towel Folding (bimanual PiperX) — ConRFT

This task adapts ConRFT to a bimanual PiperX robot using two Hugging Face
datasets:

- `Ishan-Axibo/piperx_flatten_depth` — clean successful expert demos (LeRobot
  v2.1). Used for **Stage I (Cal-ConRFT)** offline pretraining. (Depth columns
  are ignored; only the RGB cameras are used.)
- `Ishan-Axibo/piperx_towel_rollouts` — labelled rollouts whose per-episode
  `is_success` lives in `meta/episodes.jsonl`. Used to train the **reward
  classifier**.

> Runs on **JAX-GPU + the custom Octo fork** (Linux/CUDA). The data conversion
> step also needs `pip install av pyarrow opencv-python huggingface_hub`.

## Camera mapping

Octo-small consumes exactly two views, so the bimanual cameras are aliased:

| LeRobot key                          | ConRFT key         | Octo input      | size |
|--------------------------------------|--------------------|-----------------|------|
| `observation.images.cam_front`       | `side_policy_256`  | `image_primary` | 256² |
| `observation.images.cam_right_wrist` | `wrist_1`          | `image_wrist`   | 128² |

The left wrist view is unused by Octo (single primary + single wrist). Change
`--wrist_cam` in the converter to use the left wrist instead.

## Action / state

14-D bimanual joints (`left_joint_1..6`, `left_gripper`, `right_joint_1..6`,
`right_gripper`). `setup_mode = "bimanual-learned-gripper"` runs the
single-arm consistency-policy agent with `fix_gripper=False`, so the policy
outputs all 14 dimensions and no single-gripper grasp penalty is applied.

## Pipeline

### 1. Reward classifier data + training

```bash
cd examples
python convert_lerobot_to_conrft.py classifier \
    --repo_id Ishan-Axibo/piperx_towel_rollouts \
    --out_dir experiments/task_towel_fold/classifier_data

cd experiments/task_towel_fold
python ../../train_reward_classifier.py --exp_name task_towel_fold
```

Positives = final `--success_tail_frames` frames of `is_success` episodes;
negatives = failure episodes + non-terminal frames of successes.

### 2. Stage I demos (needs Octo + GPU for embeddings)

```bash
cd examples
python convert_lerobot_to_conrft.py demos \
    --repo_id Ishan-Axibo/piperx_flatten_depth \
    --out_dir experiments/task_towel_fold/demo_data \
    --octo_path /path/to/octo-small \
    --task_desc "pick towel from pile, fold and stack" \
    --control_hz 10 --max_len 200
```

Point `--demo_path` in the learner scripts at the produced `.pkl`.

### 3. Stage I pretrain (offline)

```bash
cd experiments/task_towel_fold
bash run_learner_conrft_pretrain.sh
```

### 4. Stage II online HIL-ConRFT (needs the real robot)

Start your PiperX robot server (see `piperx_env.py` for the required HTTP
endpoints and the `# TODO(piperx)` hardware hooks), then:

```bash
bash run_learner_conrft.sh   # terminal 1
bash run_actor_conrft.sh     # terminal 2
```

Wire your teleop device in `wrapper.py::PiperXIntervention.get_teleop_action`
to provide human interventions during online training.

## What you still must provide

- A PiperX robot HTTP server implementing `getstate` / `command` / `reset`.
- Teleop for HIL interventions (optional but recommended for Stage II).
- A trained `classifier_ckpt/` before launching the actor.
