# Towel Folding (PiperX) — GPU Box Setup & Run Guide

End-to-end setup for running ConRFT on the bimanual PiperX towel-folding task,
from a fresh Linux + NVIDIA GPU machine. This consolidates every dependency and
the exact run order.

> Requires Linux + CUDA. JAX-GPU and the Octo fork do **not** work on Windows.

---

## 1. Conda environment + JAX (GPU)

```bash
conda create -n conrft python=3.10 -y
conda activate conrft

# GPU JAX (see the repo README for your CUDA version)
pip install --upgrade "jax[cuda11_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 2. Octo (custom fork) + serl_launcher

```bash
# Octo fork (provides the frozen VLA backbone)
git clone git@github.com:cccedric/octo.git
cd octo && pip install -e . && pip install -r requirements.txt && cd ..

# This repo's launcher
cd serl_launcher
pip install -e . && pip install -r requirements.txt && cd ..
```

## 3. Data-conversion dependencies

Only needed to turn the Hugging Face LeRobot datasets into ConRFT pickles:

```bash
pip install -r examples/requirements_convert.txt
```

## 4. Pretrained ResNet-10 weights (REQUIRED)

The reward classifier and the agent's image encoder load ImageNet-pretrained
ResNet-10 weights. The code expects the file at `examples/resnet10_params.pkl`
(referenced as `../resnet10_params.pkl` from a task folder):

```bash
cd examples
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
cd ..
```

## 5. Octo-small model

Download the `octo-small` model the fork uses, then point the config at it:

- Edit `examples/experiments/task_towel_fold/config.py` → `octo_path` to the
  local model directory (default placeholder: `/root/online_rl/octo_model/octo-small`).

## 6. Hugging Face auth (if the datasets are private)

```bash
huggingface-cli login
```

---

## 7. Reward classifier (offline)

```bash
cd examples
python convert_lerobot_to_conrft.py classifier \
    --repo_id Ishan-Axibo/piperx_towel_rollouts \
    --out_dir experiments/task_towel_fold/classifier_data

cd experiments/task_towel_fold
python ../../train_reward_classifier.py --exp_name task_towel_fold
# -> writes experiments/task_towel_fold/classifier_ckpt/
```

## 8. Stage I demos (offline, needs Octo for embeddings)

```bash
cd examples
python convert_lerobot_to_conrft.py demos \
    --repo_id Ishan-Axibo/piperx_flatten_depth \
    --out_dir experiments/task_towel_fold/demo_data \
    --octo_path /path/to/octo-small \
    --task_desc "pick towel from pile, fold and stack" \
    --control_hz 10 --max_len 200
```

Then set `--demo_path` in `run_learner_conrft_pretrain.sh` /
`run_learner_conrft.sh` to the produced `.pkl` filename.

## 9. Stage I (Cal-ConRFT) pretrain — first real training

```bash
cd examples/experiments/task_towel_fold
bash run_learner_conrft_pretrain.sh
```

## 10. Stage II (HIL-ConRFT) — online, needs the robot

Blocked until you provide:
- a PiperX robot HTTP server (`getstate` / `command` / `reset`, see
  `piperx_env.py` `# TODO(piperx)` hooks),
- teleop in `wrapper.py::PiperXIntervention.get_teleop_action`,
- real `RESET_JOINTS` / `ACTION_SCALE` / joint limits in `config.py`.

Then:

```bash
bash run_learner_conrft.sh   # terminal 1 (GPU)
bash run_actor_conrft.sh     # terminal 2 (robot host)
```

---

## Dependency checklist

| Item | Where | Needed for |
|------|-------|-----------|
| `jax[cuda]==0.4.20` | pip | all training |
| Octo fork (`cccedric/octo`) | `pip install -e .` | embeddings + actor encoder |
| `serl_launcher` | `pip install -e .` | all training |
| `av`, `pyarrow`, `opencv-python`, `huggingface_hub` | `requirements_convert.txt` | data conversion |
| `resnet10_params.pkl` | `examples/` | classifier + encoder |
| `octo-small` model | `octo_path` in config | embeddings + actor |
| PiperX server + teleop | you provide | Stage II only |
