# Stage II (HIL-ConRFT) — Split Deployment

Run the **learner on the cloud GPU VM** and the **actor on the rollout box** next
to the PiperX. They communicate over TCP via `agentlace`.

```
  CLOUD VM (GPU)                         ROLLOUT BOX (near robot)
  ─────────────                          ────────────────────────
  learner  ── publish weights ─────────► actor (policy inference)
  TrainerServer :3333/:3334  ◄── transitions ── TrainerClient
        ▲                                          │ HTTP
        │ loads Stage I checkpoint                 ▼
        └─ keeps training                     PiperX robot server (SERVER_URL)
```

## Who does what

- **Cloud learner**: all gradient updates. Loads the Stage I checkpoint, then
  publishes the initial weights to the actor and keeps training as data arrives.
- **Rollout actor**: runs the consistency policy **locally** at the control rate,
  talks to the robot over local HTTP, streams transitions up, and swaps in fresh
  weights as they arrive. The actor does **no training**.

### Latency note (important)
The 10 Hz control loop is **not** blocked by cloud round-trips: action sampling
(Octo + consistency policy) runs locally on the rollout box. The network only
carries (a) transitions up and (b) weight updates down, both asynchronous. So
cloud↔robot latency affects *how fresh* the policy weights are, not control
timing. The rollout box therefore needs enough compute (ideally a GPU) to run
Octo + the policy at 10 Hz.

## 1. Connectivity (pick one)

The actor must reach the learner on **TCP 3333 and 3334**.

| Option | When | Setup |
|--------|------|-------|
| **Tailscale / WireGuard VPN** (recommended) | cloud VM + robot on different networks | install on both; use the VM's tailnet IP as `LEARNER_IP` |
| **SSH reverse tunnel** | quick, no VPN | from the rollout box: `ssh -N -L 3333:localhost:3333 -L 3334:localhost:3334 user@vm` then use `LEARNER_IP=localhost` |
| **Public IP + open ports** | VM has a public IP you control | open 3333/3334 in the firewall/security group; use the public IP |

Verify from the rollout box before launching:

```bash
nc -vz $LEARNER_IP 3333
nc -vz $LEARNER_IP 3334
```

## 2. Cloud VM — start the learner

The learner's `--checkpoint_path` must point at the **Stage I pretrain output**
so it loads the pretrained policy and publishes it to the actor:

```bash
cd examples/experiments/task_towel_fold
bash run_learner_conrft.sh
```

It will fill the replay buffer from incoming actor data, then begin online
updates and `publish_network` every `steps_per_update` (50) steps.

## 3. Rollout box — prerequisites

Same code + models as the VM (so the agent definition matches):

```bash
git clone https://github.com/axiboai/axibo-conrft.git && cd axibo-conrft
# JAX (GPU strongly recommended for 10 Hz), Octo fork, serl_launcher  (see towel_fold_setup.md steps 1-2)
# resnet10_params.pkl in examples/  (step 3)
# set octo_path in config.py  (step 4)
```

Also required on this box:
- the **PiperX robot server** running (implements `getstate` / `command` /
  `reset`; see `piperx_env.py` `# TODO(piperx)`), with `SERVER_URL` in
  `EnvConfig` pointing at it (local, e.g. `http://127.0.0.1:5000/`),
- **teleop** wired in `wrapper.py::PiperXIntervention.get_teleop_action` for
  interventions,
- real `RESET_JOINTS` / `ACTION_SCALE` / joint limits in `config.py`.

## 4. Rollout box — start the actor

```bash
cd examples/experiments/task_towel_fold
LEARNER_IP=100.x.y.z bash run_actor_conrft.sh     # VM's reachable IP
```

The actor connects to the learner, receives the pretrained weights, and begins
rolling out. Your teleop interventions during rollout are routed to the
intervention buffer and reach the learner each episode (50/50 demo/online
sampling), so corrections feed the next weight updates.

## Notes
- `--checkpoint_path` on each side is independent and local: the learner writes
  checkpoints/buffers on the VM; the actor dumps its buffers locally. The actor
  gets weights over the network, not from the VM's checkpoint dir.
- Ports are fixed in `make_trainer_config()` (3333/3334). Change there if they
  collide with something on the VM.
- Keep the cloud↔robot link stable; if it drops, the actor keeps running on its
  last weights and reconnects, but no new weights arrive meanwhile.
