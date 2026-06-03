"""Wrappers for the bimanual PiperX towel-folding task."""

import time

import numpy as np
import gymnasium as gym


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """Compute the reward from camera images via a classifier function.

    Inlined from franka_env.envs.wrappers so the towel task does not depend on
    the Franka robot infra (serl_robot_infra / franka_env).
    """

    def __init__(self, env, reward_classifier_func, target_hz=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or (rew > 0.5)
        info["succeed"] = bool(rew > 0.5)
        if self.target_hz is not None:
            time.sleep(max(0, 1 / self.target_hz - (time.time() - start_time)))
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class PiperXIntervention(gym.ActionWrapper):
    """Human-in-the-loop intervention hook for HIL-ConRFT (Stage II).

    By default this is a no-op (online RL runs without human help). To enable
    interventions, wire your teleop device in ``get_teleop_action`` to return a
    14-D action (or ``None`` when not intervening). When an intervention action
    is returned it overrides the policy action and is flagged in ``info`` so the
    actor loop routes the transition into the demo/intervention buffer.
    """

    def __init__(self, env, action_indices=None, use_keyboard_toggle=True):
        super().__init__(env)
        self.action_indices = action_indices
        self.intervention_enabled = False
        self._listener = None
        if use_keyboard_toggle:
            try:
                from pynput import keyboard

                def on_press(key):
                    try:
                        if getattr(key, "char", None) == "i":
                            self.intervention_enabled = not self.intervention_enabled
                            print(
                                f"[intervention] {'ENABLED' if self.intervention_enabled else 'DISABLED'}"
                            )
                    except Exception:
                        pass

                self._listener = keyboard.Listener(on_press=on_press)
                self._listener.start()
            except Exception:
                # Keyboard toggle is optional; interventions can still be wired
                # externally by setting intervention_enabled.
                self._listener = None

    def _pack_leader_command_14(self, state_msg):
        left = state_msg["leader_left"]
        right = state_msg["leader_right"]
        out = np.zeros((14,), dtype=np.float32)
        out[:6] = np.asarray(left["q"][:6], dtype=np.float32)
        out[6] = float(left["gripper"])
        out[7:13] = np.asarray(right["q"][:6], dtype=np.float32)
        out[13] = float(right["gripper"])
        return out

    def get_teleop_action(self):
        if not self.intervention_enabled:
            return None
        state_msg = getattr(self.env.unwrapped, "latest_state_msg", None)
        if state_msg is None:
            return None
        try:
            return self._pack_leader_command_14(state_msg)
        except (KeyError, TypeError, ValueError):
            return None

    def close(self):
        if self._listener is not None:
            self._listener.stop()
        if hasattr(self.env, "close"):
            self.env.close()

    def __del__(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_abs = self.get_teleop_action()
        if expert_abs is not None:
            # Teleop stream provides absolute leader joint commands, while the
            # policy/env action space here is normalized deltas in [-1, 1].
            expert_abs = np.asarray(expert_abs, dtype=np.float32)
            curr = np.asarray(self.env.unwrapped.currpos, dtype=np.float32)
            scale = np.asarray(self.env.unwrapped.config.ACTION_SCALE, dtype=np.float32)
            safe_scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
            expert_a = (expert_abs - curr) / safe_scale
            expert_a = np.clip(expert_a, -1.0, 1.0)
            if self.action_indices is not None:
                filtered = np.asarray(action, dtype=np.float32).copy()
                filtered[self.action_indices] = expert_a[self.action_indices]
                expert_a = filtered
            self.intervened = True
            return expert_a
        self.intervened = False
        return action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if getattr(self, "intervened", False):
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
