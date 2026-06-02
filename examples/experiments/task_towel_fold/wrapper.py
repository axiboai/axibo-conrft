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

    def __init__(self, env, action_indices=None):
        super().__init__(env)
        self.action_indices = action_indices

    def get_teleop_action(self):
        # TODO(piperx): return a 14-D np.ndarray from your teleop device, or None.
        return None

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self.get_teleop_action()
        if expert_a is not None:
            expert_a = np.asarray(expert_a, dtype=np.float32)
            if self.action_indices is not None:
                filtered = action.copy()
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
