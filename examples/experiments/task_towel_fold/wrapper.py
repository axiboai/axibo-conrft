"""Wrappers for the bimanual PiperX towel-folding task."""

import numpy as np
import gymnasium as gym


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
