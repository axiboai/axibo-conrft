import os
import jax
import numpy as np
import jax.numpy as jnp

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.task_towel_fold.piperx_env import PiperXEnv, PiperXEnvConfig
from experiments.task_towel_fold.wrapper import PiperXIntervention

from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper


class EnvConfig(PiperXEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    CAMERAS = {
        "side_policy_256": {"index": 0},
        "wrist_1": {"index": 1},
    }
    IMAGE_CROP = {}
    # Home configuration the robot resets to (absolute joint targets, 14-D).
    RESET_JOINTS = np.zeros((14,), dtype=np.float32)
    ACTION_SCALE = np.full((14,), 0.05, dtype=np.float32)
    JOINT_LIMIT_LOW = np.full((14,), -np.pi, dtype=np.float32)
    JOINT_LIMIT_HIGH = np.full((14,), np.pi, dtype=np.float32)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 200
    CONTROL_HZ = 10.0


class TrainConfig(DefaultTrainingConfig):
    # Octo uses side_policy_256 (image_primary) + wrist_1 (image_wrist).
    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = ["side_policy_256"]
    proprio_keys = ["joint_pos"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    # Bimanual 14-D action, two learned grippers, no single-gripper grasp penalty.
    setup_mode = "bimanual-learned-gripper"
    reward_neg = -0.05
    success_reward = 1.0
    classifier_threshold = 0.9
    task_desc = "pick towel from pile, fold and stack"
    octo_path = "/root/online_rl/octo_model/octo-small"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = PiperXEnv(fake_env=fake_env, save_video=save_video, config=EnvConfig())
        if not fake_env:
            env = PiperXIntervention(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        if classifier:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                def sigmoid(x):
                    return 1 / (1 + jnp.exp(-x))
                if sigmoid(classifier_func(obs)[0]) > self.classifier_threshold:
                    return self.success_reward
                return self.reward_neg

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
