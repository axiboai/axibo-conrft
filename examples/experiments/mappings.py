CONFIG_MAPPING = {}

# The banana task depends on franka_env (serl_robot_infra). Import it lazily so
# tasks that don't need the Franka infra (e.g. task_towel_fold) still work when
# serl_robot_infra is not installed.
try:
    from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
    CONFIG_MAPPING["task1_pick_banana"] = PickBananaTrainConfig
except ImportError:
    pass

from experiments.task_towel_fold.config import TrainConfig as TowelFoldTrainConfig
CONFIG_MAPPING["task_towel_fold"] = TowelFoldTrainConfig
