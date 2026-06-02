export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
# LEARNER_IP = reachable address of the cloud learner (TCP 3333/3334 must be open).
# Override at call time:  LEARNER_IP=100.x.y.z bash run_actor_conrft.sh
LEARNER_IP=${LEARNER_IP:-localhost} && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=task_towel_fold \
    --ip=$LEARNER_IP \
    --checkpoint_path=$(pwd)/conrft \
    --actor \
    # --eval_checkpoint_step=26000 \
