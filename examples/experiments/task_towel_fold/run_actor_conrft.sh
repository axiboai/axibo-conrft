export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=task_towel_fold \
    --checkpoint_path=$(pwd)/conrft \
    --actor \
    # --eval_checkpoint_step=26000 \
