export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python ../../train_conrft_octo.py "$@" \
    --exp_name=task_towel_fold \
    --checkpoint_path=$(pwd)/conrft \
    --q_weight=1.0 \
    --bc_weight=0.1 \
    --demo_path=./demo_data/towel_demos.pkl \
    --pretrain_steps=20000 \
    --debug=False \
    --learner \
