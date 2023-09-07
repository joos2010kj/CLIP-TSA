if [[ "$1" == "ucf" ]]; then
    # ucf
    python3 main.py --disable_wandb \
        --lr [0.001]*4000 \
        --batch_size 16 \
        --max_epoch 4000 \
        --k 0.95 \
        --dataset ucf \
        --seed 2 \
        --gpu "$2"
elif [[ "$1" == "xd" ]]; then
    # xd
    python3 main.py --disable_wandb \
        --lr [0.001]*4000 \
        --batch_size 16 \
        --max_epoch 4000 \
        --k 0.9 \
        --dataset xd \
        --seed 0 \
        --gpu "$2"
elif [[ "$1" == "sh" ]]; then
    # sh
    python3 main.py --disable_wandb \
        --lr [0.001]*35000 \
        --batch_size 16 \
        --max_epoch 35000 \
        --k 0.01 \
        --dataset sh \
        --seed 1 \
        --gpu "$2"
fi