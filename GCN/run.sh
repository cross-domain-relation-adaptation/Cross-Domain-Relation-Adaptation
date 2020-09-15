python train.py \
    --dataset citation_citeseer \
    --epoch_size 50 \
    --lr 2e-4 \
    --supervised False \
    --multiheads True \
    --hidden_c 128 \
    --out_c 64 \
    --co_coef 1e5 \
    --da_coef 5e1 \
    --gpus 0

