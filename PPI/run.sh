python train.py \
        --bb tape \
        --lr 2e-4 \
        --bb_lr 1e-6 \
        --co_coef 1.0 \
        --da_coef 1.0 \
        --max_length 500 \
        --batch_size 3 \
        --epoch_size 5000 \
        --batch_size_val 45 \
        --epoch_size_val 9000 \
        --data_root ./STRING \
	--checkpoint_callback 0

