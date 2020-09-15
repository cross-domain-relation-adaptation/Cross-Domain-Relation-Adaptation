python train.py citeseer \
                --embedding_dim 64 \
                --batch_size 128 \
                --learning_rate 1e-3 \
                --learning_rate_bb 1e-3 \
                --co_coef 1.0 \
                --da_coef 1.0 \
                --use_multihead \
                --supervised \
		--n_trial 1

