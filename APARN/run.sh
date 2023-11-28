#!/bin/bash

# * laptop
# python ./APARN/train.py --model_name relationalbert --dataset laptop_amr --bert_lr 2e-5 --bert_dropout 0.1 --attn_dropout 0.4 --adam_beta 0.999 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --parseamr --seed 1000

# * restaurant
python ./APARN/train.py --model_name relationalbert --dataset restaurant_amr --bert_lr 4e-5 --bert_dropout 0.3 --attn_dropout 0.4 --adam_beta 0.999 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --parseamr --seed 1000

# * twitter
# python ./APARN/train.py --model_name relationalbert --dataset twitter_amr --bert_lr 1e-5 --bert_dropout 0.4 --attn_dropout 0.0 --adam_beta 0.99 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --parseamr --seed 1000

# * mams
# python ./APARN/train.py --model_name relationalbert --dataset mams_amr --bert_lr 1e-5 --bert_dropout 0.1 --attn_dropout 0.5 --adam_beta 0.98 --num_epoch 15 --hidden_dim 768 --max_length 100 --cuda 0 --parseamr --seed 1000

# Use --model_path to test an existing model