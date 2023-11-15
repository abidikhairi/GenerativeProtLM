#!/bin/bash

set -x

python main.py --d-model 512 --nhead 4 --dropout 0.6 --dim-ff 256 --num-layers 4 \
    --learning-rate 0.003 --betas 0.9 0.998 --warmup-steps 500 --batch-size 16 --num-workers 3 \
    --training-file data/refseq/human_proteome_training.txt \
    --validation-file data/refseq/human_proteome_validation.txt \
    --tokenizer-path models/tokenizer \
    --checkpoint-path models/checkpoints
