#!/bin/bash

source ~/axo121/bin/activate
module load cuda/12.1

export HF_HOME=/tmp

accelerate launch ft_indiv.py llama3-8b.yml \
--base_model $2 \
--learning_rate 2e-6 \
--wandb_project train-indiv \
--output_dir /tmp/outmodel/ \
--num_epochs 20 \
--eval_steps 1 \
--save_steps 1 \
--wandb_name ft-$3-$1 \
--task $1 \
--warmup_steps 50

python ../evaluation/hf_eval.py \
--model_dir /tmp/outmodel/ \
--task_dir ../tasks/ \
--task_name $1 \
--save_dir ../notebooks/results/specialization/$3/$1 \
--context_size 8192

find /tmp/outmodel/ -type d -name 'checkpoint*' -exec rm -r {} \;
mv /tmp/outmodel ../models/lawma-specialization-saves/$3/$1