#!/bin/bash

source ~/axo121/bin/activate
module load cuda/12.1

export HF_HOME=/tmp

accelerate launch ft_indiv.py llama3-8b.yml \
--learning_rate 2e-6 \
--wandb_project train-eff \
--output_dir /tmp/outmodel/ \
--num_epochs 3 \
--save_steps 1 \
--eval_steps 1 \
--wandb_name eff-$1-$2-$3 \
--task $1 \
--train_size $2 \
--seed $3 \
--num_gpus 1

base_folder="/tmp/outmodel/"
subfolder_pattern="checkpoint-*"
eval_folder="/tmp/evalmodel"

for subfolder in "$base_folder"/$subfolder_pattern; do
    if [ -d "$subfolder" ]; then
        mv "$subfolder" "$eval_folder"
    fi
done

python ../evaluation/hf_eval.py \
--model_dir $eval_folder \
--task_dir ../tasks/ \
--task_name $1 \
--save_dir ../notebooks/results/sample-efficiency/$1-$2-$3.json \
--context_size 8192 \
--max_samples 1000
