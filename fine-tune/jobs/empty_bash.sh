!/bin/bash

# Load conda environment
source /home/USER/axo121/bin/activate
module load cuda/12.1

export HF_HOME=/tmp

$@