#!/bin/bash

VENV_PATH="/home/rolmedo/axo121/"
python -m venv $VENV_PATH
source "$VENV_PATH/bin/activate"

pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers
pip install accelerate
pip install deepspeed
pip install datasets

pip install git+https://github.com/RicardoDominguez/lm-evaluation-harness@sft

pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation