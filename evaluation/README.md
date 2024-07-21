# Evaluation scripts

The purpose of these scripts is to evaluate language models for the different classification tasks under consideration. The primary script for this purpose is `hf_eval.py`, which can be called as follows:

```bash
python hf_eval.py \
    --model_dir meta-llama/Meta-Llama-3-8B-Instruct \
    --task_dir ../tasks/ \
    --task_name sc_issuearea \ 
    --save_dir model_responses/llama-3-8b-instruct \
    --eval_split test \
    --context_size 8192 \
    --verbose \
    --max_samples 1000
```

If task_name is not provided, then all tasks within `task_dir` are evaluated.
This script logs the model's responses. We then process these responses within the `notebooks` folder.
For BERT-stlye encoder models, use `bert_eval.py` instead.

#### GPT-4

We evaluate GPT-4 via Azure's OpenAI API. The relevant scripts are `gpt4_eval.py` and `gpt4_fewshot_eval.py`. To run the evaluations, make sure to fill the `openai_model` and `azure_kwargs` variables with your API user details. Then, the models are evaluated as follows:

```
python gpt4_eval.py --save_dir ../results/model_responses/gpt-4 --task_dir ../tasks/ --verbose
python gpt4_fewshot_eval.py --save_dir ../results/model_responses/gpt-4 --task_dir ../tasks/ --verbose
```

#### Job scripts

For our experiments, we use use an internal cluster with `htcondor`. You can see the specific job files in the `jobs/` folder, in particular:
    * `jobs_evaluate.py` - for the zero-shot and Lawma evaluations
    * `jobs_evaluate_scaling.py` - for the scaling experiments (e.g., the fine-tuned Pythia, Llama, etc.)
    * `jobs_evaluate_specialized.py` - for the specialization experiments