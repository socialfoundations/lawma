# Generate the legal classification tasks from the Supreme Court and Songer databases

You can download the task files directly by using

```bash
wget -qO- https://huggingface.co/datasets/ricdomolm/lawma-task-files/resolve/main/tasks.tar.gz | tar -xz -C ../
```

The tasks will be saved in `../tasks/`.

Alternatively, you can generate the task files yourself by following these steps. First, download the court opinions of each court case in the SC/Songer databases:

```bash
pip install pyreadstat pandas tqdm
python download_data.py --sc --songer --save_dir caselaw/
```

This saves the `caselaw_sc.jsonl` and `caselaw_sc.jsonl` files. Then, generate the task files for each of the legal classification tasks:

```bash
python generate_scdb_tasks.py --data_file caselaw/caselaw_sc.jsonl --save_dir ../tasks/
python generate_songer_tasks.py --data_file caselaw/caselaw_songer.jsonl --save_dir ../tasks/
```

The output task files can be used to benchmark LLMs (see the evaluation folder). They can also be formatted and tokenized for supervised fine-tuning.

```bash
python tasks2instructions.py --task_dir ../tasks/ --tokenizer_dir meta-llama/Meta-Llama-3-8B --tokenizer_name llama3_8k --context_size 8192 --save_dir instructions/ --val_split val
```

We used the job file `jobs/jobs_tasks2instructions.py` to parallelize tokenization in our internal cluster.
