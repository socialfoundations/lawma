# Generating classification tasks from the Supreme Court and Songer databases

Here we provide the steps to generate the legal classification tasks considered in our work.
We include these tasks in the `../tasks/` folder as `.jsonl` files.

## Download the court opinions of each court case in the SC/Songer databases

Follow the instructions in `download_sc_data.py` and `download_songer_data.py` in order to download the
required files from the SC/Songer databases, as well as the relevant Caselaw Project metadata.

Then, run

```
python download_sc_data.py --scdb_file scdb_labels.csv --metadata_file caselaw_us_metadata.jsonl --save_file caselaw_sc.jsonl
python download_songer_data.py --songer_file cta96_stata.dta --metadata_file f2d_metadata.jsonl --save_file caselaw_songer.jsonl
```

## Generate the legal classification tasks

```
mkdir ../tasks
python generate_scdb_tasks.py --data_file caselaw_sc.jsonl --save_dir ../tasks/
python generate_songer_tasks.py --data_file caselaw_songer.jsonl --save_dir ../tasks/
```

The output task files can be used to benchmark LLMs (see the evaluation folder). They can also be formatted and tokenized for supervised fine-tuning.

## Format and tokenize tasks for supervised fine-tuning

```
mkdir instructions
python tasks2instructions.py --task_dir ../tasks/ --tokenizer_dir meta-llama/Meta-Llama-3-8B --tokenizer_name llama3_8k --context_size 8192 --save_dir instructions/ --val_split val
```

We used the job file `jobs/jobs_tasks2instructions.py` to parallelize tokenization.
