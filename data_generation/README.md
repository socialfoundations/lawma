# Generating classification tasks from the Supreme Court and Songer databases

Here we provide the steps to generate the classification tasks considered.
We include these tasks in the `../tasks/` folder as `.jsonl` files.

## Download the opinions corresponding to each Supreme Court case in the SC/Songer databases

Follow the instructions in `download_sc_data.py` and `download_songer_data` in order to download the
required files from the SC/Songer databases, as well as the relevant Caselaw Project metadata.

Then, run

```
python downlooad_sc_data.py --scdb_file scdb_labels.csv --metadata_file caselaw_us_metadata.jsonl --save_file caselaw_sc.jsonl
python downlooad_songer_data.py --songer_file cta96_stata.dta --metadata_file f2d_metadata.jsonl --save_file caselaw_songer.jsonl
```

in order to download the court opinions corresponding to the court cases in the SC/Songer databases.
Of course, adapt the file paths to wherever you downloaded the relevant files. 

## Generate the feature extraction tasks

```
mkdir tasks
python generate_scdb_tasks.py --data_file caselaw_sc.jsonl --save_dir ../tasks/
python generate_songer_tasks.py --data_file caselaw_songer.jsonl --save_dir ../tasks/
```

The output task files can be used to benchmark LLMs on the feature extraction tasks, or
tokenized for fine-tuning.

## Tokenize the task data

In order to fine-tune on the task data, it must first be tokenized as follows:

```
mkdir instructions
python tasks2instructions.py --task_dir ../tasks/ --tokenizer_dir meta-llama/Meta-Llama-3-8B --tokenizer_name llama3_8k --context_size 8192 --save_dir instructions/ --val_split val
```

We used the job file `jobs/jobs_tasks2instructions.py` to tokenize the model for the different language models considered in our work. 