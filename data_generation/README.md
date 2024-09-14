# Generate the legal classification tasks from the Supreme Court and Songer databases

You can retrieve the legal classification tasks from [`ricdomolm/lawma-tasks`](https://huggingface.co/datasets/ricdomolm/lawma-tasks). For example, to retrieve the train split of the `sc_issuearea` task, use:

```python
import pandas
import datasets

task_data = datasets.load_dataset('ricdomolm/lawma-tasks', 'sc_issuearea', split='train')
task_data = pandas.DataFrame(task_data)
```

The output task files can be used to benchmark LLMs (see the evaluation folder). They can also be formatted and tokenized for supervised fine-tuning (see the fine-tune folder).

### From the task .jsonl files to HF datasets

You can download the `.jsonl` task files by using

```bash
wget -qO- https://huggingface.co/datasets/ricdomolm/lawma-task-files/resolve/main/tasks.tar.gz | tar -xz -C ../
```

The tasks will be saved in `../tasks/`. Then, convert them to parquet datasets 

```bash
pip install datasets
python loader.py --task_dir ../tasks/ --save_dir ../datasets/
```

### Generating the task .jsonl files

First, download the court opinions of each court case in the SC/Songer databases:

```bash
pip install pyreadstat pandas tqdm
python download_data.py --sc --songer --save_dir ../caselaw/
```

This saves the `caselaw_sc.jsonl` and `caselaw_sc.jsonl` files. Then, generate the task files for each of the legal classification tasks:

```bash
python generate_scdb_tasks.py --data_file ../caselaw/caselaw_sc.jsonl --save_dir ../tasks/
python generate_songer_tasks.py --data_file ../caselaw/caselaw_songer.jsonl --save_dir ../tasks/
```

We upload the task `.jsonl` files to the HF hub as follows
```bash
cp -r ../tasks ./
python upload_task_files_to_hub.py
rm -r tasks
```
