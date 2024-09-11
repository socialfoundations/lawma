import os
import datasets
import transformers

import sys
sys.path.append('../evaluation/')
from hf_eval_new import Evaluator


def read_task_txt(filename):
    tasks = []
    with open(filename) as f:
        for line in f:
            tasks.append(line.strip())
    return tasks

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, default="../datasets/")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_n_tokens", type=int, default=8000)
    parser.add_argument("--save_name", type=str, default="-8k")
    
    args = parser.parse_args()
    task_dir = args.task_dir

    # Only take opinions which have at least 2000 characters
    min_chars_opinion = 2000

    # Load all tasks as a dictionary
    task_dict = {}
    tasks = sorted(os.listdir(task_dir))
    for task in tasks:
        task_dict[task] = datasets.load_from_disk(task_dir + task)['test']

    # Examples from Supreme Court and Songer Court of Appeals
    print("Loading SC and Songer tasks")
    sc_tasks = [dset for task, dset in task_dict.items() if task.startswith('sc_')]
    songer_tasks = [dset for task, dset in task_dict.items() if task.startswith('songer_')]

    sc_examples = datasets.concatenate_datasets(sc_tasks)
    songer_examples = datasets.concatenate_datasets(songer_tasks)

    sc_examples = sc_examples.filter(lambda x: len(x['opinion']) >= min_chars_opinion)
    songer_examples = songer_examples.filter(lambda x: len(x['opinion']) >= min_chars_opinion)

    sc_examples = sc_examples.shuffle(seed=0).select(range(5000))
    songer_examples = songer_examples.shuffle(seed=0).select(range(5000))

    # Tiny tasks
    print("Loading tiny tasks")
    tiny_tasks = read_task_txt('tiny_tasks.txt')
    tiny_tasks = [dset for task, dset in task_dict.items() if task in tiny_tasks]
    tiny_examples = datasets.concatenate_datasets(tiny_tasks)

    tiny_examples = tiny_examples.filter(lambda x: len(x['opinion']) >= min_chars_opinion)
    tiny_examples = tiny_examples.shuffle(seed=0)

    # Hard tasks
    print("Loading hard tasks")
    target_size = 5000  # number of examples
    hard_tasks = read_task_txt('hard_tasks.txt')

    hard_tasks = [dset for task, dset in task_dict.items() if task in hard_tasks]
    hard_tasks = [task.filter(lambda x: len(x['opinion']) >= min_chars_opinion) for task in hard_tasks]
    hard_tasks = sorted(hard_tasks, key=lambda x: len(x))

    # Prefer examples from tasks with fewer examples
    total_examples = 0
    hard_examples = []
    for i in range(len(tasks)):
        if total_examples > target_size:
            break

        n = len(hard_tasks[i])
        if n == 0:
            continue
        
        for j in range(i, len(hard_tasks)):
            task = hard_tasks[j].shuffle(seed=0)

            pop_indices = range(n)
            keep_indices = range(n, len(task))
            
            hard_examples.append(task.select(pop_indices))
            hard_tasks[j] = task.select(list(keep_indices))

            total_examples += n

    hard_examples = datasets.concatenate_datasets(hard_examples)
    hard_examples = hard_examples.shuffle(seed=0).select(range(target_size))

    benchmark_datasets = datasets.DatasetDict({
        'sc': sc_examples,
        'songer': songer_examples,
        'tiny': tiny_examples,
        'hard': hard_examples,
    })
    columns = sc_examples.column_names

    # Use Llama 3 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    # Shorten the court opinions such that they are at most max_n_tokens tokens long
    print("Shortening datasets")
    shortened_datasets = {}
    for name, dataset in benchmark_datasets.items():
        short_dataset = Evaluator(
            task=dataset,
            tokenizer=tokenizer,
            context_size=args.max_n_tokens,
        ).get_dataset()

        short_dataset = short_dataset.remove_columns([col for col in short_dataset.column_names if col not in columns])
        shortened_datasets[name] = short_dataset

    # Push to HF hub
    print("Pushing to HF hub")
    for name, dataset in shortened_datasets.items():
        dataset_dict = datasets.DatasetDict({"test": dataset})
        dataset_dict.push_to_hub(f'ricdomolm/caselawqa{args.save_name}', config_name=name)
