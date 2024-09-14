import os
import datasets
import transformers

import sys
sys.path.append('../evaluation/')
from hf_eval_new import Evaluator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, default="../datasets/")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_n_tokens", type=int, default=8000)
    parser.add_argument("--save_name", type=str, default="-8k")
    parser.add_argument("--push", action="store_true")
    
    args = parser.parse_args()
    task_dir = args.task_dir

    # Only take opinions which have at least 2000 characters, and 1 answer
    min_chars_opinion = 2000
    def filter_func(x):
        return (len(x['opinion']) >= min_chars_opinion) and (len(x['answer']) == 1)

    # Load all tasks as a dictionary
    task_dict = {}
    if os.path.exists(task_dir):
        tasks = sorted(os.listdir(task_dir))
        for task in tasks:
            task_dict[task] = datasets.load_from_disk(task_dir + task)['test']
    else:  # load from HF hub
        print(f"Loading tasks from HF hub ({task_dir} does not exist)")
        tasks = datasets.get_dataset_config_names('ricdomolm/lawma-tasks')
        for task in tasks:
            task_dict[task] = datasets.load_dataset('ricdomolm/lawma-tasks', task, split='test')

    # Examples from Supreme Court and Songer Court of Appeals
    print("Loading SC and Songer tasks")
    sc_tasks = [dset for task, dset in task_dict.items() if task.startswith('sc_')]
    songer_tasks = [dset for task, dset in task_dict.items() if task.startswith('songer_')]

    sc_examples = datasets.concatenate_datasets(sc_tasks)
    songer_examples = datasets.concatenate_datasets(songer_tasks)

    sc_examples = sc_examples.filter(filter_func)
    songer_examples = songer_examples.filter(filter_func)

    sc_examples = sc_examples.shuffle(seed=0).select(range(5000))
    songer_examples = songer_examples.shuffle(seed=0).select(range(5000))

    benchmark_datasets = datasets.DatasetDict({
        'sc': sc_examples,
        'songer': songer_examples,
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

    for name, dataset in shortened_datasets.items():
        dataset_dict = datasets.DatasetDict({"test": dataset})
        if args.push:  # push to HF hub
            dataset_dict.push_to_hub(f'ricdomolm/caselawqa{args.save_name}', config_name=name)
        else:  # save to disk
            dataset_dict.save_to_disk(f"caselawqa{args.save_name}_{name}")
    
    # print the first example of each dataset
    for name, dataset in shortened_datasets.items():
        print(name)
        print(dataset[0])
        print()