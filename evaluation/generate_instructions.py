import os
import numpy as np
import datasets

import sys
sys.path.append('../evaluation/')
from hf_eval_new import Evaluator


def random_choice(x):
    return x[np.random.choice(len(x))]

def process_dataset(all_datasets, tokenizer, context_size):
    dataset = datasets.concatenate_datasets(list(all_datasets.values()))
    dataset = Evaluator(
        task=dataset.shuffle(0),
        tokenizer=tokenizer,
        context_size=context_size,
    ).get_dataset()
    
    # only keep prompt and ground_truth columns
    col_to_remove = [col for col in dataset.column_names if col not in ['prompt', 'ground_truth', 'task']]
    dataset = dataset.remove_columns(col_to_remove)

    # turn ground truth from list of str to str by randomly selecting one
    dataset = dataset.map(lambda x: {'ground_truth': random_choice(x)}, input_columns='ground_truth')

    # rename following the Alpaca convention
    dataset = dataset.rename_column('prompt', 'instruction')
    dataset = dataset.rename_column('ground_truth', 'output')
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, default="../datasets/")
    parser.add_argument("--tokenizer", type=str, default=None)  # local path or HF model name
    parser.add_argument("--max_n_tokens", type=int, default=None)
    parser.add_argument("--save_name", type=str, default="")  # e.g., llama3-8k for 8192
    parser.add_argument("--n_val", type=int, default=1000)  # number of examples to keep for validation, rest for training
    parser.add_argument("--push", action="store_true")
    
    args = parser.parse_args()
    task_dir = args.task_dir

    if args.max_n_tokens is not None:
        assert args.tokenizer is not None, "Please provide a tokenizer if you want to limit the number of tokens"
    
    if args.max_n_tokens is not None:
        assert args.save_name != "", "Please provide a save name if you want to limit the number of tokens"

    tokenizer = None
    if args.tokenizer is not None:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    use_local = os.path.exists(task_dir)
    if use_local:
        tasks = sorted(os.listdir(task_dir))
    else:
        tasks = sorted(datasets.get_dataset_config_names('ricdomolm/lawma-tasks'))

    val_sets = {}
    train_sets = {}
    for task in tasks:
        if use_local:
            dataset = datasets.load_from_disk(task_dir + task)
        else:
            dataset = datasets.load_dataset('ricdomolm/lawma-tasks', task)

        val_sets[task] = dataset['val']
        train_sets[task] = dataset['train']

    # add task column to each dataset
    for task in train_sets:
        train_sets[task] = train_sets[task].add_column("task", [task] * len(train_sets[task]))
        val_sets[task] = val_sets[task].add_column("task", [task] * len(val_sets[task]))

    # process the datasets to get prompts (this is what takes the longest)
    val_dataset = process_dataset(val_sets, tokenizer, args.max_n_tokens)
    train_dataset = process_dataset(train_sets, tokenizer, args.max_n_tokens)

    # only keep `n` examples each from SC and Songer for validation, use the rest for training
    n = args.n_val // 2

    sc_val = val_dataset.filter(lambda x: x['task'].startswith('sc_'))
    songer_val = val_dataset.filter(lambda x: x['task'].startswith('songer_'))

    val_for_train = datasets.concatenate_datasets([
        sc_val.select(range(n, len(sc_val))),
        songer_val.select(range(n, len(songer_val)))
    ])

    val_dataset = datasets.concatenate_datasets([
        sc_val.select(range(n)),
        songer_val.select(range(n))
    ])

    train_dataset = datasets.concatenate_datasets([train_dataset, val_for_train])

    # save
    final_dataset = datasets.DatasetDict({
        'train': train_dataset.shuffle(0),
        'validation': val_dataset.shuffle(0),
    })

    if args.push:  # push to HF hub
        final_dataset.push_to_hub(f'ricdomolm/lawma-instructions{args.save_name}')
    else:  # save to disk
        final_dataset.save_to_disk(f"../lawma-instructions{args.save_name}")
    
    # print the first example of train
    print(train_dataset[0]['instruction'])
    print()
    print(train_dataset[0]['output'])
    print()