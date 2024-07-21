# Load a task, and convert it to input / output
import sys
sys.path.append('.')
sys.path.append('../')
from eval.hf_eval import get_auto_evaluator, TaskEvaluator

import random
import json

MASK_TOKEN_ID = -100

class TaskInstructions(TaskEvaluator):
    def get_prompts(self):
        for example in self:
            target = example['target']
            
            # if output is a list, then randomly select one
            if isinstance(target, list):
                target = random.choice(target)

            # number of tokens in ground_truth
            n_tokens = len(self.tokenizer.encode(target, add_special_tokens=False))

            # construct prompt
            prompt_inputs = {k: example.pop(k) for k in self.keys_prompt}
            prompt, _ = self.build_prompt(
                tokenizer=self.tokenizer,
                context_size=self.context_size - n_tokens,
                **prompt_inputs,
            )
            
            yield prompt, target
    
    def get_tokenized(self, mask=True, add_eos=True):
        first = True
        for input, output in self.get_prompts():
            input_text = self.apply_conv_template([(input, None)])
            input_output_text = self.apply_conv_template([(input, output)])

            input_output_text = self.tokenizer.encode(input_output_text)
            if add_eos:
                input_output_text += [self.tokenizer.eos_token_id]
            assert len(input_output_text) <= self.context_size, f"Somehow label is too long: {len(input_output_text)}"

            label = input_output_text.copy()
            if mask:
                len_input_ids = len(self.tokenizer.encode(input_text))
                label[:len_input_ids] = [MASK_TOKEN_ID for _ in range(len_input_ids)]

            assert len(input_output_text) == len(label), f"Lengths don't match: {len(input_output_text)} vs {len(label)}"
            first_non_masked = next(i for i, x in enumerate(label) if x != MASK_TOKEN_ID)
            non_masked_tokens = label[first_non_masked:]
            decoded = self.tokenizer.decode(non_masked_tokens)
            if first:
                first = False
                print(input_text)
                print("non-masked tokens:", len(non_masked_tokens))
                print(self.tokenizer.tokenize(decoded))

            yield {'input_ids': input_output_text, 'labels': label, 'length': len(input_output_text)}


if __name__ == "__main__":
    import os
    import argparse
    import datasets
    import transformers

    parser = argparse.ArgumentParser()

    # task directory with the json files
    parser.add_argument('--task_dir', type=str, required=True)
    # tokenizer directory
    parser.add_argument('--tokenizer_dir', type=str, required=True)
    # name of the task (json file), if None, then all tasks in task_dir are used
    parser.add_argument('--task_name', type=str, default=None)
    # name of tokenizer (used for saving the dataset)
    parser.add_argument('--tokenizer_name', type=str, default=None)
    # directory to save the dataset
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--context_size', type=int, default=4096)
    # dataset split to tokenize
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default=None)
    # conversation template to use (defaults to Question: ... Answer: ...)
    parser.add_argument('--conv_template', type=str, default=None)
    # skips tasks that have already been processed
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()

    random.seed(args.seed)

    if args.tokenizer_name is None:
        tokenizer_name = args.tokenizer_dir
        if tokenizer_name.endswith('/'):
            tokenizer_name = tokenizer_name[:-1]
        if tokenizer_name.endswith('/snapshots/model'):
            tokenizer_name = tokenizer_name.split('/')[-3]
        else:
            tokenizer_name = tokenizer_name.split('/')[-1]
    else:
        tokenizer_name = args.tokenizer_name

    print(f"Parsing task {args.task_name} with tokenizer {tokenizer_name}...")

    import os
    files = os.listdir(args.task_dir)
    opinion_files = [f for f in files if f.endswith('opinions.json')]
    if args.task_name is None:
        tasks = [t[:-5] for t in files if t.endswith('.json') and t not in opinion_files]
    else:
        tasks = [args.task_name]
    
    print('Loading opinions...')
    opinions = {}
    for opinion_file in opinion_files:
        with open(f"{args.task_dir}/{opinion_file}", 'r') as f:
            opinions.update(json.load(f))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_dir, cache_dir="/tmp")

    dir_name = f"tok_{tokenizer_name}_{args.context_size}"

    splits = {'train': args.train_split}
    if args.val_split is not None:
        splits['val'] = args.val_split

    for task_name in tasks:
        if args.save_dir is not None:
            save_name = f"{args.save_dir}/{dir_name}/{task_name}"
            if os.path.exists(save_name) and not args.overwrite:
                print(f"Skipping task {task_name}...")
                continue
        else:
            save_name = None

        final_dataset = {}
        for split_name, split in splits.items():
            print(f"Processing task {task_name}...")
            evaluator = get_auto_evaluator(
                opinions=opinions,
                task_dir=f"{args.task_dir}{task_name}.json",
                eval_split=split,
                tokenizer=tokenizer,
                max_samples=args.max_samples,
            )

            task_instructions = TaskInstructions(
                evaluator=evaluator,
                conv_template=args.conv_template,
                tokenizer=tokenizer,
                context_size=args.context_size,
                verbose=True,
            )

            dataset = list(task_instructions.get_tokenized())
            dataset = datasets.Dataset.from_list(dataset)
            final_dataset[split_name] = dataset

        dataset = datasets.DatasetDict(final_dataset)
        if save_name is not None:
            dataset.save_to_disk(save_name)
