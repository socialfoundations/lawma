import re
import json
import numpy as np
from tqdm import tqdm

import datasets

from utils import load_tokenizer_model, return_logprobs_choices, greedy_decode


def get_choice_labels(choices):
    n_choices = len(choices)
    if n_choices < 26: # A, B, C, ...
        return [chr(65 + i) for i in range(n_choices)]
    n_digits = len(str(n_choices))
    return [str(i+1).zfill(n_digits) for i in range(n_choices)]


def choices_to_text(choices, choice_labels):
    return '\n'.join([f"{label.strip()}. {text.strip()}" for label, text in zip(choice_labels, choices)])


def get_choices_text_answer(choices, answer):
    if len(choices) == 0:
        return '', [' ' + str(a).strip() for a in answer], None
    choice_labels = get_choice_labels(choices)
    choices_text = choices_to_text(choices, choice_labels)
    choice_labels = [' ' + label for label in choice_labels]
    target = [choice_labels[i] for i in answer]
    return choices_text, target, choice_labels


def get_question_target(choices, answer, question):
    choices_text, target, choice_labels = get_choices_text_answer(choices, answer)
    question = f"Question: {question.strip()}\n{choices_text}\nAnswer:"
    return question, target, choice_labels


def construct_prompt(instruction, opinion, question):
    return f"{instruction}\n\n{opinion}\n\n{question}"


def shorten_opinion_batch(top, bot, input_text, tokenizer, context_size, headroom=10):
    output_text = ['' for _ in range(len(input_text))]

    # We tokenize top + bot, and calculate the remaining number of tokens for the input text
    top_bot = [t + b for t, b in zip(top, bot)]
    prompt_template_len = [len(input_ids) for input_ids in tokenizer(top_bot).input_ids]
    remaining_tokens = [context_size - ptl - headroom for ptl in prompt_template_len]  # some safety margin

    # Process those for which some amount of input_text fits
    ids_fit = {i for i, rt in enumerate(remaining_tokens) if rt > 0}

    if len(ids_fit) == 0:
        return output_text

    input_text = [input_text[i] for i in ids_fit]
    remaining_tokens = [remaining_tokens[i] for i in ids_fit]
    tokenized_body = tokenizer(input_text).input_ids

    # Some of those need to be shortened
    need_to_shorten = [len(tb) > rt for tb, rt in zip(tokenized_body, remaining_tokens)]
    to_decode = [tb[:rt] for tb, rt, ns in zip(tokenized_body, remaining_tokens, need_to_shorten) if ns]
    if len(to_decode) > 0:
        decoded = tokenizer.batch_decode(to_decode, skip_special_tokens=True)

    # Reconstruct the output_text
    for i, id in enumerate(ids_fit):
        output_text[id] = input_text[i] if not need_to_shorten[i] else decoded.pop(0)
    
    return output_text


class Evaluator:
    def __init__(self, task, tokenizer=None, context_size=None, encode_batch_size=1000, verbose=False):
        """
        Takes in an iterator over 'opinion', 'instruction', 'question', 'choices', 'answer' and
        returns an iterator over 'prompt', 'ground_truth'
        """
        self.verbose = verbose

        if type(task) != datasets.Dataset:
            if verbose: print("Converting task to a dataset...")
            task = datasets.Dataset.from_list(list(task))

        if verbose: print('Constructing the question-target pairs...')
        def map_get_question_target(example):
            prompt_question, ground_truth, choice_labels = get_question_target(
                example['choices'], example['answer'], example['question']
            )
            return {
                'prompt_question': prompt_question, 
                'ground_truth': ground_truth, 
                'choice_labels': choice_labels
            }
        task = task.map(map_get_question_target)

        if tokenizer is not None and context_size is not None:
            if verbose: print('Shortening the opinions...')
            def batch_shorten(examples):
                tops = [inst+'\n\n' for inst in examples['instruction']]
                bots = ['\n\n'+q for q in examples['prompt_question']]
                return {'opinion': shorten_opinion_batch(tops, bots, examples['opinion'], tokenizer, context_size)}
            task = task.map(batch_shorten, batched=True, batch_size=encode_batch_size)

        self.task = task.map(lambda ex: {'prompt': construct_prompt(
            ex['instruction'], ex['opinion'], ex['prompt_question']
        )})

        self.keys_to_return = ['prompt', 'ground_truth', 'choice_labels']
    
    def __len__(self):
        return len(self.task)
    
    def __iter__(self):
        for example in self.task:
            yield self(example)

    def __call__(self, example):
        return {k: example[k] for k in self.keys_to_return}
    
    def get_dataset(self):
        return self.task


class TaskEvaluator:
    def __init__(self, evaluator, context_size, mc=False, verbose=False):
        self.evaluator = evaluator
        self.context_size = context_size
        self.mc = mc  # use multiple choice if possible
        self.verbose = verbose

    def evaluate_dataset(self, model, tokenizer):
        results = []
        for example in self.evaluator:
            prompt = example['prompt']
            target = example['ground_truth']
            choices = example['choice_labels']

            if choices is not None and self.mc:
                _, response = return_logprobs_choices(prompt, choices, tokenizer, model, self.context_size)
                response = {float(value) for value in response.values()}
            else:
                response = greedy_decode(model, tokenizer, prompt, max_gen=10)

            result = {
                'ground_truth': target,
                'model_response': response,
                'prompt_len': len(tokenizer.encode(prompt)),
            }

            results.append(result)

            if self.verbose:
                self.print(results)

        metric = self.compute_metric(results) if model is not None else None
        return metric, results
    
    def print(self, results):
        result = results[-1]
        print(
            '------------------------------\n' \
            f'Input size: {result["prompt_len"]}\n' \
            f'Ground truth: {result["ground_truth"]}\n' \
            '------------------------------'
        )

        if 'model_response' in result:
            print(
                'Model response\n' \
                '------------------------------\n' \
                f'{result["model_response"]}\n\n' \
                f'Accuracy: {self.compute_metric(results)}'
            )
    
    def compute_metric(self, results):
        y_pred = [r['model_response'] for r in results]
        y_true = [r['ground_truth'] for r in results]

        outcomes = []
        for yp, yt in zip(y_pred, y_true):
            any_match = False
            for yt_ in yt:
                yt_ = yt_.strip()
                if yt_.isdigit():  # find the first integer 
                    yt_ = int(yt_)
                    yp_ = int(re.search(r'\d+', yp).group())
                elif yt_.isupper():  # find the first capital letter
                    yt_ = yt_.strip()
                    yp_ = re.search(r'[A-Z]', yp).group()
                else:
                    yp_ = yp.strip()
                
                if yt_ == yp_:
                    any_match = True
                    break

            outcomes.append(any_match)
        
        return np.mean(outcomes)


if __name__ == "__main__":
    import os
    import torch

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--task_dir', type=str, default=None)  # None for HF Hub
    parser.add_argument('--save_dir', type=str, default=None)

    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--context_size', type=int, default=4096)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--mc', action='store_true')  # multiple choice

    args = parser.parse_args()

    # Load the dataset
    if args.task_dir is None:
        task_dataset = datasets.load_dataset('ricdomolm/lawma-tasks', args.task, split=args.eval_split)
    else:
        task_dataset = datasets.load_from_disk(f"{args.task_dir}/{args.task}")[args.eval_split]

    # Load the model
    print('Loading model...', args.model_dir)
    tokenizer, model = load_tokenizer_model(
        args.model_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    print('Creating the evaluator...')
    evaluator = Evaluator(
        task=task_dataset,
        tokenizer=tokenizer,
        context_size=args.context_size,
    )
    task_evaluator = TaskEvaluator(
        evaluator=evaluator,
        context_size=args.context_size,
        mc=args.mc,
        verbose=args.verbose,
    )

    # Evaluate the dataset
    print('Evaluating the dataset...')
    _, results = task_evaluator.evaluate_dataset(model, tokenizer)

    # save results as a json file
    if args.save_dir is not None:
        file_name = f'{args.save_dir}/{args.task}.json'
        print(f'Saving results to {file_name}...')
        with open(file_name, 'w') as f:
            json.dump(results, f)
