import time
import tiktoken
from openai import AzureOpenAI

import json
from tqdm import tqdm
import random
import itertools

from evaluation.hf_eval import MCTaskEvaluator, IntTaskEvaluator, get_auto_evaluator, TaskEvaluator

openai_model = 'USER-gpt-4-32k-0613'
azure_kwargs = dict(
    api_key='',  
    api_version='2024-02-01',
    azure_endpoint='https://openai-USER-3.openai.azure.com/'
)

def get_openai_completion(client, prompt):
    try:
        completion = client.chat.completions.create(
            model = openai_model,
            messages = [
                {'role': 'system', 'content': 'Please respond with a single letter or number.'},
                {'role': 'user', 'content': prompt},
            ],
            max_tokens=5,
            temperature=0,
            seed=0,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return str(e)


class TaskInstructions(TaskEvaluator):
    def get_prompts(self):
        for example in self:
            target = example['target']
            
            # if output is a list, then randomly select one
            if isinstance(target, list):
                target = random.choice(target)

            # number of tokens in ground_truth
            n_tokens = len(self.tokenizer.encode(target))

            # construct prompt
            prompt_inputs = {k: example.pop(k) for k in self.keys_prompt}
            prompt, _ = self.build_prompt(
                tokenizer=self.tokenizer,
                context_size=self.context_size - n_tokens,
                **prompt_inputs,
            )
            
            yield prompt, target

class OpenAIMCTaskEvaluator(MCTaskEvaluator):
    def __call__(self, model, tokenizer, prompt, choice_labels):
        time.sleep(0.1)  # to avoid rate limit
        answer = get_openai_completion(model, prompt)
        return {'model_response': answer}
    
class OpenAIIntTaskEvaluator(IntTaskEvaluator):
    def __call__(self, model, tokenizer, prompt):
        time.sleep(0.1)  # to avoid rate limit
        answer = get_openai_completion(model, prompt)
        return {'model_response': answer}


def construct_example(prompt, target):
    return prompt.strip() + ' ' + target.strip()

class FewShotMC(OpenAIMCTaskEvaluator):
    def __init__(self, task_instructions, n_fewshot, **kwargs):
        super().__init__(**kwargs)
        self.n_fewshot = n_fewshot
        self.task_instructions = itertools.cycle(iter(task_instructions.get_prompts()))

    def __call__(self, model, tokenizer, prompt, choice_labels):
        examples = []
        for _ in range(self.n_fewshot):
            ex_prompt, target = next(self.task_instructions)
            examples.append(construct_example(ex_prompt, target))

        examples.append(prompt)
        prompt = '\n\n'.join(examples)

        return super().__call__(model, tokenizer, prompt, choice_labels)
    
class FewShotInt(OpenAIIntTaskEvaluator):
    def __init__(self, task_instructions, n_fewshot, **kwargs):
        super().__init__(**kwargs)
        self.n_fewshot = n_fewshot
        self.task_instructions = iter(task_instructions.get_prompts())

    def __call__(self, model, tokenizer, prompt):
        examples = []
        for _ in range(self.n_fewshot):
            ex_prompt, target = next(self.task_instructions)
            examples.append(construct_example(ex_prompt, target))

        examples.append(prompt)
        prompt = '\n\n'.join(examples)

        return super().__call__(model, tokenizer, prompt)


def get_auto_task_evaluator(evaluator, **kwargs):
    from evaluation.hf_eval import MCEvaluator, IntEvaluator
    if isinstance(evaluator, MCEvaluator):
        return FewShotMC(evaluator=evaluator, **kwargs)
    if isinstance(evaluator, IntEvaluator):
        return FewShotInt(evaluator=evaluator, **kwargs)
    raise ValueError("Evaluator type not recognized")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--task_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default=None)

    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--context_size', type=int, default=4096*2-50)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)

    # use_qa - "Answer:" or otherwise a fastchat template (i.e., for instruction-tuned)
    parser.add_argument('--conv_template', type=str, default=None)  # e.g., mpt-7b-chat, 'template' for its own template
    parser.add_argument('--skip_prev_fit', action='store_true')

    # when evaluating using multiple jobs
    parser.add_argument('--n_splits', type=int, default=1)
    parser.add_argument('--split_id', type=int, default=0)
    
    args = parser.parse_args()

    import os
    files = os.listdir(args.task_dir)
    opinion_files = [f for f in files if f.endswith('opinions.json')]
    if args.task_name is None:
        import os
        tasks = os.listdir(args.task_dir)
        tasks = [t[:-5] for t in tasks if t.endswith('.json') and not t.endswith('opinions.json')]
    else:
        tasks = [args.task_name]
    
    print('Loading opinions...')
    opinions = {}
    for opinion_file in opinion_files:
        with open(f"{args.task_dir}/{opinion_file}", 'r') as f:
            opinions.update(json.load(f))

    tokenizer = tiktoken.encoding_for_model("gpt-4")

    model = AzureOpenAI(
        **azure_kwargs,
    )

    special_tasks = {
        'sc_issuearea': 'SC Issue Area',
        'sc_issue': 'SC Issue',
        'sc_decisiondirection': 'SC Direction',
        'sc_casedisposition': 'SC Disposition',
        'sc_casesource': 'SC Case Source',
        'sc_lcdisposition': 'SC LC Disposition',
        'songer_geniss': 'Songer Gen. Issue',
        'songer_casetyp1': 'Songer Case Type',
        'songer_direct1': 'Songer Direction',
        'songer_treat': 'Songer Treatment',
        'songer_origin': 'Songer Case Origin',
        'sc_lcdispositiondirection': 'SC LC Direction',
    }

    for task in tqdm(tasks):
        print(f"Processing task {task}...")

        file_name = f'{args.save_dir}/{task}.json'
        if os.path.exists(file_name):
            print(f'{file_name} already exists. Skipping...')
            continue

        if args.max_samples is not None:
            max_samples = args.max_samples
        elif task in special_tasks:
            max_samples = 5
        else:
            max_samples = 5

        evaluator_kwargs = {
            'task_dir': f"{args.task_dir}{task}.json",
            'opinions': opinions,
            'tokenizer': tokenizer,
            'numbers': True,
        }

        try:
            mc_evaluator = get_auto_evaluator(
                eval_split=args.eval_split,
                max_samples=max_samples,
                **evaluator_kwargs,
            )
        except Exception as e:
            print(f"Error in task {task}: {e}")
            continue
        
        task_kwargs = {
            'conv_template': args.conv_template,
            'tokenizer': tokenizer,
            'context_size': args.context_size,
            'verbose': args.verbose,
            'skip_prev_fit': False,
        }

        train_evaluator = get_auto_evaluator(
            eval_split='train',
            **evaluator_kwargs,
        )
        task_instructions = TaskInstructions(
            evaluator=train_evaluator,
            **task_kwargs,
        )
       
        # Create the task evaluator
        print('Creating task evaluator...')
        task_evaluator = get_auto_task_evaluator(
            evaluator=mc_evaluator,
            task_instructions=task_instructions,
            n_fewshot=3,
            **task_kwargs,
        )

        # Evaluate the dataset
        print('Evaluating the dataset...')
        _, results = task_evaluator.evaluate_dataset(model, tokenizer)

        # save results as a json file
        print(f'Saving results to {file_name}...')
        with open(file_name, 'w') as f:
            json.dump(results, f)
