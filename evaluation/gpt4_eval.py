import time
import tiktoken
from openai import AzureOpenAI
from tqdm import tqdm

from evaluation.hf_eval import MCTaskEvaluator, IntTaskEvaluator, get_auto_evaluator

openai_model = 'USER-gpt-4-0613'
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


class OpenAIMCTaskEvaluator(MCTaskEvaluator):
    def __call__(self, model, tokenizer, prompt, choice_labels):
        # to avoid geting rate limited, sleep for .1 seconds
        time.sleep(0.1)
        answer = get_openai_completion(model, prompt)
        return {'model_response': answer}
    
class OpenAIIntTaskEvaluator(IntTaskEvaluator):
    def __call__(self, model, tokenizer, prompt):
        answer = get_openai_completion(model, prompt)
        return {'model_response': answer}

def get_auto_task_evaluator(evaluator, **kwargs):
    from evaluation.hf_eval import MCEvaluator, IntEvaluator
    if isinstance(evaluator, MCEvaluator):
        return OpenAIMCTaskEvaluator(evaluator, **kwargs)
    if isinstance(evaluator, IntEvaluator):
        return OpenAIIntTaskEvaluator(evaluator, **kwargs)
    raise ValueError("Evaluator type not recognized")


if __name__ == '__main__':
    import json
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
        **azure_kwargs
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
            max_samples = 250
        else:
            max_samples = 100

        try:
            mc_evaluator = get_auto_evaluator(
                task_dir=f"{args.task_dir}{task}.json",
                opinions=opinions,
                eval_split=args.eval_split,
                tokenizer=tokenizer,
                max_samples=max_samples,
                numbers=True,
            )
        except Exception as e:
            print(f"Error in task {task}: {e}")
            continue

        # Create the task evaluator
        print('Creating task evaluator...')
        task_evaluator = get_auto_task_evaluator(
            evaluator=mc_evaluator,
            conv_template=args.conv_template,
            tokenizer=tokenizer,
            context_size=args.context_size,
            verbose=args.verbose,
            skip_prev_fit=args.skip_prev_fit,
        )

        # Evaluate the dataset
        print('Evaluating the dataset...')
        _, results = task_evaluator.evaluate_dataset(model, tokenizer)

        # save results as a json file
        print(f'Saving results to {file_name}...')
        with open(file_name, 'w') as f:
            json.dump(results, f)