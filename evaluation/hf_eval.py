import json
import random
from tqdm import tqdm

from utils import load_tokenizer_model, return_logprobs_choices, greedy_decode
from utils import get_conv_template, build_prompt_task


def test_proper_labels(tokenizer, candidates):
    test1 = lambda x: f"\n{x}. This is a test"
    test2 = lambda x: f"\nAnswer:{x}"

    n1 = len(tokenizer.encode(test1('A')))
    n2 = len(tokenizer.encode("\nAnswer:"))
    for cand in candidates:
        assert len(tokenizer.encode(test1(cand))) == n1, f"Failed for {cand}"
        assert len(tokenizer.encode(test2(cand))) == n2 + 1, f"Failed for {cand}"

def get_vocab_cn_chr(tokenizer):
    def is_cn_char(char):
        """Check if a character is a Chinese character."""
        # Chinese characters Unicode range
        # Common Chinese characters (including Simplified and Traditional)
        if len(char) != 1:
            return False
        chinese_ranges = [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
            (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
            (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
            (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
            (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
            (0x2CEB0, 0x2EBEF),  # CJK Unified Ideographs Extension F
            (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
            (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
        ]
        # Convert char to Unicode code point
        code_point = ord(char)
        # Check if the code point falls within any of the Chinese ranges
        for start, end in chinese_ranges:
            if start <= code_point <= end:
                return True
        return False

    cn_char = []
    for i in range(len(tokenizer.get_vocab())):
        ch = tokenizer.decode(i)
        if is_cn_char(ch):
            cn_char.append(ch)

    return cn_char

def get_integers_choices(n, prefix=' '):
    # n is the total number of choices
    digits = len(str(n))  # number of digits needed
    return [prefix + str(i).zfill(digits) for i in range(1, n+1)]

def accuracy(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        correct += true == pred
    return correct / len(y_true)
    

class Evaluator:
    def __init__(self, opinions, task, eval_split='test', tokenizer=None,
                 max_samples=None, seed=42):
        self.opinions = opinions
        self.tokenizer = tokenizer

        self.task = task['task']
        self.question = self.task['question']
        self.instruction = self.task['instruction']
        self.fill_in = self.task.get('fill_in', [])  # list of keys
        self.examples = task['examples'][eval_split]

        # Shuffle the dataset
        random.seed(seed)
        random.shuffle(self.examples)

        if max_samples is not None:
            self.examples = self.examples[:max_samples]

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for example in self.examples:
            # Question and target
            qa = self(example)
            assert 'question' in qa, f"Question not found in {qa}"
            assert 'target' in qa, f"Target not found in {qa}"

            id_ = str(example['input'])
            yield {
                'id': id_,
                'opinion': self.opinions[id_],
                'instruction': self.get_instruction(example),
                **qa,
            }

    def __call__(self, example):
        raise NotImplementedError
    
    def fill(self, string, example):
        fills = {}
        for key in self.fill_in:
            if '{' + key + '}' in string:
                fills[key] = example[key]
        return string.format(**fills)
    
    def get_question(self, example):
        question = self.question
        if 'key' in example and type(self.question) == dict:
            question = self.question[example['key']]
        return self.fill(question, example)
    
    def get_instruction(self, example):
        instruction = self.instruction
        if 'key' in example and type(self.instruction) == dict:
            instruction = self.instruction[example['key']]
        return self.fill(instruction, example)
    

def get_choice2labels(choices, tokenizer=None, numbers=False):
    n = len(choices)
    # ' A', ' B', ' C', ...
    if n <= 26:
        candidates = [' ' + chr(65 + i) for i in range(26)]
    elif numbers:
        candidates = get_integers_choices(n)
    else:
        assert tokenizer is not None, "Tokenizer must be provided for CN chars"
        candidates = get_vocab_cn_chr(tokenizer)
    
    assert len(candidates) >= n
    candidates = candidates[:n]

    if tokenizer is not None and not numbers:
        test_proper_labels(tokenizer, candidates)
    
    # choice code -> choice label (e.g., A, B, C, ...)
    return {c: l for c, l in zip(choices, candidates)}


class MCEvaluator(Evaluator):
    """ Multiple choice questions """
    def __init__(self, opinions, task,
                 mc=True, rand_tgt=False, numbers=False, **kwargs):
        super().__init__(opinions, task, **kwargs)

        self.mc = mc  # whether to use multiple choice prompting
        self.rand_tgt = rand_tgt  # whether to randomize the target

        # choice code -> choice text
        self.choices = self.task['answer_choices']

        if self.mc:
            # if choices is a dict of dict...
            if type(list(self.choices.values())[0]) == dict:
                self.choice2label = {}
                for key, choices in self.choices.items():
                    self.choice2label[key] = get_choice2labels(choices, self.tokenizer, numbers)
            else:
                self.choice2label = get_choice2labels(self.choices, self.tokenizer, numbers)
            
    def __call__(self, example):
        question = self.get_question(example)
        choices = self.get_choices(example)
        target_code = example['target']

        if self.mc:
            choices2label = self.get_choices2label(example)
            return self.example_to_mc(question, choices, choices2label, target_code)
        return self.example_to_qa(question, choices, target_code)
    
    def get_choices(self, example):
        if 'choices' in example:
            return {c: self.choices[str(c)] for c in example['choices']}
        if 'key' in example and type(list(self.choices.values())[0]) == dict:
            return self.choices[example['key']]
        return self.choices
    
    def get_choices2label(self, example):
        if 'key' in example and type(list(self.choices.values())[0]) == dict:
            return self.choice2label[example['key']]
        return self.choice2label

    def example_to_mc(self, question, choices, choices2label, target_code):
        choice_texts = list(choices.values())
        choice_labels = [choices2label[c] for c in choices]

        question = f"Question: {question}\n"
        for label, text in zip(choice_labels, choice_texts):
            question += f"{label.strip()}. {text}\n"
        question += 'Answer:'  # added by the prompter?
        # question = question[:-1]  # remove last //

        if self.rand_tgt:
            target = random.choice(choice_labels)
        else:
            if type(target_code) == list:
                target = [choices2label[str(tc)] for tc in target_code]
            else:
                target = choices2label[str(target_code)]

        return {'question': question, 'target': target, 'choice_labels': choice_labels}
    
    def example_to_qa(self, question, choices, target_code):
        question = f"Question: {question}\nAnswer:"

        if self.rand_tgt:
            answer = random.choice(list(choices.values()))
        else:
            if type(target_code) == list:
                answer = [choices[str(tc)] for tc in target_code]
            else:
                answer = choices[str(target_code)]
        answer = ' ' + answer  # added by the prompter?

        return {'question': question, 'target': answer}
    

class IntEvaluator(Evaluator):
    """ Numerical questions expecting integer answers """
    def __call__(self, example):
        question = self.get_question(example)
        question = f"Question: {question}\nAnswer:"
        target = example['target']
        if type(target) == list:
            target = [' ' + str(t) for t in target]
        else:
            target = ' ' + str(target)
        return {'question': question, 'target': target}


class TaskEvaluator:
    def __init__(self, evaluator, conv_template=None, tokenizer=None, context_size=4096, 
                 verbose=False, skip_prev_fit=False, take_top=True):

        self.evaluator = evaluator
        self.context_size = context_size
        self.verbose = verbose
        self.skip_prev_fit = skip_prev_fit
        self.tokenizer = tokenizer

        apply_conv_template = get_conv_template(conv_template, tokenizer)
        def _build_prompt(question, instruction, opinion, tokenizer, context_size):
            text, fits = build_prompt_task(instruction, question, opinion, 
                                           tokenizer, apply_conv_template,
                                            context_size, take_top=take_top)
            return text, fits
        self.build_prompt = _build_prompt
        self.keys_prompt = ['question', 'instruction', 'opinion']
        self.apply_conv_template = apply_conv_template

    def keys_call(self):
        return []
    
    def __len__(self):
        return len(self.evaluator)
    
    def __iter__(self):
        iter_ = self.evaluator
        if self.verbose:
            iter_ = tqdm(iter_, total=len(iter_))
        yield from iter_

    def evaluate_dataset(self, model, tokenizer):
        results = []
        for inputs in self:
            prompt_inputs = {k: inputs.pop(k) for k in self.keys_prompt}

            if self.skip_prev_fit and self.would_fit(tokenizer, prompt_inputs):
                continue

            prompt, _ = self.build_prompt(
                tokenizer=tokenizer,
                context_size=self.context_size,
                **prompt_inputs,
            )

            prompt = self.apply_conv_template(prompt)
            call_inputs = {k: inputs.pop(k) for k in self.keys_call()}
            result = self(model, tokenizer, prompt, **call_inputs)

            result['id'] = inputs['id']
            result['ground_truth'] = inputs['target']
            result['prompt_len'] = len(tokenizer.encode(prompt))
            results.append(result)
            
            if self.verbose:
                self.print(results)

        metric = self.compute_metric(results)
        return metric, results
    
    def would_fit(self, tokenizer, prompt_inputs):
        _, would_fit = self.build_prompt(
            tokenizer=tokenizer,
            context_size=self.context_size // 2,
            **prompt_inputs,
        )
        return would_fit
    
    def print(self, results):
        result = results[-1]
        print(
            '------------------------------\n' \
            f'Input size: {result["prompt_len"]}\n' \
            f'Ground truth: {result["ground_truth"]}\n' \
            '------------------------------\n' \
            'Model response\n' \
            '------------------------------\n' \
            f'{result["model_response"]}\n\n' \
            f'Accuracy: {self.compute_metric(results)}'
        )

    def compute_metric(self, results):
        raise NotImplementedError
    

class MCTaskEvaluator(TaskEvaluator):
    def keys_call(self):
        return ['choice_labels']
    
    def __call__(self, model, tokenizer, prompt, choice_labels):
        answer, probs = return_logprobs_choices(prompt, choice_labels, 
                                                tokenizer, model, self.context_size)
        probs = {key: float(value) for key, value in probs.items()}
        return {'model_response': answer, 'logprobs': probs}

    def compute_metric(self, results):
        return accuracy([r['ground_truth'] for r in results],
                        [r['model_response'] for r in results])


class IntTaskEvaluator(TaskEvaluator):
    def __call__(self, model, tokenizer, prompt):
        answer = greedy_decode(model, tokenizer, prompt, max_gen=10)
        return {'model_response': answer}

    def compute_metric(self, results):
        def find_first_integer(input_string):
            num_str = ''
            for char in input_string.replace(',', ''):
                if char.isdigit():
                    num_str += char
                elif num_str:
                    return int(num_str)
        
        responses = [find_first_integer(r['model_response']) for r in results]
        return accuracy([int(r['ground_truth']) for r in results], responses)
    

def get_auto_evaluator(opinions, task_dir, **kwargs):
    with open(task_dir, 'r') as f:
        task = json.load(f)
    
    task_cfg = task['task']
    if 'answer_choices' in task_cfg and task_cfg['answer_choices'] is not None:
        return MCEvaluator(opinions, task, **kwargs)
    if task_cfg['type'] == 'int':
        if 'numbers' in kwargs:
            del kwargs['numbers']
        return IntEvaluator(opinions, task, **kwargs)
    raise ValueError("Task type not recognized")
    
def get_auto_task_evaluator(evaluator, **kwargs):
    if isinstance(evaluator, MCEvaluator):
        return MCTaskEvaluator(evaluator, **kwargs)
    if isinstance(evaluator, IntEvaluator):
        return IntTaskEvaluator(evaluator, **kwargs)
    raise ValueError("Evaluator type not recognized")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--task_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)

    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--context_size', type=int, default=4096)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)

    # use_qa - "Answer:" or otherwise a fastchat template (i.e., for instruction-tuned)
    parser.add_argument('--conv_template', type=str, default=None)  # e.g., mpt-7b-chat, 'template' for its own template
    parser.add_argument('--skip_prev_fit', action='store_true')
    parser.add_argument('--n_splits', type=int, default=1)
    parser.add_argument('--split_id', type=int, default=0)

    args = parser.parse_args()

    import os
    files = os.listdir(args.task_dir)
    opinion_files = [f for f in files if f.endswith('opinions.json')]
    if args.task_name is None:
        tasks = [t[:-5] for t in files if t.endswith('.json') and t not in opinion_files]

        if args.n_splits > 1:
            n = len(tasks)
            tasks = tasks[n * args.split_id // args.n_splits: n * (args.split_id + 1) // args.n_splits]

    else:
        tasks = [args.task_name]
    
    print('Loading opinions...')
    opinions = {}
    for opinion_file in opinion_files:
        with open(f"{args.task_dir}/{opinion_file}", 'r') as f:
            opinions.update(json.load(f))

    # Load the model
    import torch
    print('Loading model...', args.model_dir)

    tokenizer, model = load_tokenizer_model(
        args.model_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    for task in tqdm(tasks):
        print(f"Processing task {task}...")

        try:
            evaluator = get_auto_evaluator(
                task_dir=f"{args.task_dir}{task}.json",
                opinions=opinions,
                eval_split=args.eval_split,
                tokenizer=tokenizer,
                max_samples=args.max_samples,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        # Create the task evaluator
        print('Creating task evaluator...')
        task_evaluator = get_auto_task_evaluator(
            evaluator=evaluator,
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
        if args.save_dir is not None:
            file_name = f'{args.save_dir}/{task}.json'
            print(f'Saving results to {file_name}...')
            with open(file_name, 'w') as f:
                json.dump(results, f)
