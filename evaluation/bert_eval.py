# These script is used to evaluate a BERT-style models, namely LegalBert.

import torch
from tqdm import tqdm

from evaluation.hf_eval import MCTaskEvaluator, IntTaskEvaluator, get_auto_evaluator

def query_model_bert(text_input, tokenizer, model, context_size):
    # print(text_input)
    input_ids = tokenizer.encode(text_input, return_tensors='pt').to(model.device)
    assert input_ids.shape[0] == 1, "Only one input at a time"
    input_ids = input_ids[:, :context_size]

    # assert that mask_token_id only appears once in the input_ids
    mask_token_id = tokenizer.mask_token_id
    assert (input_ids == mask_token_id).sum() == input_ids.shape[0]

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
    mask_position = input_ids[0].tolist().index(mask_token_id)
    mask_logits = outputs.logits[0, mask_position]
    mask_probs = torch.nn.functional.softmax(mask_logits, dim=-1)
    return mask_probs.cpu().numpy()

def return_logprobs_choices(prompt, answers, tokenizer, model, context_size):
    """ Given a prompt and a list of possible answers, return the most likely answer from the model """
    p = query_model_bert(prompt, tokenizer, model, context_size)

    # make all answers lowercase, without punctuation
    answer_probs = {}
    for answer in answers:
        without_space = tokenizer.encode(answer, add_special_tokens=False)[-1]
        answer_probs[answer] = p[without_space]

    answer = max(answer_probs, key=answer_probs.get)
    return answer

class BertMCTaskEvaluator(MCTaskEvaluator):
    def __call__(self, model, tokenizer, prompt, choice_labels):
        # add mask to the prompt
        prompt = f"{prompt} {tokenizer.mask_token}."

        choice_labels = [c.strip() for c in choice_labels]
        answer = return_logprobs_choices(prompt, choice_labels, tokenizer, model, 512)
        return {'model_response': answer}
    
class BertIntTaskEvaluator(IntTaskEvaluator):
    def __init__(self, evaluator, **kwargs):
        super().__init__(evaluator, **kwargs)
        # get all possible answers
        self.targets = set(ex['target'].strip() for ex in self)

    def __call__(self, model, tokenizer, prompt):
        prompt = f"{prompt} {tokenizer.mask_token}."
        answer = return_logprobs_choices(prompt, self.targets, tokenizer, model, 512)
        return {'model_response': answer}

def get_auto_task_evaluator(evaluator, **kwargs):
    from evaluation.hf_eval import MCEvaluator, IntEvaluator
    if isinstance(evaluator, MCEvaluator):
        return BertMCTaskEvaluator(evaluator, **kwargs)
    if isinstance(evaluator, IntEvaluator):
        return BertIntTaskEvaluator(evaluator, **kwargs)
    raise ValueError("Evaluator type not recognized")


if __name__ == '__main__':
    import json
    import transformers

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--task_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default=None)

    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--context_size', type=int, default=512)
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
        tasks = [t[:-5] for t in tasks if t.endswith('.json') and t != 'opinions.json']
    else:
        tasks = [args.task_name]
    
    print('Loading opinions...')
    opinions = {}
    for opinion_file in opinion_files:
        with open(f"{args.task_dir}/{opinion_file}", 'r') as f:
            opinions.update(json.load(f))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_dir)

    for task in tqdm(tasks):
        print(f"Processing task {task}...")

        file_name = f'{args.save_dir}/{task}.json'
        if os.path.exists(file_name):
            print(f'{file_name} already exists. Skipping...')
            continue

        # is_bert_type = isinstance(model, (transformers.BertPreTrainedModel, transformers.RobertaPreTrainedModel))
        mc_evaluator = get_auto_evaluator(
            task_dir=f"{args.task_dir}{task}.json",
            opinions=opinions,
            eval_split=args.eval_split,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            numbers=True,
        )

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