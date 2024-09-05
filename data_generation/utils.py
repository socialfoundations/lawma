import os
import json
import random
random.seed(42)


def get_majority_opinion(case):
    maj_opinion = None
    for opinion in case['caselaw']['casebody']['opinions']:
        if opinion['type'] == 'majority':
            assert maj_opinion is None, case['caselaw']['casebody']['opinions']
            maj_opinion = opinion['text']
    return maj_opinion

# return cases with a valid majority opinion
def get_cases_with_maj_opinion(dataset):
    for case_ in dataset:
        maj_opinion = get_majority_opinion(case_)
        if maj_opinion is None or len(maj_opinion) == 0:
            continue
        yield case_

# Save the opinions corresponding to each id in the dataset
def save_opinions(dataset, ids, save_dir, prefix=''):
    print("Saving the opinions...")
    opinions = {}
    for case in dataset:
        id_ = case['caselaw']['id']
        if id_ in ids:
            opinions[id_] = get_majority_opinion(case)
    
    with open(f"{save_dir}/{prefix}opinions.json", "w") as jsonl_file:
        json.dump(opinions, jsonl_file)

def subsample_majority_class(decisions, verbose=False):  # decisions -> decisions
    # compute the majority class
    counts = {}
    for ex in decisions.values():
        target = ex['target']
        # only a few are multilabel, safe to ignore
        if type(target) == list:
            continue
        if target not in counts:
            counts[target] = 0
        counts[target] += 1
    maj_class = max(counts, key=counts.get)
    ids_majority = [id_ for id_, ex in decisions.items() if ex['target'] == maj_class]

    # Now we subsample the majortiy class
    n_majority = len(ids_majority)
    n_minority = len(decisions) - n_majority

    multiplier = 1  # 50% of the majority class at most
    downsampled_by = 1
    if n_majority > multiplier * n_minority:
        if verbose:
            print(f"Before, number of decisions: {len(decisions)}, number of ids discussed: {n_minority}")
        remove_ids = set(random.sample(ids_majority, n_majority - multiplier * n_minority))
        decisions = {id_: ex for id_, ex in decisions.items() if id_ not in remove_ids}
        if verbose:
            print(f"After, number of decisions: {len(decisions)}, number of ids discussed: {n_minority}")

        removed = n_majority - multiplier * n_minority
        downsampled_by = (n_majority - removed) / n_majority

    return decisions, {'downsampled_by': downsampled_by, 'majority_class': maj_class}

def subsample_and_save_decisions(task, decisions, splits, save_dir=None,
                                limit_train=True, limit_test=True, verbose=False):

    if verbose and (not 'answer_choices' in task):
        print("No answer choices for the task", task['name'])

    # Create the train-test splits
    n = 0
    ids = set()
    examples = {}
    for split, ids_ in splits.items():
        scaling = 1
        split_examples = {id_: decisions[id_] for id_ in ids_ if id_ in decisions}
        if (
            ((split == 'train' and limit_train) or
            (split in ['val', 'test'] and limit_test)) and
            len(split_examples) > 10
        ):
            split_examples, scaling = subsample_majority_class(
                decisions=split_examples,
                verbose=verbose
        )
        examples[split] = list(split_examples.values())
        ids.update(split_examples.keys())
        n += len(split_examples)

        if split == 'test' and save_dir is not None:
            scaling_save_dir = f"{save_dir}/scaling_factors/"
            if not os.path.exists(scaling_save_dir):
                os.makedirs(scaling_save_dir)

            with open(f"{scaling_save_dir}{task['name']}.json", 'w') as f:
                json.dump(scaling, f)

    if verbose:
        for split, exs in examples.items():
            print(f"Number of examples in {split}: {len(exs)}")

    dataset = {'task': task, 'examples': examples}

    # Save as a json file
    if save_dir is not None:
        save_file = f"{save_dir}/{task['name']}.json"
        with open(save_file, "w") as jsonl_file:
            json.dump(dataset, jsonl_file)

    return ids, n