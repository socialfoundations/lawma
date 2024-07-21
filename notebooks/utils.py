"""Helper functions for processing responses."""

import os
import json
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import numpy as np

def load_responses(responses, models, base_dir):
    for model in models:
        responses[model] = {}
        model_dir = os.path.join(base_dir, model)
        for file in tqdm(os.listdir(model_dir)):
            task_name = file.split('.')[0]
            
            if task_name in responses[model]:
                continue

            with open(os.path.join(model_dir, file), 'r') as f:
                responses[model][task_name] = json.load(f)

        print(f'{model} has {len(responses[model])} tasks')
    return responses


def find_first_integer(input_string):
    if input_string is None:
        return None
    
    num_str = ''
    for char in input_string.replace(',', ''):
        if char.isdigit():
            num_str += char
        elif num_str:
            break
    
    if num_str:
        return str(int(num_str))

    return None


def find_first_char(input_string):
    # find first capital character
    if not input_string:
        return ''
    for char in input_string:
        if char.isupper():
            return char
    return ''


def compute_counts(y_true, y_pred):
    """Compute number of correct predictions and total number of predictions."""
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if type(true) == list:
            correct += pred.strip() in [t.strip() for t in true]
        else:
            if pred is not None:
                correct += true.strip() == pred.strip()    
    return correct, len(y_true)


def accuracy(y_true, y_pred):
    """Compute accuracy."""
    successes, trials = compute_counts(y_true, y_pred)
    return successes / trials


def clopper_pearson_ci(successes, trials, confidence_level=0.95):
    """
    Calculate 95% confidence interval using Clopper-Pearson method (exact method).
    """
    alpha = 1 - confidence_level
    if successes == 0:
        lower_bound = 0.0
    else:
        lower_bound = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    if successes == trials:
        upper_bound = 1.0
    else:
        upper_bound = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return [lower_bound, upper_bound]


def compute_bars(s, n, confidence_level=0.9):
    """Compute lower and upper width for error bars."""
    acc = s/n
    (l, u) = clopper_pearson_ci(s, n, confidence_level)
    return acc - l, u - acc


def responses_to_predictions(responses, gpt4=False):
    """Compute ground truth and predictions from responses."""

    y_true = [r['ground_truth'] for r in responses]
    y_pred = [r['model_response'] for r in responses]

    is_digit = y_true[0]
    if type(is_digit) == list:
        is_digit = is_digit[0]
    is_digit = is_digit.strip().isdigit()
    if is_digit:
        y_pred = [find_first_integer(pred) for pred in y_pred]
        if gpt4:
            y_true = [find_first_integer(true) for true in y_true]
    elif gpt4:
        y_pred = [find_first_char(pred) for pred in y_pred]
    
    return y_true, y_pred


def responses_to_acc(responses, gpt4=False):
    y_true, y_pred = responses_to_predictions(responses, gpt4=gpt4)
    return accuracy(y_true, y_pred)


def responses_to_counts(responses, gpt4=False):
    y_true, y_pred = responses_to_predictions(responses, gpt4=gpt4)
    return compute_counts(y_true, y_pred)


def majority_responses(responses):
    """Compute predictions and ground truth for majority classifier."""
    y_true = [r['ground_truth'] for r in responses]

    # count how many times each response appears
    counts = {}
    for response in y_true:
        if type(response) != list:
            response = [response]
        for r in response:
            if r not in counts:
                counts[r] = 0
            counts[r] += 1
    
    # get the most common response
    most_common = max(counts, key=counts.get)
    y_pred = [most_common for _ in y_true]

    return y_true, y_pred


def majority_trials(responses):
    y_true, y_pred = majority_responses(responses)
    return compute_counts(y_true, y_pred)


def majority_acc(responses):
    y_true, y_pred = majority_responses(responses)
    return accuracy(y_true, y_pred)


def compute_mean_acc(accs, model, prefix=''):
    if model not in accs:
        return 0.
    
    mean_accs = {}
    for task, acc in accs[model].items():
        if task.startswith(prefix):
            mean_accs[task] = acc
    return np.mean(list(mean_accs.values()))


def compute_averages(responses, tasks_to_average):
    """Compute task averages for given responses and tasks."""
    pull_responses = {m: {} for m in responses.keys()}
    for model in responses.keys():
        for task in responses[model].keys():
            found = False
            for pull, new in tasks_to_average.items():
                if task.startswith(pull):
                    if new not in pull_responses[model]:
                        pull_responses[model][new] = []
                    pull_responses[model][new].extend(responses[model][task])
                    found = True
                    break
            if not found:
                pull_responses[model][task] = responses[model][task]

    # compute pull_accuracy
    pull_accs = {}
    for model, model_responses in pull_responses.items():
        pull_accs[model] = {}
        for file, response in model_responses.items():
            pull_accs[model][file] = responses_to_acc(response)
    pull_accs['maj'] = {k: majority_acc(v) for k, v in list(pull_responses.values())[0].items()}

    return pull_accs


def compute_case_acc(trials, prefix=''):
    num_successes = 0
    num_trials = 0
    for task in trials.keys():
        if task.startswith(prefix):
            (s, n) = trials[task]
            num_successes += s
            num_trials += n
    acc = num_successes / num_trials
    lb, ub = compute_bars(num_successes, num_trials)
    return acc, (lb, ub)


def compute_task_bars(trials, prefix=''):
    lower_bars = []
    upper_bars = []
    for task, (s, n) in trials.items():
        if task.startswith(prefix):
            lb, ub = compute_bars(s, n)
            lower_bars.append(lb)
            upper_bars.append(ub)
    return (np.mean(lower_bars), np.mean(upper_bars))    