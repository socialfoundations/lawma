#!/usr/bin/env python3

import os
import htcondor
from pathlib import Path

executable = '/home/USER/axo121/bin/python3'

JOB_BID = 36
JOB_CPUS = 8
JOB_MEMORY = "64GB"

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        model_dir,
        save_dir,
        task_dir,
        task_name,
        eval_split='test',
        context_size=4096,
        max_samples=None,
        JOB_GPUS=1,
        GPU_MEM=None,
):
    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    if 'bert' in model_dir:
        script = 'bert_eval.py'
    else:
        script = 'hf_eval.py'

    arguments = (
        f"{script} "
        f"--model_dir {model_dir} "
        f"--save_dir {save_dir} "
        f"--task_dir {task_dir} "
        f"--task_name {task_name} "
        f"--eval_split {eval_split} "
        f"--context_size {context_size} "
    )
    if max_samples is not None:
        arguments += f"--max_samples {max_samples} "

    # print(arguments)

    # Construct job description
    job_settings = {
        "executable": f"{executable}",
        "arguments":  arguments,
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_gpus": f"{JOB_GPUS}",
        "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        "request_memory": JOB_MEMORY,  # how much memory we want
        "request_disk": JOB_MEMORY,
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "rdo@tue.mpg.de",
        "notification": "error",
    }

    if GPU_MEM is not None:
        job_settings["requirements"] = f"TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}"

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == '__main__':
    CLUSTER_LOGS_SAVE_DIR = Path('/fast/USER/logs/')

    model_dirs = {
        'legalbert': 'nlpaueb/legal-bert-base-uncased',
        'llama-3-70b-instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'llama-3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'mistral-7b-inst': 'mistralai/Mistral-7B-Instruct-v0.2',
        'mixtral-inst': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'saulinst': 'Equall/Saul-7B-Instruct-v1',
        'lawma-8b': 'ricdomolm/lawma-8b',
        'lawma-70b': 'ricdomolm/lawma-70b',
    }   

    save_dir = '../notebooks/results/model_responses/'
    task_dir = '../tasks/'

    def n_gpus(model):
        if '70b' in model or 'mixtral' in model:
            return 3
        return 1
    
    def gpu_mem(model):
        if n_gpus(model) > 1:
            return 45000
        
        if '8b' in model.lower() or '7b' in model.lower():
            return 38000
        return 45000

    for model_name, model_dir in model_dirs.items():
        the_save_dir = f"{save_dir}{model_name}/"

        # Ensure the save directory exists
        os.makedirs(the_save_dir, exist_ok=True)

        # launch one job per .json file
        tasks = os.listdir(task_dir)
        tasks = [t[:-5] for t in tasks if t.endswith('.json') and not t.endswith('opinions.json')]

        for task_name in tasks:
            file_name = f"{the_save_dir}{task_name}.json"

            # Skip if the file already exists
            if os.path.exists(file_name):
                print(f"Skipping {file_name}")
                continue

            launch_experiment_job(
                CLUSTER_LOGS_SAVE_DIR,
                model_dir,
                the_save_dir,
                task_dir,
                task_name,
                eval_split='test',
                context_size=512 if 'bert' in model_dir else 8192,
                max_samples=1000,
                JOB_GPUS=n_gpus(model_name),
                GPU_MEM=gpu_mem(model_name),
                )

    print("Done")