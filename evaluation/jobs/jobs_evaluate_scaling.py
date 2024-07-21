#!/usr/bin/env python3

import re
import sys
from pathlib import Path

import htcondor

JOB_BID = 100
JOB_CPUS = 4
JOB_MEMORY = "32GB"

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        model_dir,
        save_dir,
        task_dir,
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

    executable = '/home/USER/axo121/bin/python3'

    if 'bert' in model_dir:
        script = 'bert_eval.py'
    else:
        script = 'hf_eval.py'

    arguments = (
        f"{script} "
        f"--model_dir {model_dir} "
        f"--save_dir {save_dir} "
        f"--task_dir {task_dir} "
        f"--eval_split {eval_split} "
        f"--context_size {context_size} "
    )
    if max_samples is not None:
        arguments += f"--max_samples {max_samples} "

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
    import os

    CLUSTER_LOGS_SAVE_DIR = Path('/fast/USER/logs/')

    model_dirs = {}   
    base_dir = '/fast/USER/lawma-scale-saves/'
    save_dir = '../notebooks/results/scaling-experiments/'
    task_dir = '../tasks/'

    for file in os.listdir(base_dir):
        subfiles = os.listdir(f"{base_dir}/{file}")
        if len(subfiles) > 0:
            model_dirs[file] = f"{base_dir}/{file}"

    def n_gpus(model):
        if '70b' in model or 'mixtral' in model:
            return 3
        return 1
    
    def gpu_mem(model):
        def get_number_before(string, char):
            pattern = fr'(\d+(\.\d+)?)\s*[{char}]'
            match = re.search(pattern, string)
            if match:
                return float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
            else:
                return None
            
        is_b = get_number_before(model, 'b')
        if is_b is not None:
            if 3 < is_b < 9:
                return 38000
            elif is_b < 3:
                return 38000
            else:
                return 45000

        is_m = get_number_before(model, 'm')
        if is_m is not None:
            return 38000

        return 45000
    
    def get_context_size(model):
        if 'bert' in model:
            return 512
        if 'pythia' in model:
            return 2048
        if 'llama-3' in model:
            return 8192
        return 4096

    for model_name, model_dir in model_dirs.items():
        the_save_dir = f"{save_dir}{model_name}/"

        # Ensure the save directory exists
        os.makedirs(the_save_dir, exist_ok=True)

        launch_experiment_job(
            CLUSTER_LOGS_SAVE_DIR,
            model_dir,
            the_save_dir,
            task_dir,
            eval_split='test',
            context_size=get_context_size(model_name),
            max_samples=10,  # since there are many many models
            JOB_GPUS=n_gpus(model_name),
            GPU_MEM=gpu_mem(model_name),
        )

    print("Done")