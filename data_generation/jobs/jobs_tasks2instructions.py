#!/usr/bin/env python3

import sys
from pathlib import Path

import htcondor

JOB_BID = 199
JOB_CPUS = 4
JOB_MEMORY = "32GB"

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        task_dir,
        tokenizer_dir,
        tokenizer_name,
        context_size,
        task_name,
        save_dir,
):
    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/home/rolmedo/axo121/bin/python'

    # Construct job description
    job_settings = {
        "executable": f"{executable}",
        "arguments": (
            "tasks2instructions.py "
            f"--task_dir {task_dir} "
            f"--tokenizer_dir {tokenizer_dir} "
            f"--tokenizer_name {tokenizer_name} "
            f"--context_size {context_size} "
            f"--task_name {task_name} "
            f"--save_dir {save_dir} "
            f"--overwrite "
            f"--val_split val "
        ), # type: ignore
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        "request_memory": JOB_MEMORY,  # how much memory we want
        "request_disk": JOB_MEMORY,
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "rdo@tue.mpg.de",
        "notification": "error",
    }

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")


if __name__ == '__main__':
    import os

    task_dir = '../tasks/'
    save_dir = '../instructions/'

    tokenizer_kwargs = [
        {
            'tokenizer_dir': 'meta-llama/Meta-Llama-3-8B',
            'context_size': 8192,
            'tokenizer_name': 'llama-3-8k',
        },
        {
            'tokenizer_dir': 'EleutherAI/pythia-6.9b',
            'context_size': 2048,
            'tokenizer_name': 'pythia-2k',
        },
        {
            'tokenizer_dir': 'meta-llama/Llama-2-7b-hf',
            'context_size': 4096,
            'tokenizer_name': 'llama-2-4k',
        },
    ]

    # create the save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the tasks from all the files in task_dir
    tasks = []
    for filename in os.listdir(task_dir):
        if filename.endswith('.json') and not filename.endswith('opinions.json'):
            tasks.append(filename[:-5])

    print(f"Found {len(tasks)} tasks")

    for kwargs in tokenizer_kwargs:
        for task_name in tasks:
            print(f"Launching experiment for task {task_name}")
            launch_experiment_job(
                CLUSTER_LOGS_SAVE_DIR=Path("/fast/rolmedo/logs/"),
                task_dir=task_dir,
                task_name=task_name,
                save_dir=save_dir,
                **kwargs,
            )
    
    print('All jobs submitted')