#!/usr/bin/env python3

import sys
from pathlib import Path

import htcondor

JOB_BID = 52
JOB_CPUS = 8
JOB_GPUS = 1
JOB_MEMORY = "200GB"

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        task_name,
        train_size,
        seed,
        GPU_MEM=None,
):
    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = 'run_train_eff.sh'

    arguments = (
        f"{task_name} {train_size} {seed}"
    )

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
        job_settings["requirements"] = f'(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (TARGET.CUDACapability >= 9.0)'

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == '__main__':

    base_task_dir = '/fast/rolmedo/lawma/tasks20/'

    files = [
        'sc_issuearea',
        'sc_casedisposition',
        'sc_decisiondirection',
        'sc_lcdisposition',
        'sc_lcdispositiondirection',
        'songer_treat',
        'songer_geniss',
        'songer_origin',
        'songer_direct1',
        'sc_casesource',
    ]
    max_n_seeds = 5

    n_sizes = [10, 50, 100, 250, 500, 1000]
    for size in n_sizes:
        for task_name in files:
            n_seeds = 1 if size == 'all' else max_n_seeds
            for seed in range(n_seeds):
                launch_experiment_job(
                    CLUSTER_LOGS_SAVE_DIR=Path('/fast/rolmedo/logs/'),
                    task_name=task_name,
                    train_size=size,
                    seed=seed,
                    GPU_MEM=45000,
                )