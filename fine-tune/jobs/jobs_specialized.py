#!/usr/bin/env python3

import sys
from pathlib import Path

import htcondor

JOB_BID = 250
JOB_CPUS = 8
JOB_GPUS = 1
JOB_MEMORY = "150GB"

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        task_name,
        base_model,
        model_name,
        GPU_MEM=None,
):
    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = 'run_train_specialized.sh'

    arguments = (
        f"{task_name} "
        f"{base_model} "
        f"{model_name} "
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
        job_settings["requirements"] = f'TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}'

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == '__main__':

    base_task_dir = '../tasks/'

    base_models = [
        'ricdomolm/lawma-8b',
        '../models/lawma-scale-saves/llama-3-8b/',
        'meta-llama/Meta-Llama-3-8B-Instruct',
    ]

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

    for base_model in base_models:
        # mk save_dir and model_dir if not exist
        model = base_model.split('/')[-2]
        save_dir = Path(f'../notebooks/results/specialization/{model}')
        model_dir = Path(f'../models/lawma-specialization-saves/{model}')
        save_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        for task_name in files:
            launch_experiment_job(
                CLUSTER_LOGS_SAVE_DIR=Path('/fast/rolmedo/logs/'),
                task_name=task_name,
                base_model=base_model,
                model_name=model,
                GPU_MEM=45000,
            )