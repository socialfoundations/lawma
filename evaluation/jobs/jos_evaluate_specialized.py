#!/usr/bin/env python3

import sys
from pathlib import Path

import htcondor

JOB_BID = 55
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
    import os

    CLUSTER_LOGS_SAVE_DIR = Path('/fast/USER/logs/')

    save_dir = '../notebooks/results/specialization/'
    task_dir = '../tasks/'
    models_base_dir = '/fast/USER/lawma-specialized-warmup/'

    os.makedirs(save_dir, exist_ok=True)
    all_tasks = os.listdir(task_dir)

    models = os.listdir(models_base_dir)
    for task_name in models:
        model_dir = f"{models_base_dir}{task_name}/"
        my_save_dir = f"{save_dir}{task_name}/"
        os.makedirs(my_save_dir, exist_ok=True)

        # match the task name with all the tasks
        if task_name[-1] == '_':
            do_tasks = [task[:-5] for task in all_tasks if task.startswith(task_name)]  # remove .json
        else:
            do_tasks = [task_name]

        print("Eval tasks: ", do_tasks)
        for task in do_tasks:
                launch_experiment_job(
                    CLUSTER_LOGS_SAVE_DIR,
                    model_dir,
                    my_save_dir,
                    task_dir,
                    task,
                    eval_split='test',
                    context_size=8192,
                    max_samples=1000,
                    JOB_GPUS=1,
                    GPU_MEM=38000
                )

    print("Done")