#!/usr/bin/env python3

import sys
from pathlib import Path

import htcondor

JOB_BID = 201
JOB_CPUS = 8
JOB_MEMORY = "64GB"

accelerate_template = """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: {layer}
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: {num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

def launch_experiment_job(
        CLUSTER_LOGS_SAVE_DIR,
        run_name,
        model_path,
        train_task,
        output_dir,
        gradient_accumulation_steps,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        lr,
        flash_attention,
        num_train_epochs=1,
        gradient_checkpointing=False,
        JOB_GPUS=1,
        GPU_MEM=None,
        transformer_layer=None,
        i=0,
):
    print(run_name)
        
    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = 'empty_bash.sh'

    if JOB_GPUS > 1:
        header = f"accelerate launch --main_process_port {29500+1+i} --config_file accelerate_config_{run_name}.yaml train_basic.py"
        accelerate_config = accelerate_template.format(
            layer=transformer_layer,
            num_processes=JOB_GPUS,
        )
        with open(f'accelerate_config_{run_name}.yaml', 'w') as f:
            f.write(accelerate_config)
    else:
        header = "python train_basic.py"

    train_command = (
        f"{header} "
        f"--model_name_or_path {model_path} "
        f"--dataset_name {train_task} "
        f"--output_dir {output_dir} "
        f"--per_device_train_batch_size {per_device_train_batch_size} "
        f"--per_device_eval_batch_size {per_device_eval_batch_size} "
        f"--gradient_accumulation_steps {gradient_accumulation_steps} "
        f"--bf16 "
        f"--learning_rate {lr} "
        f"--num_train_epochs {num_train_epochs} "
        f"--logging_strategy steps "
        f"--evaluation_strategy no "
        f"--save_strategy no "
        f"--report_to wandb "
        f"--run_name {run_name} "
        f"--adam_beta1 0.9 "
        f"--adam_beta2 0.95 "
        f"--lr_scheduler_type cosine "
        f"--adam_epsilon 1e-8 "
        f"--weight_decay 0.1 "
        f"--warmup_ratio 0.03 "
    )

    if gradient_checkpointing:
        train_command += "--gradient_checkpointing "
    if JOB_GPUS > 1:
        train_command += "--multi_gpu "

    all_commands = train_command

    job_settings = {
        "executable": f"{executable}",
        "arguments": f"{all_commands}",
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

    req = ''
    if GPU_MEM is not None and flash_attention:
        req = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (TARGET.CUDACapability >= 9.0)"
    elif GPU_MEM is not None:
        req = f"TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}"
    elif flash_attention:
        req = "TARGET.CUDACapability >= 8.0"
    if req:
        job_settings["requirements"] = req

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")

if __name__ == '__main__':

    models = {
        'llama-2-7b': {
            'model_path': '/fast/USER/llama-2-7b-hf/',
            'block_size': 4096,
            'GPU_MEM': 45000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-5,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_llama-2-4k_4096/',
            'JOB_GPUS': 4,
            'transformer_layer': 'LlamaDecoderLayer'
        },
        'llama-3-8b': {
            'model_path': '/fast/USER/models/llama-3-8b/snapshots/model/',
            'block_size': 8196,
            'GPU_MEM': 45000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-6,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_llama-3-8k/',
            'JOB_GPUS': 4,
            'transformer_layer': 'LlamaDecoderLayer',
        },
        'pythia-6.9b': {
            'model_path': '/fast/USER/models/pythia-6.9b/',
            'block_size': 2048,
            'GPU_MEM': 45000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-5,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_pythia-2k_2048/',
            'JOB_GPUS': 4,
            'transformer_layer': 'GPTNeoXLayer'
        },
        'pythia-70m': {
            'model_path': '/fast/USER/models/pythia-70m/snapshots/model/',
            'block_size': 2048,
            'GPU_MEM': 39000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-4,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_pythia-2k_2048/',
        },
        'pythia-160m': {
            'model_path': '/fast/USER/models/pythia-160m/snapshots/model/',
            'block_size': 2048,
            'GPU_MEM': 39000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-4,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_pythia-2k_2048/',
        },
        'pythia-410m': {
            'model_path': '/fast/USER/models/pythia-410m/snapshots/model/',
            'block_size': 2048,
            'GPU_MEM': 39000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-5,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_pythia-2k_2048/',
        },
        'pythia-1b': {
            'model_path': '/fast/USER/models/pythia-1b/snapshots/model/',
            'block_size': 2048,
            'GPU_MEM': 45000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-5,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_pythia-2k_2048/',
        },
        'pythia-2.8b': {
            'model_path': '/fast/USER/models/pythia-2.8b/snapshots/model/',
            'block_size': 2048,
            'GPU_MEM': 45000,
            'flash_attention': True,
            'per_device_eval_batch_size': 1,
            'per_device_train_batch_size': 1,
            'lr': 2e-5,
            'train_task': '/fast/USER/lawma/instructions-axo/tok_pythia-2k_2048/',
        },

    }

    batch_size = 64
    base_save_dir = '../models/lawma-scale-saves/'

    # if save dir does not exist, create it
    Path(base_save_dir).mkdir(parents=True, exist_ok=True)

    # for each model, need to specify the context size...
    for model_name, model_args in models.items():
        JOB_GPUS = model_args['JOB_GPUS'] if 'JOB_GPUS' in model_args else 1
        grad_acc_steps = int(batch_size / model_args['per_device_train_batch_size'] / JOB_GPUS)
        model_name_ = f"{model_name}-ft"

        my_model_args = model_args.copy()
        del my_model_args['block_size']

        launch_experiment_job(
            Path('/fast/USER/logs/'),
            run_name=model_name_,
            output_dir=f'{base_save_dir}{model_name}/',
            num_train_epochs=1,
            gradient_accumulation_steps=grad_acc_steps,
            **my_model_args,
        )
