"""
CLI to run training on a model
"""
import os
import logging
from pathlib import Path

import fire
import transformers

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train

from datasets import load_from_disk, concatenate_datasets
from axolotl.utils.trainer import calculate_total_num_steps

LOG = logging.getLogger("axolotl.cli.train")

def do_cli(config: Path = Path("examples/"), **kwargs):
    if 'num_gpus' in kwargs:
        num_gpus = kwargs['num_gpus']
        del kwargs['num_gpus']
    else:
        num_gpus = 1

    assert 'task' in kwargs, "task must be set"
    task = kwargs['task']

    if 'train_size' in kwargs:
        train_size = kwargs['train_size']
        del kwargs['train_size']
    else:
        train_size = None

    if train_size == 'all':
        train_size = None

    if 'seed' in kwargs:
        seed = kwargs['seed']
        del kwargs['seed']
    else:
        seed = None

    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((TrainerCliArgs))

    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    assert parsed_cfg.dataset_prepared_path is not None, "dataset_prepared_path must be set"
    LOG.info(f"Loading dataset from disk {parsed_cfg.dataset_prepared_path}")
    
    dset_dir = parsed_cfg.dataset_prepared_path
    print(f"Loading dataset from disk {dset_dir}")

    files = os.listdir(dset_dir)
    if task[-1] == '_':
        files = [f for f in files if f.startswith(task)]
    else:
        files = [f for f in files if f == task]

    train_dataset = []
    val_dataset = []
    for file in files:
        dataset = load_from_disk(f"{dset_dir}/{file}")
        train_dataset.append(dataset['train'])
        val_dataset.append(dataset['val'])

    train_dataset = concatenate_datasets(train_dataset)
    val_dataset = concatenate_datasets(val_dataset)

    if train_size is not None:
        assert seed is not None, "seed must be set"
        train_dataset = train_dataset.shuffle(seed=seed).select(range(train_size))

    n_eval = 250
    val_dataset = val_dataset.shuffle(seed=42).select(range(min(n_eval, len(val_dataset))))

    print('Train dataset')
    print(train_dataset)

    print('Val dataset')
    print(val_dataset)

    LOG.info(train_dataset)
    LOG.info("Dataset loaded from disk")

    parsed_cfg.sample_packing = False

    # set the appropiate number of gradient accumulation steps
    parsed_cfg.batch_size = min(len(train_dataset), 64) // parsed_cfg.micro_batch_size
    parsed_cfg.gradient_accumulation_steps = parsed_cfg.batch_size // num_gpus

    print(f"Micro batch size: {parsed_cfg.micro_batch_size}")
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Batch size: {parsed_cfg.batch_size}")
    print(f"Gradient accumulation steps: {parsed_cfg.gradient_accumulation_steps}")
    print(f"Total batch size: {parsed_cfg.gradient_accumulation_steps * parsed_cfg.micro_batch_size}")

    parsed_cfg.early_stopping_patience = 3

    multiplier = 0.7 if parsed_cfg.sample_packing else 1.0
    parsed_cfg.save_steps = int(len(train_dataset) // parsed_cfg.batch_size * multiplier)
    parsed_cfg.eval_steps = parsed_cfg.save_steps

    total_num_steps = calculate_total_num_steps(parsed_cfg, train_dataset)
    print("The total number of steps would be: ", total_num_steps)
    print("The number of epochs would be: ", parsed_cfg.num_epochs)
    steps_per_epoch = max(total_num_steps // parsed_cfg.num_epochs, 1)
    if parsed_cfg.max_steps is not None and parsed_cfg.max_steps < total_num_steps:
        total_num_steps = parsed_cfg.max_steps

    n_saves = total_num_steps // steps_per_epoch
    if n_saves < 5:
        parsed_cfg.save_steps = total_num_steps // 5
    else:
        parsed_cfg.save_steps = steps_per_epoch

    # eval every 0.1 epochs
    parsed_cfg.save_steps = int(parsed_cfg.save_steps * 0.1)
    parsed_cfg.eval_steps = parsed_cfg.save_steps

    print(f"Total number of steps: {total_num_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Save steps: {parsed_cfg.save_steps}")

    from axolotl.train import TrainDatasetMeta
    dataset_meta = TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        total_num_steps=total_num_steps,
    )

    # train the model
    train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
