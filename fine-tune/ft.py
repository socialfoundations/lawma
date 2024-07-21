"""
CLI to run training on a model
"""
import logging
from pathlib import Path

import fire
import transformers

from tqdm import tqdm

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
    prefix = kwargs.get('prefix', None)

    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((TrainerCliArgs))

    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    cfg = parsed_cfg
    assert cfg.dataset_prepared_path is not None, "dataset_prepared_path must be set"
    LOG.info(f"Loading dataset from disk {cfg.dataset_prepared_path}")
    
    dset_dir = cfg.dataset_prepared_path
    print(f"Loading dataset from disk {dset_dir}")

    # list all of the subdirs
    subdirs = [str(f) for f in Path(dset_dir).iterdir() if f.is_dir()]
    if prefix:
        print(f"Filtering by prefix: {prefix}")
        subdirs = [f for f in subdirs if f.split('/')[-1].startswith(prefix)]
        
    print("Number of subdirs:", len(subdirs))

    # each subdir is a dataset, load and concatenate them
    train_dataset = concatenate_datasets([load_from_disk(subdir)['train'] for subdir in tqdm(subdirs)])

    print('Train dataset')
    print(train_dataset)

    LOG.info(train_dataset)
    LOG.info("Dataset loaded from disk")

    if cfg.max_steps:
        total_num_steps = cfg.max_steps
    else:
        LOG.info("Calculating total number of steps")
        total_num_steps = calculate_total_num_steps(cfg, train_dataset)

    from axolotl.train import TrainDatasetMeta
    dataset_meta = TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=None,
        total_num_steps=total_num_steps,
    )

    train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
