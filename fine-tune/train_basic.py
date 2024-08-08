#!/usr/bin/env python

"""
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
from the Accelerate example zoo https://huggingface.co/docs/accelerate/usage_guides/training_zoo
"""
import numpy as np
import torch
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoTokenizer,

)
from datasets import load_from_disk, concatenate_datasets
from dataclasses import dataclass

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        if self.tokenizer.padding_side == "right":
            input_ids = [torch.Tensor(instance["input_ids"]).long() for instance in instances]
            labels = [torch.Tensor(instance["labels"]).long() for instance in instances]
        else:  # reverse
            input_ids = [torch.Tensor(instance["input_ids"][::-1]).long() for instance in instances]
            labels = [torch.Tensor(instance["labels"][::-1]).long() for instance in instances]
        attention_mask = [torch.Tensor([1] * len(instance["input_ids"])).long() for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        if self.tokenizer.padding_side == "left":
            input_ids = input_ids.flip(1)
            labels = labels.flip(1)
            attention_mask = attention_mask.flip(1)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    

def load_model_tokenizer(model_name_or_path, tokenizer_name=None, multi_gpu=False):
    assert model_name_or_path, "You must pass a model_name_or_path"

    # Load the tokenizer
    tokenizer_kwargs = {
        'pretrained_model_name_or_path': tokenizer_name if tokenizer_name else model_name_or_path,
        'trust_remote_code': True,
    }
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    model_kwargs = {
        'pretrained_model_name_or_path': model_name_or_path,
    }

    if not multi_gpu:
        model_kwargs['device_map'] = 'auto'

    def class_to_use(model_name):
        if 'falcon' in model_name.lower():
            return transformers.FalconForCausalLM
        return transformers.AutoModelForCausalLM

    # try to load the model with flash_attention_2
    # for some reason if dtype is not specified, the model will be loaded with float32
    try:
        # flash attention only works for cuda compatibility >= 8, raise error if not
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise ValueError("Flash attention only works for CUDA compatibility >= 8")
        
        model = class_to_use(model_name_or_path).from_pretrained(
            # torch_dtype=torch.float16,  # or bfloat16, depending
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            **model_kwargs
        )
        print("LOADED THE MODEL WITH FLASH ATTENTION")
        tokenizer.padding_side = 'left'
    except:
        model = class_to_use(model_name_or_path).from_pretrained(
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )
        print("Loaded the model without flash attention")

    if 'qwen' in model_name_or_path.lower():
        tokenizer.padding_side = 'left'

    # print the dtype of the model
    print(f"Model dtype: {model.dtype}")
    print("Warning... Setting pad token to EOS")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing the embeddings to match the tokenizer size.")
        model.resize_token_embeddings(len(tokenizer))

    # print GPU memory usage
    print('After loading the model and tokenizer...')
    return model, tokenizer

from dataclasses import field, dataclass
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    multi_gpu: Optional[bool] = field(
        default=False,
        metadata={"help": "Use multiple GPUs"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


if __name__ == "__main__":
    from pathlib import Path
    from transformers import HfArgumentParser, Trainer, TrainingArguments

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    subdirs = [str(f) for f in Path(data_args.dataset_name).iterdir() if f.is_dir()]

    # filter by task name
    task_name = data_args.task_name
    if task_name:
        if task_name[-1] == '_':
            rule = lambda f: f.split('/')[-1].startswith(task_name)
        else:
            rule = lambda f: f.split('/')[-1] == task_name
        subdirs = [f for f in subdirs if rule(f)]

    print("Number of subdirs:", len(subdirs))
    train_dataset = concatenate_datasets([load_from_disk(subdir)['train'] for subdir in tqdm(subdirs)])
    print("Training on", len(train_dataset), "examples")

    model, tokenizer = load_model_tokenizer(model_args.model_name_or_path, multi_gpu=model_args.multi_gpu)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Training
    train_result = trainer.train()
    
    print('Saving model...')
    if model_args.multi_gpu:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir=training_args.output_dir)  # Saves the tokenizer too for easy upload