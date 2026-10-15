import os

import torch
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from transformers import AutoTokenizer, HfArgumentParser


def prep_tokenizer(model_args, data_args):
    tokenizer_kwargs = {"cache_dir": model_args.cache_dir}
    if model_args.model_revision is not None:
        tokenizer_kwargs["revision"] = model_args.model_revision

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == "right":
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    return tokenizer


def prep_model_kwargs(model_args, training_args):
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model_load_kwargs = {
        "cache_dir": model_args.cache_dir,
        "torch_dtype": torch_dtype,
        "attn_implementation": model_args.attn_implementation,
    }
    if model_args.model_revision is not None:
        model_load_kwargs["revision"] = model_args.model_revision

    return model_load_kwargs


def parse_config_args(context):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # TODO look into changing into a json config?
    model_args, data_args, training_args = parser.parse_dict(
        dict(context.run_config), allow_extra_keys=True
    )
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    return model_args, data_args, training_args
