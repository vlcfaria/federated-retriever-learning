import os
from pathlib import Path

import torch
from tevatron.retriever.arguments import (
    DataArguments,
    ModelArguments,
    ValidateArguments,
)
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
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ValidateArguments)
    )

    # TODO look into changing into a json config?
    run_config = dict(context.run_config)
    # Keep `deepspeed` out of TrainingArguments: its __post_init__ resets
    # Accelerate state and would build a DeepSpeed PartialState that conflicts
    # with the manually initialized engine.
    ds_path = run_config.pop("deepspeed", None)
    if isinstance(ds_path, str) and not os.path.isabs(ds_path):
        candidate = Path(__file__).resolve().parents[1] / ds_path
        if candidate.exists():
            ds_path = str(candidate)

    model_args, data_args, training_args, validate_args = parser.parse_dict(
        run_config, allow_extra_keys=True
    )
    training_args.ds_config = ds_path
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    validate_args: ValidateArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    return model_args, data_args, training_args, validate_args


def flatten_dict(dictionary):
    """Flatten a dictionary by adding prefixing to keys"""

    ans = {}

    for k, v in dictionary.items():
        if isinstance(v, dict):
            ans = ans | {
                f"{k}/{k_nested}": v_nested
                for k_nested, v_nested in flatten_dict(v).items()
            }
        else:
            ans[k] = v

    return ans


def save_model(model_args, data_args, path: str | Path, model, state_dict):
    print("\nSaving final model to disk...")
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir, state_dict=state_dict)
    prep_tokenizer(model_args, data_args).save_pretrained(out_dir)
