"""Run tevatron's encode driver with bf16 autocast.

tevatron's encode() enters `torch.amp.autocast("cuda")` without a dtype, which
defaults to float16. For ModernBERT that is the wrong compute dtype (fp16
autocast costs several NDCG points; bf16 autocast matches fp32). This wrapper
sets the autocast dtype and then executes the stock driver unchanged, so the
CLI is identical to `python -m tevatron.retriever.driver.encode ...`.
"""

import runpy

import torch

if torch.cuda.is_available():
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    print(f"[encode_bf16] cuda autocast dtype = {torch.get_autocast_dtype('cuda')}", flush=True)

runpy.run_module("tevatron.retriever.driver.encode", run_name="__main__")
