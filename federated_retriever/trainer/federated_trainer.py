import copy
import gc
import math
import os
import socket
from logging import INFO

import deepspeed
import torch
from flwr.common import log
from tevatron.retriever.trainer import TevatronTrainer
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

_ENV_SET_BY_US = False


def ensure_single_process_dist() -> None:
    """Initialize a single-process process group for DeepSpeed if needed."""
    global _ENV_SET_BY_US

    if torch.distributed.is_initialized():
        return

    def free_port() -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return str(sock.getsockname()[1])

    if "LOCAL_RANK" not in os.environ:
        os.environ.update(
            LOCAL_RANK="0",
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=free_port(),
        )
        _ENV_SET_BY_US = True
    elif _ENV_SET_BY_US:
        # A previous engine destroyed the process group; pick a fresh port.
        os.environ["MASTER_PORT"] = free_port()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


class FederatedTrainer(TevatronTrainer):
    """Local DeepSpeed training loop for federated training rounds.

    Preferred over TevatronTrainer since Deepspeed oddly trains better models than regular HF trainer in Tevatron
    """

    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        data_collator,
        ds_config=None,
        min_steps: int = 1,
        proximal_mu: float = 0.0,
        **kwargs,
    ):
        if ds_config is None:
            ds_config = getattr(args, "ds_config", None)
        if ds_config is None:
            raise ValueError("A DeepSpeed config must be provided via `deepspeed`.")

        # Stop HF Trainer from creating its own DeepSpeed plugin/Accelerator state.
        trainer_args = copy.copy(args)
        trainer_args.deepspeed = None
        trainer_args.deepspeed_plugin = None
        trainer_args.hf_deepspeed_config = None

        super().__init__(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            **kwargs,
        )
        self.ds_config = ds_config
        self.min_steps = max(1, int(min_steps))
        self.proximal_mu = float(proximal_mu)
        self._global_params = None
        self._last_prox = 0.0
        self.steps_done = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = super().compute_loss(
            model, inputs, return_outputs, num_items_in_batch
        )
        if self.proximal_mu <= 0 or self._global_params is None:
            return loss

        module = model.module if hasattr(model, "module") else model
        params = [p for p in module.parameters() if p.requires_grad]
        prox = sum(
            (param.float() - global_param.float()).pow(2).sum()
            for param, global_param in zip(params, self._global_params)
        )
        self._last_prox = float(prox.detach())
        return loss + 0.5 * self.proximal_mu * prox

    def _resolve_ds_config(self, num_training_steps: int) -> dict:
        hf_config = HfTrainerDeepSpeedConfig(self.ds_config)
        hf_config.trainer_config_process(self.args)
        hf_config.trainer_config_finalize(self.args, self.model, num_training_steps)
        return hf_config.config

    def train_local(self) -> tuple[dict[str, torch.Tensor], float]:
        self.model.train()
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        dataloader = self.get_train_dataloader()
        num_steps = self.args.max_steps
        if num_steps is None or num_steps <= 0:
            num_steps = max(1, math.ceil(len(dataloader) * self.args.num_train_epochs))
            if num_steps < self.min_steps:
                log(
                    INFO,
                    "%s | raising local steps from %d to min_steps=%d",
                    getattr(self.train_dataset, "subset", "client"),
                    num_steps,
                    self.min_steps,
                )
                num_steps = self.min_steps

        logging_steps = self.args.logging_steps
        if isinstance(logging_steps, float) and 0 < logging_steps < 1:
            logging_steps = max(1, int(logging_steps * num_steps))
        else:
            logging_steps = max(1, int(logging_steps))

        label = getattr(self.train_dataset, "subset", "client")
        partition_id = getattr(self.train_dataset, "partition_id", None)
        if partition_id is not None:
            label = f"{label}[{partition_id}]"
        num_examples = len(self.train_dataset)

        ensure_single_process_dist()

        engine = None
        try:
            # Should initialize a fresh optimizer state. Standard for FL
            # We can alleviate this a bit using a bit of warmup
            engine, _, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=[
                    p for p in self.model.parameters() if p.requires_grad
                ],
                config=self._resolve_ds_config(num_steps),
            )
            if self.proximal_mu > 0:
                self._global_params = [
                    p.detach().clone()
                    for p in engine.module.parameters()
                    if p.requires_grad
                ]

            total_loss = 0.0
            running_loss = 0.0
            running_count = 0
            steps_done = 0
            while steps_done < num_steps:
                for batch in dataloader:
                    loss = self.compute_loss(engine, batch)
                    loss_value = loss.detach().float().item()
                    total_loss += loss_value
                    running_loss += loss_value
                    running_count += 1
                    engine.backward(loss)
                    engine.step()
                    steps_done += 1

                    if steps_done % logging_steps == 0 or steps_done >= num_steps:
                        lr = (
                            float(engine.get_lr()[0])
                            if hasattr(engine, "get_lr")
                            else self.args.learning_rate
                        )
                        log(
                            INFO,
                            "%s | n=%d | step %d/%d | loss %.4f | lr %.2e | prox %.3g",
                            label,
                            num_examples,
                            steps_done,
                            num_steps,
                            running_loss / max(1, running_count),
                            lr,
                            self._last_prox,
                        )
                        running_loss = 0.0
                        running_count = 0

                    if steps_done >= num_steps:
                        break

            self.steps_done = steps_done
            return engine.module.encoder_state_dict(), total_loss / max(1, steps_done)
        finally:
            if engine is not None and hasattr(engine, "destroy"):
                engine.destroy()
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
