from flwr.serverapp.strategy import FedAvg

import wandb


class LRScheduleMixin:
    def __init__(
        self, *args, base_lr, steps_per_round, total_steps, warmup_steps, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.steps_per_round = steps_per_round
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def _get_lr(self, server_round: int) -> float:
        # use round midpoint, avoids lr=0 on last round
        step = max(0.0, (server_round - 0.5) * self.steps_per_round)

        if step < self.warmup_steps:  # linear warmup
            return self.base_lr * step / max(1, self.warmup_steps)
        # lienar decay
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.base_lr * max(0.0, 1.0 - progress)

    def configure_train(self, server_round, arrays, config, grid):
        config["learning_rate"] = float(self._get_lr(server_round))
        return super().configure_train(server_round, arrays, config, grid)


class PeriodicEvaluationStrategyMixin:
    """Mixin for allowing periodic evaluation, evaluating only every `eval_interval` rounds of training"""

    def __init__(self, *args, eval_interval: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_interval = eval_interval

    def _should_eval(self, round: int):
        return round % self.eval_interval == 0

    def configure_evaluate(self, server_round: int, *args, **kw):
        if not self._should_eval(server_round):
            return []

        return super().configure_evaluate(server_round, *args, **kw)


class WandbStrategyMixin:
    def __init__(self, *args, project_name: str, run_name: str, config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        wandb.init(project=project_name, name=run_name, resume="auto", config=config)

    def aggregate_train(self, server_round: int, replies, *args, **kwargs):
        model, metrics = super().aggregate_train(server_round, replies, *args, **kwargs)

        to_log = {"server_round": server_round}
        for r in replies:
            subset = r.content["config"]["subset"]
            to_log |= {f"{subset}/{m}": v for m, v in r.content["metrics"].items()}
        to_log |= {f"mean/{m}": v for m, v in metrics.items()}

        wandb.log(to_log)

        return model, metrics

    def aggregate_evaluate(self, server_round: int, replies, *args, **kwargs):
        aggregated = super().aggregate_evaluate(server_round, replies, *args, **kwargs)

        to_log = {"server_round": server_round}
        for r in replies:
            subset = r.content["config"]["subset"]
            to_log |= {f"{subset}/{m}": v for m, v in r.content["metrics"].items()}
        to_log |= {f"mean/{m}": v for m, v in aggregated.items()}

        wandb.log(to_log)

        return aggregated


class ScheduledFedAvg(
    WandbStrategyMixin, PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedAvg
):
    pass
