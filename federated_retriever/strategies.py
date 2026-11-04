from flwr.serverapp.strategy import FedAvg


class LRScheduleMixin:
    """Mixin for allowing linear decay/warmup LR for strategies"""

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


class ScheduledFedAvg(PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedAvg):
    pass
