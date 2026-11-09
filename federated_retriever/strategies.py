import numpy as np
import wandb
from flwr.app import Array, ArrayRecord
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx


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
        wandb.init(project=project_name, name=run_name, resume="never", config=config)

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

        if aggregated is None:
            return aggregated

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


class ScheduledFedAvgM(
    WandbStrategyMixin, PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedAvgM
):
    pass


class ScheduledFedProx(
    WandbStrategyMixin, PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedProx
):
    pass


class FedNova(FedAvg):
    """FedAvg with per-client local-step normalization.

    The server update is

        w_next = w + tau_eff * sum_i p_i * (w_i - w) / tau_i

    where tau_i is the number of local steps performed by client i and
    tau_eff = sum_i p_i * tau_i. When all clients take the same number of
    steps this is exactly FedAvg.
    """

    def configure_train(self, server_round, arrays, config, grid):
        self._global_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = self._aggregate_arrays(reply_contents)
            metrics = self.train_metrics_aggr_fn(
                reply_contents, self.weighted_by_key
            )
        return arrays, metrics

    def _aggregate_arrays(self, reply_contents):
        if self._global_arrays is None:
            raise RuntimeError(
                "FedNova.configure_train must run before aggregate_train"
            )

        weights = []
        taus = []
        for record in reply_contents:
            metricrecord = next(iter(record.metric_records.values()))
            weights.append(float(metricrecord[self.weighted_by_key]))
            taus.append(float(metricrecord["num-steps"]))

        total_weight = sum(weights)
        p = [weight / total_weight for weight in weights]
        tau_eff = sum(pi * tau for pi, tau in zip(p, taus))

        global_arrays = self._global_arrays.to_numpy_ndarrays()
        keys = list(self._global_arrays.keys())

        aggregated = {}
        for key, global_value in zip(keys, global_arrays):
            update = np.zeros_like(global_value)
            for record, pi, tau in zip(reply_contents, p, taus):
                if tau <= 0:
                    raise ValueError("FedNova requires positive num-steps")
                client_arrays = next(iter(record.array_records.values()))
                update += pi * (client_arrays[key].numpy() - global_value) / tau
            aggregated[key] = global_value + tau_eff * update

        return ArrayRecord(
            {key: Array(np.asarray(value)) for key, value in aggregated.items()}
        )


class ScheduledFedNova(
    WandbStrategyMixin, PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedNova
):
    pass


SERVER_STRATEGIES = {
    "fedavg": ScheduledFedAvg,
    "fedavgm": ScheduledFedAvgM,
    "fedprox": ScheduledFedProx,
    "fednova": ScheduledFedNova,
}
