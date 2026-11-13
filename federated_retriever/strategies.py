import numpy as np
from flwr.app import Array, ArrayRecord
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedProx
from scipy.optimize import minimize

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
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
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


class FedNovaMGDA(FedAvg):
    """FedNova + FedMGDA+ for fairness-based multi-objective gradient descent with step correction

    We combine both algorithms by applying:
        - Normalizing pseudo-gradients by steps taken
        - Extracting the unit vector of each pseudo-gradient
        - Solving the pareto update from MGDA according to weights and epsilon
        - Re-scaling according to steps again
    """

    def __init__(self, *args, epsilon: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def configure_train(self, server_round, arrays, config, grid):
        self._global_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = self._aggregate_arrays(reply_contents, server_round)
            metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays, metrics

    def _aggregate_arrays(self, reply_contents, server_round):
        if self._global_arrays is None:
            raise RuntimeError(
                "FedNovaMGDA.configure_train must run before aggregate_train"
            )

        weights = []
        taus = []
        for record in reply_contents:
            metricrecord = next(iter(record.metric_records.values()))
            weights.append(float(metricrecord[self.weighted_by_key]))
            taus.append(float(metricrecord["num-steps"]))

        m = len(reply_contents)
        weights_arr = np.asarray(weights, dtype=np.float64)
        taus_arr = np.asarray(taus, dtype=np.float64)
        if np.any(taus_arr <= 0):
            raise ValueError("FedNovaMGDA requires positive num-steps")
        p = weights_arr / np.sum(weights_arr)

        global_model_np = {k: v.numpy() for k, v in self._global_arrays.items()}

        # define each fednova's step-normalized pseudo-gradient
        sq_norms = np.zeros(m, dtype=np.float64)
        pseudogradients = []

        for i, (record, tau) in enumerate(zip(reply_contents, taus_arr)):
            client_arrays = next(iter(record.array_records.values()))
            d_tilde = {}

            # Apply (w0 - w) / tau, then normalize
            for k, v in client_arrays.items():
                diff = (global_model_np[k] - v.numpy()) / tau
                d_tilde[k] = diff
                sq_norms[i] += float(np.sum(diff.astype(np.float64) ** 2))

            pseudogradients.append(d_tilde)
        norms = np.sqrt(sq_norms) + 1e-12

        # Build mxm Gram Matrix, G_ij = dot(d_i, d_j)
        G = np.zeros((m, m), dtype=np.float64)
        for k in global_model_np:
            # Shape: (m, num_elements_in_layer_k)
            layer_stack = np.stack(
                [
                    pg[k].reshape(-1).astype(np.float64) for pg in pseudogradients
                ],  # (m, nxn)
                axis=0,
            )
            G += (
                layer_stack @ layer_stack.T
            )  # dot(d_i, d_j) for this layer, this is mxm
        # Unit normalize
        G /= np.outer(norms, norms)  # divide by norm(d_i) * norm(d_j)

        solver_success = True
        solver_message = "skipped (m==1 or epsilon<=0)"
        if m == 1 or self.epsilon <= 0.0:
            lam_star = p
        else:

            def objective(lam: np.ndarray) -> float:
                return float(0.5 * np.dot(lam, np.dot(G, lam)))

            def jac(lam: np.ndarray) -> np.ndarray:
                return np.dot(G, lam)

            constraints = {"type": "eq", "fun": lambda lam: np.sum(lam) - 1.0}
            bounds = [
                (
                    max(0.0, float(p[i] - self.epsilon)),
                    min(1.0, float(p[i] + self.epsilon)),
                )
                for i in range(m)
            ]

            res = minimize(
                objective,
                x0=p,
                method="SLSQP",
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 200},
            )
            solver_success = bool(res.success)
            solver_message = str(res.message)
            lam_star = res.x if solver_success else p
            # Clean up any numerical drift on the simplex
            lam_star = np.clip(lam_star, 0.0, 1.0)
            lam_star /= np.sum(lam_star)

        # Anchor the FedNova step size to the prior weighting p. MGDA only
        # changes the direction through lam_star.
        tau_eff = float(np.dot(p, taus_arr))
        # Rescale each unit direction back to the FedNova gradient scale.
        step_scale = float(np.dot(p, norms))

        mgda_log = {
            "mgda/solver_success": int(solver_success),
            "mgda/tau_eff": tau_eff,
            "mgda/step_scale": step_scale,
        }
        for i, record in enumerate(reply_contents):
            subset = record["config"]["subset"]
            mgda_log[f"mgda/lambda/{subset}"] = float(lam_star[i])
        if wandb.run is not None:
            wandb.log({**mgda_log, "server_round": server_round})
        print(
            f"[FedNovaMGDA] round={server_round} solver={solver_message} "
            f"success={solver_success} tau_eff={tau_eff:.4f} "
            f"step_scale={step_scale:.6g} lambda={np.round(lam_star, 6).tolist()}"
        )

        # tau_eff and step_scale use p; lam_star only blends the unit directions.
        client_coeffs = (tau_eff * step_scale * lam_star) / norms

        new_global_dict = {}
        for k, w0 in global_model_np.items():
            update_k = np.zeros_like(w0, dtype=np.float64)
            for i in range(m):
                update_k += client_coeffs[i] * pseudogradients[i][k]

            updated_param = (w0 - update_k).astype(w0.dtype)
            new_global_dict[k] = Array(updated_param)

        return ArrayRecord(new_global_dict)


class ScheduledFedNova(
    WandbStrategyMixin, PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedNova
):
    pass


class ScheduledFedNovaMGDA(
    WandbStrategyMixin, PeriodicEvaluationStrategyMixin, LRScheduleMixin, FedNovaMGDA
):
    pass


SERVER_STRATEGIES = {
    "fedavg": ScheduledFedAvg,
    "fedavgm": ScheduledFedAvgM,
    "fedprox": ScheduledFedProx,
    "fednova": ScheduledFedNova,
    "fednovamgda": ScheduledFedNovaMGDA,
}
