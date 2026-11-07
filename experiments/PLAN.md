# Experiment plan

## Controls

- Centralized baseline: 10,000 steps x batch 256 (~2.56M query-samples).
- Fixed-step FedAvg: 60 rounds x 95 steps, and 30 x 190 / 45 x 125 (compute-matched to centralized).
- Variable-step FedAvg: local epochs `E=0.125`, 30 rounds (~3.75 epochs/client, matches 30x190).
  Per-client steps: arguana 8, fever 56, fiqa 11, hotpotqa 165, msmarco 949, nq 114, scidocs 25.
- Weighting: size-weighted (current, msmarco ~71%) by default. Uniform is a separate ablation.

Keep each strategy's control matched on step regime and weighting. Never compare across regimes.

## Implementation order

1. `strategy` run-config dispatch (flags, not new sbatch files).
2. Server momentum (FedAvgM): `beta=0.9`, `server_lr=1.0`.
3. Variable-step local-epochs mode; client reports actual local steps `tau`.
4. FedProx client proximal term; sweep `mu`.
5. FedNova normalized aggregation.
6. Composites (only after individual effects are known).

Keep the global LR schedule keyed to a nominal `steps_per_round` (95 for 60 rounds, 190 for 30)
so schedules stay comparable across variable-step runs.

## Ablations

### Wave 1 (independent, queue together)

| ID | Run | Compare to | Purpose |
|----|-----|-----------|---------|
| C0 | FedAvg, `E=0.125`, 30 rounds | fedavg_long_steps (30x190) | Does equalizing epochs fix the 120x imbalance? |
| A1 | FedAvgM, fixed steps, `beta=0.9, eta=1.0` | fedavg (60x95) | Server momentum |
| B1 | FedProx, fixed steps, `mu in {1e-3, 1e-2, 1e-1}` | fedavg (60x95) | Client drift; small sweep for loss scale |

### Wave 2 (depends on Wave 1)

| ID | Run | Compare to | Purpose |
|----|-----|-----------|---------|
| D0 | FedAvg, fixed steps, uniform weights | fedavg (60x95) | Screen uniform vs size weighting; dismiss if no gain |
| C1 | FedNova, `E=0.125`, 30 rounds | C0 | Isolate FedNova normalization |

If D0 helps, rerun the winning variable-step run with uniform weighting too (FedNova's `p_i` changes `tau_eff`).

### Wave 3 (only if the pieces pay off)

- FedProx + FedAvgM under the winning step regime.
- FedNova + FedAvgM if C1 beats C0.
- Retune `mu` / `beta` under variable steps; hyperparameters may not transfer.

If GPU budget forces a choice, run C0 first; it can change what every other result means.

## Stacking rules

- **FedProx + FedAvgM**: clean; client loss vs server aggregation, no interaction. Best combo to try.
- **FedNova + FedAvgM**: custom hybrid, not a published method. Works while `fraction-train=1.0`
  (stable `tau_eff`); risky if clients are sampled. Only after FedNova alone.
- **FedProx + FedNova**: technically compatible, but conflates drift control with step-count
  correction. Run each alone first.
- **Triple stack**: don't.

## Caveats

- Server momentum is also an effective LR multiplier: steady state is `~1/(1-beta) = 10x` with
  `beta=0.9`. To separate smoothing from step size, compare against an LR-matched FedAvg.
- With msmarco at 71% weight, momentum can entrench domain bias; uniform weighting or lower `beta`
  may matter.
- Local AdamW restarts every round. This is fine with FedAvgM, but each round has a short transient.
- FedNova assumes SGD-like updates; local AdamW and normalized updates make it approximate.
- FedNova changes the effective server step (`tau_eff`), so re-check LR rather than assuming transfer.

## Running

Queue with `script/submit.sh` / sbatch; do not run `flwr run` interactively on the headnode.
