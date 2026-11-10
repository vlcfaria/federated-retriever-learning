# Script to submit a wandb run
import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml
from tevatron.eval.benchmarks import BEIR_13, BEIR_15, CQADUP_SUBS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("run_dir", type=str)

    return parser.parse_args()


def flatten_metrics(eval_dir: Path, train_datasets: list[str] = []) -> dict[str, float]:
    """Flatten ``eval/<bench>/<dataset>.metrics.json`` into summary keys:

    ``{"eval/<dataset>/<group>/<metric>": value, "eval/<dataset>/<scalar>": value}``
    """
    agg = {}
    for path in sorted(eval_dir.glob("*/*.metrics.json")):
        data = json.loads(path.read_text())
        dataset = data.get("dataset", path.name.split(".")[0])
        agg[dataset] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    agg[dataset][k] = v
            elif isinstance(value, (int, float)):
                agg[dataset][key] = value

    # Condensate cqadupstack values
    dedup_beir_15 = [x for x in BEIR_15 if not x.startswith("cqadupstack")] + [
        "cqadupstack"
    ]

    # in and out of domain beir variants
    beir_15_id = [x for x in dedup_beir_15 if x in train_datasets]
    beir_15_ood = [x for x in dedup_beir_15 if x not in train_datasets]

    print("In domain", beir_15_id)
    print("OOD", beir_15_ood)

    # Add BEIR aggregates
    for b_datasets, b_name in [
        (CQADUP_SUBS, "cqadupstack"),
        (BEIR_13, "BEIR-13"),
        (dedup_beir_15, "BEIR-15"),
        (beir_15_ood, "BEIR-15-OOD"),
        (beir_15_id, "BEIR-15-ID"),
    ]:
        filtered_datasets = {k: v for k, v in agg.items() if k in b_datasets}

        agg[b_name] = defaultdict(float)
        agg[b_name]["evaluated"] = list(filtered_datasets.keys())
        agg[b_name]["num_evaluated"] = len(filtered_datasets)

        for d in b_datasets:
            if d not in agg:
                continue

            for k, v in agg[d].items():
                if k == "num_evaluated" or k == "evaluated":
                    continue
                agg[b_name][k] += v

        # mean out metrics
        for k, v in agg[b_name].items():
            if "@" in k:
                agg[b_name][k] = v / len(filtered_datasets)

    flat = {
        f"eval/{dataset}/{k}": v
        for dataset, keys in agg.items()
        for k, v in keys.items()
    }

    return flat


def run_path(run_url: str) -> str:
    """``https://wandb.ai/<entity>/<project>/runs/<id>`` -> ``<entity>/<project>/<id>``.

    The public API's ``Api.run`` takes a path, not the URL ``wandb sync`` prints.
    """
    match = re.search(
        r"forge.coreweave.com/wandb/([^/\s]+)/([^/\s]+)/runs/([^/\s]+)", run_url
    )
    if not match:
        return run_url
    entity, project, run_id = match.groups()
    return f"{entity}/{project}/{run_id}"


def report_to_wandb(run_url: str, item: Path, force: bool = False) -> None:
    """Attach config, eval metrics and NOTES.md to the uploaded run."""
    import wandb

    run = wandb.Api().run(run_path(run_url))

    # NOTE fixed here since we train on all datasets
    train_datasets = [
        "arguana",
        "fiqa",
        "fever",
        "hotpotqa",
        "msmarco",
        "nq",
        "scidocs",
    ]

    metrics = flatten_metrics(item / "eval", train_datasets)
    if metrics:
        run.summary.update(metrics)

    run.update()


def sync_directory(wandb_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    cmd = [
        "wandb",
        "sync",
        str(wandb_path),
        "--project",
        "federated-retriever-learning",  # TODO remove this hardcoded project
        "--yes",
        "--no-skip-synced",
    ]
    if dry_run:
        print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return True, ""

    print(f"Syncing: {wandb_path}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


if __name__ == "__main__":
    args = parse_args()

    wandb_dir = Path(args.run_dir) / "wandb" / "wandb"
    _, output = sync_directory(wandb_dir)

    match = re.findall(r"https?://forge\.coreweave\.com/\S+", output)
    if not match:
        print(f"warning: no wandb URL in sync output: {output}")

    for url in sorted(set(match)):
        report_to_wandb(url, Path(args.run_dir), True)
