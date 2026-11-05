from pathlib import Path

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from .strategies import ScheduledFedAvg
from .trainer.federated_model import FederatedDenseModel
from .utils import parse_config_args, prep_model_kwargs, prep_tokenizer

server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    model_args, data_args, training_args, _ = parse_config_args(context)

    model = FederatedDenseModel.build(
        model_args,
        training_args,
        **prep_model_kwargs(model_args, training_args),
    )
    arrays = ArrayRecord(model.encoder_state_dict())

    # Instantiate strategy
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    steps_per_round = int(training_args.max_steps)

    total_steps = num_rounds * steps_per_round
    warmup_steps = training_args.get_warmup_steps(total_steps)

    strategy = ScheduledFedAvg(
        fraction_train=float(fraction_train),
        fraction_evaluate=float(fraction_evaluate),
        base_lr=training_args.learning_rate,
        steps_per_round=steps_per_round,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eval_interval=int(context.run_config["eval-interval"] or 1),
        project_name=str(context.run_config["project-name"]),
        run_name=str(context.run_config["run-name"]),
        config=context.run_config,
    )

    # Start the strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=int(num_rounds),
    )

    if context.run_config.get("save-model", False):
        # Save final model to disk
        print("\nSaving final model to disk...")
        out_dir = Path(str(context.run_config.get("output-dir", "final_model_export")))
        out_dir.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(
            out_dir, state_dict=result.arrays.to_torch_state_dict()
        )
        prep_tokenizer(model_args, data_args).save_pretrained(out_dir)
