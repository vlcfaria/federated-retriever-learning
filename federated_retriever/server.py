from pathlib import Path

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from .strategies import SERVER_STRATEGIES
from .trainer.federated_model import FederatedDenseModel
from .utils import (
    parse_config_args,
    prep_model_kwargs,
    save_model,
    load_strategy_params,
)

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

    # Theoretically could be anything here for the scheduler to work...
    steps_per_round = max(1, int(training_args.max_steps))

    total_steps = num_rounds * steps_per_round
    warmup_steps = training_args.get_warmup_steps(total_steps)

    output_dir = Path(str(context.run_config.get("output-dir", "final_model_export")))
    eval_interval = int(context.run_config["eval-interval"] or 1)

    server_strategy = SERVER_STRATEGIES[str(context.run_config["server-strategy"])]
    server_strategy_params = load_strategy_params(
        context.run_config, "server-strategy-params"
    )

    strategy = server_strategy(
        fraction_train=float(fraction_train),
        fraction_evaluate=float(fraction_evaluate),
        base_lr=training_args.learning_rate,
        steps_per_round=steps_per_round,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        eval_interval=eval_interval,
        project_name=str(context.run_config["project-name"]),
        run_name=str(context.run_config["run-name"]),
        config=context.run_config,
        **server_strategy_params,
    )

    # Start the strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=int(num_rounds),
        evaluate_fn=lambda round, array_record: (
            save_model(
                model_args,
                data_args,
                output_dir / f"checkpoint_round_{round}",
                model,
                array_record.to_torch_state_dict(),
            )
            if round != 0 and round % eval_interval == 0
            else None
        ),
    )

    if context.run_config.get("save-model", True):
        save_model(
            model_args,
            data_args,
            output_dir,
            model,
            result.arrays.to_torch_state_dict(),
        )
