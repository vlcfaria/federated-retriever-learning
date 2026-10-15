import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .trainer.federated_model import FederatedDenseModel
from .utils import parse_config_args, prep_model_kwargs

server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    model_args, _, training_args = parse_config_args(context)

    model = FederatedDenseModel.build(
        model_args,
        training_args,
        **prep_model_kwargs(model_args, training_args),
    )
    arrays = ArrayRecord(model.encoder_state_dict())

    # Instantiate strategy
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    num_rounds = context.run_config["num-server-rounds"]
    # Start the strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    if context.run_config.get("save-model", False):
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
