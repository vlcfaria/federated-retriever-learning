from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.trainer import TevatronTrainer as Trainer

from .trainer.dataset import FederatedTrainDataset
from .trainer.federated_model import FederatedDenseModel
from .utils import parse_config_args, prep_model_kwargs, prep_tokenizer

app = ClientApp()


def _partition_info(context: Context) -> tuple[str, int, int]:
    """Return this node's (subset, partition-id, num-partitions)."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    subsets = [
        s.strip() for s in context.run_config["subsets"].split(",") if s.strip()
    ]
    subset = subsets[partition_id]
    return subset, partition_id, num_partitions


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Get this client's dataset partition
    subset, partition_id, num_partitions = _partition_info(context)
    model_args, data_args, training_args = parse_config_args(context)

    train_dataset = FederatedTrainDataset(
        subset, partition_id, num_partitions, data_args
    )
    tokenizer = prep_tokenizer(model_args, data_args)
    collator = TrainCollator(data_args, tokenizer)

    # Prep the model
    model = FederatedDenseModel.build(
        model_args,
        training_args,
        **prep_model_kwargs(model_args, training_args),
    )
    # Initialize it with the received weights
    model.load_from_arrays(msg.content["arrays"])

    # Train!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataset.set_trainer(trainer)

    trainer.train()

    # Construct and return reply Message
    model_record = ArrayRecord(model.encoder_state_dict())
    metrics = MetricRecord({"num-examples": len(train_dataset)})

    content = RecordDict({"arrays": model_record, "metrics": metrics})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""
    # TODO eval is not trivial on IR, dummy metrics for now
    model_args, _, training_args = parse_config_args(context)

    model = FederatedDenseModel.build(
        model_args,
        training_args,
        **prep_model_kwargs(model_args, training_args),
    )
    model.load_from_arrays(msg.content["arrays"])

    model_record = ArrayRecord(model.encoder_state_dict())
    metrics = MetricRecord({"num-examples": 1, "loss": 0.0, "accuracy": 1.0})

    content = RecordDict({"arrays": model_record, "metrics": metrics})

    return Message(content=content, reply_to=msg)
