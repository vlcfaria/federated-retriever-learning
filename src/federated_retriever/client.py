from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.trainer import TevatronTrainer as Trainer

from .trainer.dataset import FederatedTrainDataset
from .trainer.federated_model import FederatedDenseModel
from .utils import prep_model_kwargs, prep_tokenizer

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Prepare the dataset
    subset = context.node_config["subset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    data_args = msg.content["config"]["model_args"]
    model_args = msg.content["config"]["model_args"]
    training_args = msg.content["config"]["training_args"]

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
    # TODO this is empty for now, eval is not trivial on IR

    # Get this client's dataset partition
    subset = context.node_config["subset"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    data_args = context.run_config["data_args"]
    model_args = context.run_config["model_args"]
    training_args = context.run_config["training_args"]

    model = FederatedDenseModel.build(
        model_args,
        training_args,
        **prep_model_kwargs(model_args, training_args),
    )
    model.load_from_arrays(msg.content["arrays"])

    model_record = ArrayRecord(model.encoder_state_dict())
    metrics = MetricRecord(
        {
            "num-examples": 500,
            "loss": 0.0,
            "accuracy": 1.0,
        }
    )
    content = RecordDict({"arrays": model_record, "metrics": metrics})

    return Message(content=content, reply_to=msg)
