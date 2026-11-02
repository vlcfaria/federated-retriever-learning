from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from tevatron.retriever.checkpoint_validation import evaluate_group
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.trainer import TevatronTrainer as Trainer

from .trainer.dataset import FederatedTrainDataset
from .trainer.federated_model import FederatedDenseModel
from .utils import flatten_dict, parse_config_args, prep_model_kwargs, prep_tokenizer

app = ClientApp()


def _partition_info(context: Context) -> tuple[str, int, int]:
    """Return this node's (subset, partition-id, num-partitions).

    `partition_id` and `num_partitions` are subset localized.
    """

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    subsets = [s.strip() for s in context.run_config["subsets"].split(",") if s.strip()]
    subset = subsets[partition_id % len(subsets)]

    # num_partitions will indicate the number of partitions containing THIS subset
    subset_partitions = num_partitions // len(subsets)

    # account for non-perfect partitions
    if (partition_id % len(subsets)) < (num_partitions % len(subsets)):
        subset_partitions += 1

    subset_partition_id = partition_id // len(subsets)

    return subset, subset_partition_id, subset_partitions


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Get this client's dataset partition
    subset, partition_id, num_partitions = _partition_info(context)
    model_args, data_args, training_args, _ = parse_config_args(context)

    train_dataset = FederatedTrainDataset(
        subset, partition_id, num_partitions, data_args
    )
    print(subset, "training on", len(train_dataset), "queries")

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

    if training_args.max_steps > 0:
        # TODO this is incorrect on multiple gpu scenarios
        training_examples = max(
            len(train_dataset),
            training_args.max_steps * training_args.per_device_train_batch_size,
        )
    else:
        training_examples = len(train_dataset)  # epoch-based

    metrics = MetricRecord({"num-examples": training_examples})

    content = RecordDict({"arrays": model_record, "metrics": metrics})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""
    model_args, data_args, training_args, validate_args = parse_config_args(context)

    model = FederatedDenseModel.build(
        model_args,
        training_args,
        **prep_model_kwargs(model_args, training_args),
    )
    model.load_from_arrays(msg.content["arrays"])
    tokenizer = prep_tokenizer(model_args, data_args)

    subset, _, _ = _partition_info(context)

    metrics = evaluate_group(
        model, tokenizer, data_args, training_args, validate_args, subset
    )

    metrics = flatten_dict(metrics)

    metrics["num-examples"] = (
        1  # Cross-silo training should be macro-averaged for retrieval (e.g. BEIR)
    )

    model_record = ArrayRecord(model.encoder_state_dict())
    metrics = MetricRecord(metrics)

    content = RecordDict({"arrays": model_record, "metrics": metrics})

    return Message(content=content, reply_to=msg)
