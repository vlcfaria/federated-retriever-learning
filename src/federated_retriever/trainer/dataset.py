from datasets import DatasetDict, load_dataset
from flwr_datasets import FederatedDataset
from tevatron.retriever.arguments import DataArguments
from tevatron.retriever.dataset import TrainDataset


class FederatedTrainDataset(TrainDataset):
    """
    Dataset for federated learning. Automatically picks a dataset from RLHN-styled datasets,
    with `subset` column. `partition_id` and `num_partitions` allows to also split the same subset into multiple clients
    """

    def __init__(
        self,
        partition_subset: str,
        partition_id: int,
        num_partitions: int,
        data_args: DataArguments,
        trainer=None,
        dataset_name=None,
    ):
        # Init the grandparent to avoid calling load_dataset inside `TrainDataset`'s init
        super(TrainDataset, self).__init__()

        self.data_args = data_args
        self.trainer = trainer
        self.subset = partition_subset
        self.partition_id = partition_id
        self.num_partitions = num_partitions

        self.train_data = (
            FederatedDataset(
                dataset=self.data_args.dataset_name
                if dataset_name is None
                else dataset_name,
                preprocessor=lambda x: self._filter_dataset_to_subset(x, self.subset),
                partitioners={"train": num_partitions},
            )
            .load_partition(partition_id)
            .load_split(data_args.dataset_split)
        )

    def _filter_dataset_to_subset(self, dataset: DatasetDict, subset: str):
        return dataset.filter(lambda x: x["subset"] == subset)
