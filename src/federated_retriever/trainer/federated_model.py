from tevatron.retriever.modeling import DenseModel


class FederatedDenseModel(DenseModel):
    def load_from_arrays(self, arrays) -> None:
        self.encoder.load_state_dict(arrays.to_torch_state_dict(), strict=True)

    def encoder_state_dict(self):
        self.encoder.state_dict()
