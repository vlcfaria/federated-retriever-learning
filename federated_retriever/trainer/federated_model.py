import torch
from tevatron.retriever.modeling import DenseModel


class FederatedDenseModel(DenseModel):
    def load_from_arrays(self, arrays) -> None:
        self.encoder.load_state_dict(arrays.to_torch_state_dict(), strict=True)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().to(torch.float32).cpu()
            for k, v in self.encoder.state_dict().items()
        }
