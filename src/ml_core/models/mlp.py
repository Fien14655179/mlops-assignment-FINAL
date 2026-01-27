from typing import Optional
import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """
    Late-fusion MLP that supports:
      - fused: x_vis + x_txt
      - vis-only: x_txt=None
      - txt-only: x_vis=None

    forward(x_vis, x_txt, mask_txt=None)
    """

    def __init__(
        self,
        vis_dim: int,
        txt_dim: int,
        hidden_units: list[int],
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vis_dim = vis_dim
        self.txt_dim = txt_dim
        input_dim = vis_dim + txt_dim  # may be vis-only or txt-only

        if input_dim <= 0:
            raise ValueError("FusionMLP requires vis_dim + txt_dim > 0")

        layers = []
        in_features = input_dim

        for h in hidden_units:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x_vis: Optional[torch.Tensor],
        x_txt: Optional[torch.Tensor],
        mask_txt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_vis is None and x_txt is None:
            raise ValueError("At least one of x_vis or x_txt must be provided")

        if x_vis is None:
            x = x_txt
        elif x_txt is None:
            x = x_vis
        else:
            x = torch.cat([x_vis, x_txt], dim=-1)

        return self.network(x)
