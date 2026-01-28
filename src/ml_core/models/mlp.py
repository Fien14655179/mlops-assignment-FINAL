import torch
import torch.nn as nn


class LateFusionMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # Extract dimensions from config
        self.visual_dim = config.get("visual_dim", 768)
        self.text_dim = config.get("text_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_classes = config.get("num_classes", 32)
        self.dropout_rate = config.get("dropout", 0.2)

        # Determine input dimension
        self.input_dim = 0
        if self.visual_dim > 0:
            self.input_dim += self.visual_dim
        if self.text_dim > 0:
            self.input_dim += self.text_dim

        # Define the layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x_vis=None, x_txt=None):
        if x_vis is not None and x_txt is None:
            x = x_vis
        elif x_txt is not None and x_vis is None:
            x = x_txt
        elif x_vis is not None and x_txt is not None:
            x = torch.cat((x_vis, x_txt), dim=1)
        else:
            raise ValueError("Model received NO input!")

        return self.layers(x)
