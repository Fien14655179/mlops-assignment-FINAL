from .loader import get_dataloaders
from .pcam import PCAMDataset

# TCGA (final assignment)
from .tcga import TCGADataset
from .tcga_loader import get_tcga_dataloaders

__all__ = [
    "get_dataloaders",
    "PCAMDataset",
    "TCGADataset",
    "get_tcga_dataloaders",
]
