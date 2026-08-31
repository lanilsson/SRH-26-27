"""Dataset builders. Each returns a PyTorch Geometric graph with
train/val/test masks attached, built from the CSVs in data/raw/ and
cached as a .pt file in data/processed/."""

from src.data.elliptic import load_elliptic
from src.data.ieee_cis_g import load_ieee_cis_g

DATASETS = {
    "elliptic": load_elliptic,
    "ieee_cis_g": load_ieee_cis_g,
}
