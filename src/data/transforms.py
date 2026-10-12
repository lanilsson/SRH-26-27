"""Feature preprocessing shared by both dataset builders.

AI-generated: Claude Code (Opus 5.5), 2026-10-11. Log the prompt in the
project prompt log and cite per the ISEF generative-AI table.
"""

import torch


def standardize_by_train(x, train_rows):
    """Standardise every column with the mean and std of the training rows
    only (proposal, Build Procedure step 1), so no statistic is computed
    from validation or test nodes. Columns that are constant over the
    training rows are only centred (std treated as 1)."""
    mean = x[train_rows].mean(0)
    std = x[train_rows].std(0)
    std[std == 0] = 1.0
    return (x - mean) / std
