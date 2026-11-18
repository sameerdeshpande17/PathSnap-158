"""Trip-grouped cross-validation — observations from one trip never appear in
both fitting and validation parts of a fold."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.explore import ROOT
from pathsnap.labels import available_trips


@dataclass
class FoldSpec:
    fold: int
    train_trips: list[str]
    validation_trips: list[str]


def make_folds(trip_ids: list[str], n_splits: int = 3,
               shuffle: bool = True, seed: int = 42,
               save_path: Path | None = None) -> list[FoldSpec]:
    gkf = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(trip_ids, groups=trip_ids)):
        folds.append(FoldSpec(
            fold=fold_idx,
            train_trips=[trip_ids[i] for i in train_idx],
            validation_trips=[trip_ids[i] for i in val_idx],
        ))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps([
            dict(fold=f.fold, train_trips=f.train_trips,
                 validation_trips=f.validation_trips) for f in folds
        ], indent=2))
    return folds


def load_folds(path: Path | str) -> list[FoldSpec]:
    data = json.loads(Path(path).read_text())
    return [FoldSpec(fold=d["fold"], train_trips=d["train_trips"],
                     validation_trips=d["validation_trips"]) for d in data]


def assert_no_leakage(folds: list[FoldSpec]) -> None:
    for fold in folds:
        overlap = set(fold.train_trips) & set(fold.validation_trips)
        if overlap:
            raise ValueError(f"Fold {fold.fold} leaks trips: {overlap}")


def training_trips(split: str = "train") -> list[str]:
    return available_trips(split)
