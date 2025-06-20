# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""This file contains the collate function for the SeamlessNext dataset, as well as utils to
load and validate features from files in the SeamlessNext dataset format.
"""
from typing import Any, Dict, List, Tuple

import torch
from fairseq2.gang import Gang
from seamless_next.datasets._batch import DataBatch
from seamless_next.datasets.constants import (
    FEATURE_COLLECTION_ANNOTATIONS,
    FEATURE_COLLECTION_METADATA,
)

from seamless_next.datasets.dataset_validations import (
    validate_batch_feature_rate_consistency,
)


def shuffle(seed: int, window: int, builder: Any) -> Tuple[int, Any]:
    """FS2 shuffle by mutating seed and applying a window shuffle
    Args:
        seed: Seed to use for shuffling
        window: Window size to use for shuffling
        builder: Builder to apply shuffle to
    """
    seed += 1
    if window != 1:
        builder.shuffle(window, seed=seed)
    return seed, builder


def _tensorize_batch(batch: Dict[str, Any], gang: Gang) -> DataBatch:
    """Tensorize a batch of data"""
    for feature, data in batch.items():
        if feature.startswith(FEATURE_COLLECTION_METADATA) or feature.startswith(
            FEATURE_COLLECTION_ANNOTATIONS
        ):
            continue
        if isinstance(data, dict):
            batch[feature] = _tensorize_batch(data, gang)
        elif isinstance(data, list):
            if isinstance(data[0], dict):
                batch[feature] = collate_seamless_next_data_batch(data, gang).data
            elif isinstance(data[0], str):
                batch[feature] = data
            elif not isinstance(data[0], torch.Tensor):
                batch[feature] = torch.tensor(data)
            else:
                batch[feature] = torch.stack(data).to(gang.device)
                validate_batch_feature_rate_consistency(feature, data)
    return DataBatch(batch)


def collate_seamless_next_data_batch(
    batch_samples: List[Dict[str, Any]],
    gang: Gang,
) -> DataBatch:
    """Collate a batch of data from a SeamlessNext dataset"""
    batch: dict[str, Any] = {}

    # Flatten the batch into {feature: [values]}
    for sample in batch_samples:
        for key, value in sample.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(value)
    return _tensorize_batch(batch, gang)
