# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

import numpy as np


def setup_logging(
    name: str, log_file: str = "seamless_interaction_fs.log", level: int = logging.INFO
) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


def recursively_cast_to_float32(data: Any) -> Any:
    if isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            return data.astype(np.float32)
        else:
            return data
    elif isinstance(data, list):
        return [recursively_cast_to_float32(item) for item in data]
    elif isinstance(data, dict):
        return {k: recursively_cast_to_float32(v) for k, v in data.items()}
    else:
        return data  # Keep other types unchanged
