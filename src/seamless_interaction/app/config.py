# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List


CSS = """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stVideo > div {
        border-radius: 10px;
        overflow: hidden;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .interaction-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .file-status {
        font-family: monospace;
        font-size: 0.9rem;
    }
    .download-section {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    div[data-testid="stSelectbox"] > label {
        font-weight: bold;
        color: #2c3e50;
    }
</style>
"""

@dataclass
class DatasetStats:
    total_interactions: int
    total_participants: int
    vendors: List[str]
    sessions_per_vendor: Dict[str, int]
    interactions_per_session: Dict[str, int]
