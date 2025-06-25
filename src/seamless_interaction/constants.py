# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final


ALL_LABELS: Final = ["improvised", "naturalistic"]
ALL_SPLITS: Final = ["train", "dev", "test"]
ALL_FEATURES: Final = {
    "smplh": [  # npy
        "body_pose",
        "global_orient",
        "is_valid",
        "left_hand_pose",
        "right_hand_pose",
        "translation",
    ],
    "boxes_and_keypoints": [  # npy
        "box",
        "is_valid_box",
        "keypoints",
    ],
    "movement": [  # npy
        "EmotionArousalToken",
        "emotion_valence",
        "EmotionValenceToken",
        "expression",
        "FAUToken",
        "frame_latent",
        "FAUValue",
        "gaze_encodings",
        "alignment_head_rotation",
        "head_encodings",
        "alignment_translation",
        "hypernet_features",
        "emotion_arousal",
        "is_valid",
        "emotion_scores",
    ],
    "metadata": [  # json
        "transcript",  # (optional)
        "vad",  # (required)
    ],
    "annotations": [
        "1P-IS",
        "1P-R",
        "3P-IS",
        "3P-R",
        "3P-V",
    ],  # json (optional)
}
# Regular expression to parse file IDs from filenames
FILE_ID_REGEX = r"V(\d+)_S(\d+)_I(\d+)_P(\d+)"  # <vendor_id>_<session_id>_<interaction_id>_<participant_id>
