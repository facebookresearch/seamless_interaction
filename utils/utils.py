import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import numpy as np
import torch
import torchaudio

from utils.constants import (
    AUDIO_WAV_TENSOR,
    BOXES_AND_KEYPOINTS_BOXES,
    BOXES_AND_KEYPOINTS_KEYPOINTS,
    DURATION_TAG,
    FEATURE_COLLECTION_AUDIO,
    FEATURE_COLLECTION_METADATA,
    FEATURE_COLLECTION_MOVEMENT,
    FEATURE_COLLECTION_SMIRK,
    FEATURE_COLLECTION_SMPLH,
    FEATURE_COLLECTION_SPEECH_TOKENS,
    JSONL_KEY_ID,
    JSONL_KEY_SPEECH_TOKEN_RATE,
    JSONL_KEY_VISUAL_RATE,
    METADATA_TEXT,
    METADATA_VAD,
    METADATA_WORDS_KEY,
    MOVEMENT_ALIGNMENT_HEAD_ROTATION,
    MOVEMENT_ALIGNMENT_TRANSLATION,
    MOVEMENT_EMOTION_AROUSAL,
    MOVEMENT_EMOTION_AROUSAL_TOKEN,
    MOVEMENT_EMOTION_SCORES,
    MOVEMENT_EMOTION_VALENCE,
    MOVEMENT_EMOTION_VALENCE_TOKEN,
    MOVEMENT_EXPRESSION,
    MOVEMENT_FAU_TOKEN,
    MOVEMENT_FAU_VALUE,
    MOVEMENT_FRAME_LATENT,
    MOVEMENT_GAZE_ENCODINGS,
    MOVEMENT_HEAD_ENCODINGS,
    MOVEMENT_HYPERNET_FEATURES,
    MOVEMENT_HYPERNET_FEATURES_RATE,
    MOVEMENT_IS_VALID,
    SMIRK_CAM,
    SMIRK_EXPRESSION_PARAMS,
    SMIRK_EYELID_PARAMS,
    SMIRK_IS_VALID,
    SMIRK_JAW_PARAMS,
    SMIRK_POSE_PARAMS,
    SMIRK_SHAPE_PARAMS,
    SMPLH_BODY_POSE,
    SMPLH_GLOBAL_ORIENT,
    SMPLH_IS_VALID,
    SMPLH_LEFT_HAND_POSE,
    SMPLH_RIGHT_HAND_POSE,
    SMPLH_TRANSLATION,
    SPEECH_TOKENS_L4,
    SR_TAG,
    WAV_TAG,
)
from utils.dataset_errors import ConfigurationError, DatasetError


def _tensorize_visual_feat(feature: np.ndarray[Any, Any]) -> torch.Tensor:
    """Reshape a visual feature tensor following the logic of the original audio_visual dataset
    Args:
        feature: Tensor to reshape
    """
    return torch.tensor(feature).squeeze().view(feature.shape[0], -1)


def _read_jsonl(path: Path | str, allow_missing: bool = True) -> List[Dict[str, Any]]:
    """Read a jsonl file and return a list of dicts
    Args:
        path: Path to the jsonl file
        allow_missing: Whether to allow missing file
    """
    if not Path(path).exists() and allow_missing:
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_smplh_file(
    features: Set[str], basepath: Path, filename: str, visual_rate: float
) -> Dict[str, Any]:
    """Load a SMPL-H file and return a dict
    Args:
        features: Set of SMPL-H h5py keys to load
        basepath: Path to the SMPL-H file
        filename: Filename of the SMPL-H file
        visual_rate: Rate of the visual features
    """
    smplh_features: Dict[str, Any] = {}
    if SMPLH_GLOBAL_ORIENT in features:
        smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{SMPLH_GLOBAL_ORIENT}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMPLH_GLOBAL_ORIENT}/{filename}.npy")
            )
        )
    if SMPLH_LEFT_HAND_POSE in features:
        smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{SMPLH_LEFT_HAND_POSE}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMPLH_LEFT_HAND_POSE}/{filename}.npy")
            )
        )
    if SMPLH_RIGHT_HAND_POSE in features:
        smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{SMPLH_RIGHT_HAND_POSE}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMPLH_RIGHT_HAND_POSE}/{filename}.npy")
            )
        )
    if SMPLH_BODY_POSE in features:
        smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{SMPLH_BODY_POSE}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMPLH_BODY_POSE}/{filename}.npy")
            )
        )
    if SMPLH_TRANSLATION in features:
        smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{SMPLH_TRANSLATION}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMPLH_TRANSLATION}/{filename}.npy")
            )
        )
    if SMPLH_IS_VALID in features:
        smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{SMPLH_IS_VALID}"] = (
            torch.from_numpy(np.load(f"{basepath}/{SMPLH_IS_VALID}/{filename}.npy")).to(
                dtype=torch.bool
            )
        )

    visual_durations = {len(feats) for feats in smplh_features.values()}
    if len(visual_durations) != 1:
        raise DatasetError(f"Expected exactly 1 duration, got: {visual_durations}")
    visual_duration = visual_durations.pop() / visual_rate
    smplh_features[f"{FEATURE_COLLECTION_SMPLH}_{DURATION_TAG}"] = visual_duration
    return smplh_features


def load_boxes_and_keypoints_file(
    features: Set[str], basepath: Path, filename: str, visual_rate: float
) -> Dict[str, Any]:
    """Load a boxes and keypoints file and return a dict
    Args:
        features: Set of boxes and keypoints features to load
        basepath: Path to the boxes and keypoints file
        filename: Filename of the boxes and keypoints file
        visual_rate: Rate of the visual features
    """
    boxes_and_keypoints_features: Dict[str, Any] = {}
    if BOXES_AND_KEYPOINTS_BOXES in features:
        boxes_and_keypoints_features[
            f"{FEATURE_COLLECTION_BOXES_AND_KEYPOINTS}_{feature}"
        ] = _tensorize_visual_feat(np.load(f"{basepath}/{feature}/{filename}.npy"))
    if BOXES_AND_KEYPOINTS_KEYPOINTS in features:
        boxes_and_keypoints_features[
            f"{FEATURE_COLLECTION_BOXES_AND_KEYPOINTS}_{feature}"
        ] = _tensorize_visual_feat(np.load(f"{basepath}/{feature}/{filename}.npy"))
    visual_durations = {len(feats) for feats in boxes_and_keypoints_features.values()}
    if len(visual_durations) != 1:
        raise DatasetError(f"Expected exactly 1 duration, got: {visual_durations}")
    visual_duration = visual_durations.pop() / visual_rate
    boxes_and_keypoints_features[
        f"{FEATURE_COLLECTION_BOXES_AND_KEYPOINTS}_{DURATION_TAG}"
    ] = visual_duration
    return boxes_and_keypoints_features


def load_movement_file(
    features: Set[str], basepath: Path, filename: str, visual_rate: float
) -> Dict[str, Any]:
    """Load a movement file and return a dict
    Args:
        features: Set of movement h5py keys to load
        basepath: Path to the movement file
        filename: Filename of the movement file
        visual_rate: Rate of the visual features
    """
    movement_features: Dict[str, Any] = {}
    if MOVEMENT_HEAD_ENCODINGS in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_HEAD_ENCODINGS}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_HEAD_ENCODINGS}/{filename}.npy")
        )
    if MOVEMENT_GAZE_ENCODINGS in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_GAZE_ENCODINGS}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_GAZE_ENCODINGS}/{filename}.npy")
        )
    if MOVEMENT_EXPRESSION in features:
        movement_features[f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_EXPRESSION}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{MOVEMENT_EXPRESSION}/{filename}.npy")
            )
        )
    if MOVEMENT_FRAME_LATENT in features:
        movement_features[f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_FRAME_LATENT}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{MOVEMENT_FRAME_LATENT}/{filename}.npy")
            )
        )
    if MOVEMENT_ALIGNMENT_HEAD_ROTATION in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_ALIGNMENT_HEAD_ROTATION}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_ALIGNMENT_HEAD_ROTATION}/{filename}.npy")
        )
    if MOVEMENT_ALIGNMENT_TRANSLATION in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_ALIGNMENT_TRANSLATION}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_ALIGNMENT_TRANSLATION}/{filename}.npy")
        )
    if MOVEMENT_EMOTION_SCORES in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_EMOTION_SCORES}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_EMOTION_SCORES}/{filename}.npy")
        )
    if MOVEMENT_EMOTION_VALENCE in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_EMOTION_VALENCE}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_EMOTION_VALENCE}/{filename}.npy")
        )
    if MOVEMENT_EMOTION_AROUSAL in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_EMOTION_AROUSAL}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_EMOTION_AROUSAL}/{filename}.npy")
        )
    if MOVEMENT_FAU_VALUE in features:
        movement_features[f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_FAU_VALUE}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{MOVEMENT_FAU_VALUE}/{filename}.npy")
            )
        )
    if MOVEMENT_HYPERNET_FEATURES in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_HYPERNET_FEATURES}"
        ] = _tensorize_visual_feat(
            np.load(f"{basepath}/{MOVEMENT_HYPERNET_FEATURES}/{filename}.npy")
        )

    if MOVEMENT_FAU_TOKEN in features:
        movement_features[f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_FAU_TOKEN}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{MOVEMENT_FAU_TOKEN}/{filename}.npy")
            )
            .long()
            .squeeze(1)
        )
    if MOVEMENT_EMOTION_VALENCE_TOKEN in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_EMOTION_VALENCE_TOKEN}"
        ] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{MOVEMENT_EMOTION_VALENCE_TOKEN}/{filename}.npy")
            )
            .long()
            .squeeze(1)
        )
    if MOVEMENT_EMOTION_AROUSAL_TOKEN in features:
        movement_features[
            f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_EMOTION_AROUSAL_TOKEN}"
        ] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{MOVEMENT_EMOTION_AROUSAL_TOKEN}/{filename}.npy")
            )
            .long()
            .squeeze(1)
        )

    if MOVEMENT_IS_VALID in features:
        movement_features[f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_IS_VALID}"] = (
            torch.from_numpy(np.load(f"{basepath}/{MOVEMENT_IS_VALID}/{filename}.npy"))
            .to(dtype=torch.bool)
            .squeeze(1)
        )

    # hypernet_features should have different length (e.g. 15x reduced)
    visual_durations = {
        (
            len(feats)
            if key != f"{FEATURE_COLLECTION_MOVEMENT}_{MOVEMENT_HYPERNET_FEATURES}"
            else len(feats) * MOVEMENT_HYPERNET_FEATURES_RATE
        )
        for key, feats in movement_features.items()
    }
    if (max(visual_durations) - min(visual_durations)) > visual_rate:
        raise DatasetError(
            f"Duration difference for movement features is too high: {visual_durations}"
        )
    visual_duration = visual_durations.pop() / visual_rate
    movement_features[f"{FEATURE_COLLECTION_MOVEMENT}_{DURATION_TAG}"] = visual_duration
    return movement_features


def load_metadata_file(
    basepath: Path, filename: str, features: Set[str]
) -> Dict[str, Any]:
    """Load a metadata file and return a dict
    Args:
        basepath: Path to the metadata file
        filename: Filename of the metadata file
        features: Set of metadata features to load
    """
    metadata_features: Dict[str, Any] = {}
    for feature in features:
        if feature == METADATA_VAD:
            metadata_features[f"{FEATURE_COLLECTION_METADATA}_{feature}"] = _read_jsonl(
                f"{basepath}/{feature}/{filename}.jsonl", allow_missing=False
            )
        elif feature == METADATA_TRANSCRIPT:
            text_items = _read_jsonl(
                f"{basepath}/{feature}/{filename}.jsonl", allow_missing=True
            )
            words = []
            for item in text_items:
                words.extend(item[METADATA_TRANSCRIPT_WORDS_KEY])
            metadata_features[f"{FEATURE_COLLECTION_METADATA}_{feature}"] = words
        else:
            metadata_features[f"{FEATURE_COLLECTION_METADATA}_{feature}"] = _read_jsonl(
                f"{basepath}/{feature}/{filename}.json", allow_missing=True
            )
    return metadata_features


def load_audio_file(
    features: Set[str], basepath: Path, filename: str
) -> Dict[str, Any]:
    """Load a wav file and return a dict
    Args:
        features: Set of audio features to load
        basepath: Path to the wav file
        filename: Filename of the wav file
    """
    audio_features: Dict[str, Any] = {}
    wav, sr = torchaudio.load(f"{basepath}/{filename}.wav")
    duration = wav.shape[-1] / sr
    if AUDIO_WAV_TENSOR in features:
        audio_features[f"{FEATURE_COLLECTION_AUDIO}_{WAV_TAG}"] = wav.squeeze(0)
        audio_features[f"{FEATURE_COLLECTION_AUDIO}_{SR_TAG}"] = torch.tensor(sr)
    audio_features[f"{FEATURE_COLLECTION_AUDIO}_{DURATION_TAG}"] = duration
    return audio_features


def load_annotations_file(
    features: Set[str], basepath: Path, filename: str
) -> Dict[str, Any]:
    """Load a annotations file and return a dict
    Args:
        basepath: Path to the annotations file
        filename: Filename of the annotations file
        features: Set of annotations features to load
    """
    annotations_features: Dict[str, Any] = {}
    if ANNOTATIONS_1PIS in features:
        annotations_features[f"{FEATURE_COLLECTION_ANNOTATIONS}_{ANNOTATIONS_1PIS}"] = (
            _read_jsonl(
                f"{basepath}/{ANNOTATIONS_1PIS}/{filename}.json", allow_missing=True
            )
        )
    if ANNOTATIONS_1PR in features:
        annotations_features[f"{FEATURE_COLLECTION_ANNOTATIONS}_{ANNOTATIONS_1PR}"] = (
            _read_jsonl(
                f"{basepath}/{ANNOTATIONS_1PR}/{filename}.json", allow_missing=True
            )
        )
    if ANNOTATIONS_3PIS in features:
        annotations_features[f"{FEATURE_COLLECTION_ANNOTATIONS}_{ANNOTATIONS_3PIS}"] = (
            _read_jsonl(
                f"{basepath}/{ANNOTATIONS_3PIS}/{filename}.json", allow_missing=True
            )
        )
    if ANNOTATIONS_3PR in features:
        annotations_features[f"{FEATURE_COLLECTION_ANNOTATIONS}_{ANNOTATIONS_3PR}"] = (
            _read_jsonl(
                f"{basepath}/{ANNOTATIONS_3PR}/{filename}.json", allow_missing=True
            )
        )
    if ANNOTATIONS_3PV in features:
        annotations_features[f"{FEATURE_COLLECTION_ANNOTATIONS}_{ANNOTATIONS_3PV}"] = (
            _read_jsonl(
                f"{basepath}/{ANNOTATIONS_3PV}/{filename}.json", allow_missing=True
            )
        )
    return annotations_features

    for feature in features:
        # TODO: add support for each annotation type

        # At the moment, this will load a list of dictionaries of {text, start_ts, end_ts}
        annotations_features[f"{FEATURE_COLLECTION_ANNOTATIONS}_{feature}"] = (
            _read_jsonl(f"{basepath}/{feature}/{filename}.json", allow_missing=True)
        )
    return annotations_features


def get_load_feature_fn(
    feature_basepath: Path, collection: str, sample: Dict[str, Any]
) -> Callable[..., Any]:
    """Get a function to load a feature from a file
    Args:
        collection: Name of the feature collection to load (ie smplh, smirk etc.)
        sample: Sample to load the feature from
    """
    feature_path = feature_basepath / collection
    if collection == FEATURE_COLLECTION_SMPLH:
        return partial(
            load_smplh_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
            visual_rate=sample[JSONL_KEY_VISUAL_RATE],
        )
    elif collection == FEATURE_COLLECTION_BOXES_AND_KEYPOINTS:
        return partial(
            load_boxes_and_keypoints_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
            visual_rate=sample[JSONL_KEY_VISUAL_RATE],
        )
    elif collection == FEATURE_COLLECTION_MOVEMENT:
        return partial(
            load_movement_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
            visual_rate=sample[JSONL_KEY_VISUAL_RATE],
        )
    elif collection == FEATURE_COLLECTION_METADATA:
        return partial(
            load_metadata_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
        )
    elif collection == FEATURE_COLLECTION_AUDIO:
        return partial(
            load_audio_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
        )
    elif collection == FEATURE_COLLECTION_ANNOTATIONS:
        return partial(
            load_annotations_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
        )
    else:
        raise ConfigurationError(f"Unsupported feature collection: {collection}")
