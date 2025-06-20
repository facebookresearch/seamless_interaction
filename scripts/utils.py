import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import numpy as np
import torch
import torchaudio

from .constants import (
    AUDIO_WAV_TENSOR,
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
from .errors import ConfigurationError, DatasetError


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
        elif feature == METADATA_TEXT:
            text_items = _read_jsonl(
                f"{basepath}/{feature}/{filename}.jsonl", allow_missing=True
            )
            # example of one text item:
            # {"words": [{"word": "Hello,", "start": 1, "end": 2, "score": 0.104}, {"word": "world", "start": 2, "end": 3, "score": 0.954}], "start": 1, "end": 3, "text": "Hello world"}
            # convert to the word list
            words = []
            for item in text_items:
                words.extend(item[METADATA_WORDS_KEY])
            metadata_features[f"{FEATURE_COLLECTION_METADATA}_{feature}"] = words
        else:
            metadata_features[f"{FEATURE_COLLECTION_METADATA}_{feature}"] = _read_jsonl(
                f"{basepath}/{feature}/{filename}.json", allow_missing=True
            )
    return metadata_features


def load_smirk_file(
    basepath: Path, filename: str, features: Set[str], visual_rate: float
) -> Dict[str, Any]:
    """Load a Smirk file and return a dict
    Args:
        basepath: Path to the Smirk file
        filename: Filename of the Smirk file
        features: Set of smirk h5py keys to load
        visual_rate: Rate of the visual features
    """
    smirk_features: Dict[str, Any] = {}
    if SMIRK_EXPRESSION_PARAMS in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_EXPRESSION_PARAMS}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMIRK_EXPRESSION_PARAMS}/{filename}.npy")
            )
        )
    if SMIRK_EYELID_PARAMS in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_EYELID_PARAMS}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMIRK_EYELID_PARAMS}/{filename}.npy")
            )
        )
    if SMIRK_JAW_PARAMS in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_JAW_PARAMS}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMIRK_JAW_PARAMS}/{filename}.npy")
            )
        )
    if SMIRK_CAM in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_CAM}"] = (
            _tensorize_visual_feat(np.load(f"{basepath}/{SMIRK_CAM}/{filename}.npy"))
        )
    if SMIRK_POSE_PARAMS in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_POSE_PARAMS}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMIRK_POSE_PARAMS}/{filename}.npy")
            )
        )
    if SMIRK_SHAPE_PARAMS in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_SHAPE_PARAMS}"] = (
            _tensorize_visual_feat(
                np.load(f"{basepath}/{SMIRK_SHAPE_PARAMS}/{filename}.npy")
            )
        )
    if SMIRK_IS_VALID in features:
        smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{SMIRK_IS_VALID}"] = (
            torch.from_numpy(np.load(f"{basepath}/{SMIRK_IS_VALID}/{filename}.npy")).to(
                dtype=torch.bool
            )
        )

    visual_durations = {len(feats) for feats in smirk_features.values()}
    if len(visual_durations) != 1:
        raise DatasetError(f"Expected exactly 1 duration, got: {visual_durations}")
    visual_duration = visual_durations.pop() / visual_rate
    smirk_features[f"{FEATURE_COLLECTION_SMIRK}_{DURATION_TAG}"] = visual_duration
    return smirk_features


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
    for movement_feature_key in [
        MOVEMENT_HEAD_ENCODINGS,
        MOVEMENT_GAZE_ENCODINGS,
        MOVEMENT_EXPRESSION,
        MOVEMENT_FRAME_LATENT,
        MOVEMENT_ALIGNMENT_HEAD_ROTATION,
        MOVEMENT_ALIGNMENT_TRANSLATION,
        MOVEMENT_EMOTION_SCORES,
        MOVEMENT_EMOTION_VALENCE,
        MOVEMENT_EMOTION_AROUSAL,
        MOVEMENT_FAU_VALUE,
        MOVEMENT_FAU_TOKEN,
        MOVEMENT_EMOTION_VALENCE_TOKEN,
        MOVEMENT_EMOTION_AROUSAL_TOKEN,
        MOVEMENT_HYPERNET_FEATURES,
    ]:
        if movement_feature_key in features:
            visual_feat = _tensorize_visual_feat(
                np.load(f"{basepath}/{movement_feature_key}/{filename}.npy")
            )
            # tokens should be of type long
            if movement_feature_key in [
                MOVEMENT_FAU_TOKEN,
                MOVEMENT_EMOTION_VALENCE_TOKEN,
                MOVEMENT_EMOTION_AROUSAL_TOKEN,
            ]:
                visual_feat = visual_feat.long().squeeze(1)

            movement_features[
                f"{FEATURE_COLLECTION_MOVEMENT}_{movement_feature_key}"
            ] = visual_feat

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


def load_audio_file(
    basepath: Path, filename: str, features: Set[str]
) -> Dict[str, Any]:
    """Load a wav file and return a dict
    Args:
        basepath: Path to the wav file
        filename: Filename of the wav file
        features: Set of audio features to load
    """
    audio_features = {}
    wav, sr = torchaudio.load(f"{basepath}/{filename}.wav")
    duration = wav.shape[-1] / sr
    if AUDIO_WAV_TENSOR in features:
        audio_features[f"{FEATURE_COLLECTION_AUDIO}_{WAV_TAG}"] = wav.squeeze(0)
        audio_features[f"{FEATURE_COLLECTION_AUDIO}_{SR_TAG}"] = torch.tensor(sr)
    audio_features[f"{FEATURE_COLLECTION_AUDIO}_{DURATION_TAG}"] = duration
    return audio_features


def load_speech_tokens_file(
    basepath: Path, filename: str, features: Set[str], speech_token_rate: float
) -> Dict[str, Any]:
    """Load a speech tokens file and return a dict with the speech tokens and duration
    Args:
        basepath: Path to the speech tokens file
        filename: Filename of the speech tokens file
        features: Set of speech tokens to load
        speech_token_rate: Rate of the speech tokens
    """
    if SPEECH_TOKENS_L4 not in features:
        return {}
    with open(f"{basepath}/{SPEECH_TOKENS_L4}/{filename}.json", "r") as f:
        speech_tokens_dict = json.load(f)
    if len(speech_tokens_dict) != 1:
        raise DatasetError(
            f"Expected exactly 1 speech token key, got: {speech_tokens_dict.keys()}"
        )

    speech_tokens: List[int] = speech_tokens_dict[SPEECH_TOKENS_L4]
    speech_token_duration = len(speech_tokens) / speech_token_rate
    return {
        f"{FEATURE_COLLECTION_SPEECH_TOKENS}_{SPEECH_TOKENS_L4}": torch.tensor(
            speech_tokens
        ),
        f"{FEATURE_COLLECTION_SPEECH_TOKENS}_{DURATION_TAG}": speech_token_duration,
    }


def get_load_feature_fn(
    feature_basepath: Path, collection: str, version: str, sample: Dict[str, Any]
) -> Callable[..., Any]:
    """Get a function to load a feature from a file
    Args:
        collection: Name of the feature collection to load (ie smplh, smirk etc.)
        version: Version of the feature collection to load
        sample: Sample to load the feature from
    """
    feature_path = feature_basepath / collection / version
    if collection == FEATURE_COLLECTION_SMIRK:
        return partial(
            load_smirk_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
            visual_rate=sample[JSONL_KEY_VISUAL_RATE],
        )
    elif collection == FEATURE_COLLECTION_SMPLH:
        return partial(
            load_smplh_file,
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
    elif collection == FEATURE_COLLECTION_SPEECH_TOKENS:
        return partial(
            load_speech_tokens_file,
            basepath=feature_path,
            filename=sample[JSONL_KEY_ID],
            speech_token_rate=sample[JSONL_KEY_SPEECH_TOKEN_RATE],
        )
    else:
        raise ConfigurationError(f"Unsupported feature collection: {collection}")
