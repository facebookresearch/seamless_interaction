from typing import Final, List, Set

# Number of workers used by dataloader
DEFAULT_NUM_PARALLEL_CALLS: Final[int] = 10
DEFAULT_BATCH_SIZE: Final[int] = 4
DEFAULT_SEED: Final[int] = 42
DEFAULT_DURATION_MISMATCH_SECONDS_TOLERANCE: Final[float] = 0.5

# Common tags
RATE_TAG: Final[str] = "rate"
DURATION_TAG: Final[str] = "duration"
WAV_TAG: Final[str] = "wav"
SR_TAG: Final[str] = "sr"
FILELIST: Final[str] = "filelist"
SAMPLE_ID: Final[str] = "sample_id"
AUDIOVISUAL_SECOND_PADDING: float = 0.4

# Feature identifiers (in jsonl)
JSONL_KEY_ID: Final[str] = "id"
JSONL_KEY_VISUAL_RATE: Final[str] = "visual_rate"

# Feature identifiers (in dyadic jsonl)
DYADIC_JSONL_KEY_ID_0: Final[str] = "id_0"
DYADIC_JSONL_KEY_ID_1: Final[str] = "id_1"
DYADIC_VAD = "metadata_vad"
DYADIC_TRANSCRIPT = "metadata_transcript"

# Feature types
FEATURE_COLLECTION_AUDIO: Final[str] = "audio"
FEATURE_COLLECTION_METADATA: Final[str] = "metadata"
FEATURE_COLLECTION_SMPLH: Final[str] = "smplh"
FEATURE_COLLECTION_MOVEMENT: Final[str] = "movement"
FEATURE_COLLECTION_BOXES_AND_KEYPOINTS: Final[str] = "boxes_and_keypoints"
FEATURE_COLLECTION_ANNOTATIONS: Final[str] = "annotations"
VALID_COLLECTIONS = {
    FEATURE_COLLECTION_AUDIO,
    FEATURE_COLLECTION_BOXES_AND_KEYPOINTS,
    FEATURE_COLLECTION_METADATA,
    FEATURE_COLLECTION_SMPLH,
    FEATURE_COLLECTION_MOVEMENT,
    FEATURE_COLLECTION_ANNOTATIONS,
}

ANNOTATIONS_3PV: Final[str] = "3P-V"
ANNOTATIONS_3PIS: Final[str] = "3P-IS"
ANNOTATIONS_3PR: Final[str] = "3P-R"
ANNOTATIONS_1PIS: Final[str] = "1P-IS"
ANNOTATIONS_1PR: Final[str] = "1P-R"
VALID_ANNOTATION_FEATURES: Final[Set[str]] = {
    ANNOTATIONS_3PV,
    ANNOTATIONS_3PIS,
    ANNOTATIONS_3PR,
    ANNOTATIONS_1PIS,
    ANNOTATIONS_1PR,
}

# Labels for SMPL-H features (h5py subkeys)
SMPLH_GLOBAL_ORIENT: Final[str] = "global_orient"
SMPLH_LEFT_HAND_POSE: Final[str] = "left_hand_pose"
SMPLH_RIGHT_HAND_POSE: Final[str] = "right_hand_pose"
SMPLH_BODY_POSE: Final[str] = "body_pose"
SMPLH_TRANSLATION: Final[str] = "translation"  # AKA: pred_cam_t
SMPLH_IS_VALID: Final[str] = "is_valid"
VALID_SMPLH_FEATURES: Final[Set[str]] = {
    SMPLH_GLOBAL_ORIENT,
    SMPLH_LEFT_HAND_POSE,
    SMPLH_RIGHT_HAND_POSE,
    SMPLH_BODY_POSE,
    SMPLH_TRANSLATION,
    SMPLH_IS_VALID,
}

BOXES_AND_KEYPOINTS_BOXES: Final[str] = "boxes"
BOXES_AND_KEYPOINTS_KEYPOINTS: Final[str] = "keypoints"
VALID_BOXES_AND_KEYPOINTS_FEATURES: Final[Set[str]] = {
    BOXES_AND_KEYPOINTS_BOXES,
    BOXES_AND_KEYPOINTS_KEYPOINTS,
}

# Labels for movement features
MOVEMENT_HEAD_ENCODINGS: Final[str] = "head_encodings"
MOVEMENT_GAZE_ENCODINGS: Final[str] = "gaze_encodings"
MOVEMENT_EXPRESSION: Final[str] = "expression"
MOVEMENT_FRAME_LATENT: Final[str] = "frame_latent"
MOVEMENT_ALIGNMENT_HEAD_ROTATION: Final[str] = "alignment_head_rotation"
MOVEMENT_ALIGNMENT_TRANSLATION: Final[str] = "alignment_translation"
MOVEMENT_EMOTION_SCORES: Final[str] = "emotion_scores"
MOVEMENT_EMOTION_VALENCE: Final[str] = "emotion_valence"
MOVEMENT_EMOTION_AROUSAL: Final[str] = "emotion_arousal"
MOVEMENT_FAU_VALUE: Final[str] = "FAUValue"
MOVEMENT_FAU_TOKEN: Final[str] = "FAUToken"
MOVEMENT_EMOTION_VALENCE_TOKEN: Final[str] = "EmotionValenceToken"
MOVEMENT_EMOTION_AROUSAL_TOKEN: Final[str] = "EmotionArousalToken"
MOVEMENT_HYPERNET_FEATURES: Final[str] = "hypernet_features"
MOVEMENT_IS_VALID: Final[str] = "is_valid"
MOVEMENT_HYPERNET_FEATURES_RATE: Final[float] = (
    15  # hypernet_features are computed every 15 frames (sampled at 500ms when fps = 30)
)
VALID_MOVEMENT_FEATURES: Final[Set[str]] = {
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
    MOVEMENT_IS_VALID,
}

# Valid audio features
AUDIO_WAV_TENSOR: Final[str] = "waveform"
VALID_AUDIO_FEATURES: Final[Set[str]] = {
    AUDIO_WAV_TENSOR,
}

# Metadata
METADATA_VAD: Final[str] = "vad"
METADATA_TRANSCRIPT: Final[str] = "transcript"
METADATA_TRANSCRIPT_WORDS_KEY: Final[str] = "words"
