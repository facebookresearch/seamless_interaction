from typing import Any, Dict

from utils.constants import (
    DURATION_TAG,
    FEATURE_COLLECTION_ANNOTATIONS,
    FEATURE_COLLECTION_AUDIO,
    FEATURE_COLLECTION_BOXES_AND_KEYPOINTS,
    FEATURE_COLLECTION_MOVEMENT,
    FEATURE_COLLECTION_SMPLH,
    RATE_TAG,
    VALID_ANNOTATION_FEATURES,
    VALID_AUDIO_FEATURES,
    VALID_BOXES_AND_KEYPOINTS_FEATURES,
    VALID_COLLECTIONS,
    VALID_MOVEMENT_FEATURES,
    VALID_SMPLH_FEATURES,
)


def validate_feature_duration_consistency(
    sample_id: str, sample_features: Dict[str, Any], duration_mismatch_tolerance: float
) -> Dict[str, Any]:
    """Validate that the duration is consistent across features in a sample
    Return a single duration key to be used for all features if they are consistent
    Args:
        sample_id: ID of the sample
        sample_features: Features of the sample
        duration_mismatch_tolerance: Tolerance for duration mismatch (seconds)
    """
    feature_durations = {
        feature: duration
        for feature, duration in sample_features.items()
        if feature.endswith(DURATION_TAG)
    }
    if (
        abs(max(feature_durations.values()) - min(feature_durations.values()))
        >= duration_mismatch_tolerance
    ):
        raise ValueError(
            f"Found mismatched durations exceeding {duration_mismatch_tolerance}s for sample {sample_id}: {feature_durations}"
        )
    sample_duration: float = next(iter(feature_durations.values()))
    return {
        DURATION_TAG: sample_duration,
        **{k: v for k, v in sample_features.items() if not k.endswith(DURATION_TAG)},
    }


def validate_batch_feature_rate_consistency(feature: str, data: Any) -> None:
    """Validate that the rate is consistent across samples in a batch.
    Args:
        feature: Name of the feature
        data: Data of the feature
    """
    if feature.endswith(RATE_TAG) and data.unique().numel() != 1:
        raise ValueError(
            f"Feature {feature} has different rates across samples: {data.unique()}"
        )


def validate_feature_name(collection: str, feature_name: str) -> None:
    """Check that a feature is valid for a given collection
    Args:
        collection: Collection to check
        feature_name: Feature name to check
    """
    if collection not in VALID_COLLECTIONS:
        raise ValueError(f"Provided invalid feature collection {collection}!")
    if collection == FEATURE_COLLECTION_SMPLH:
        if feature_name not in VALID_SMPLH_FEATURES:
            raise ValueError(
                f"Provided invalid feature name {feature_name} for collection {collection}!"
            )
    elif collection == FEATURE_COLLECTION_BOXES_AND_KEYPOINTS:
        if feature_name not in VALID_BOXES_AND_KEYPOINTS_FEATURES:
            raise ValueError(
                f"Provided invalid feature name {feature_name} for collection {collection}!"
            )
    elif collection == FEATURE_COLLECTION_MOVEMENT:
        if feature_name not in VALID_MOVEMENT_FEATURES:
            raise ValueError(
                f"Provided invalid feature name {feature_name} for collection {collection}!"
            )
    elif collection == FEATURE_COLLECTION_AUDIO:
        if feature_name not in VALID_AUDIO_FEATURES:
            raise ValueError(
                f"Provided invalid feature name {feature_name} for collection {collection}!"
            )
    elif collection == FEATURE_COLLECTION_ANNOTATIONS:
        if feature_name not in VALID_ANNOTATION_FEATURES:
            raise ValueError(
                f"Provided invalid feature name {feature_name} for collection {collection}!"
            )
