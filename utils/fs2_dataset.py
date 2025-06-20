# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, final, List, Optional, Set

import yaml  # type: ignore[import]
from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import DataPipeline, read_sequence
from fairseq2.data.text import read_text
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.typing import override

from utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DURATION_MISMATCH_SECONDS_TOLERANCE,
    DEFAULT_NUM_PARALLEL_CALLS,
    DEFAULT_SEED,
    DYADIC_JSONL_KEY_ID_0,
    DYADIC_JSONL_KEY_ID_1,
    JSONL_KEY_ID,
    JSONL_KEY_VISUAL_RATE,
)
from utils.dataloader_utils import collate_seamless_communciation_data_batch, shuffle
from utils.errors import ConfigurationError, DatasetError
from utils.load_feature_utils import get_load_feature_fn
from utils.validations import (
    validate_feature_duration_consistency,
    validate_feature_name,
)

log = get_log_writer(__name__)


@dataclass
class DataBatch:
    data: Dict[str, Any]

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return len(next(iter(self.data.values())))


class SeamlessCommunicationDataset(ABC):
    """Interface for FS2 Seamless Communication dataset classes"""

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path) -> SeamlessCommunicationDataset:
        """Load data from yaml file (We intend for each split (train, val, test) to have its own yaml file))
        This is the entrypoint that defines the inputs to the dataset
        Args:
            path: Path to the yaml file containing the dataset configuration
        """

    @abstractmethod
    def create_reader(
        self,
        gang: Gang,
        npc: int = DEFAULT_NUM_PARALLEL_CALLS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_shuffle_window: int = 1,
        segment_shuffle_window: int = 1,
        num_prefetch: int = 1,
        num_accumulate: int = 1,
        mismatch_tolerance_seconds: float = DEFAULT_DURATION_MISMATCH_SECONDS_TOLERANCE,
        seed: int = DEFAULT_SEED,
    ) -> DataReader[DataBatch]:
        """Create a dataloader for a Seamless Communication dataset."""

    @abstractmethod
    def _load_sample(
        self, sample: Dict[str, Any], feature_basepath: Path
    ) -> Dict[str, Any]:
        """Load a single sample from the jsonl file (one line)
        Args:
            sample: Sample to load
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """


class GenericSeamlessCommunicationDataset(SeamlessCommunicationDataset):
    def __init__(
        self,
        jsonl_list: List[str],
        jsonl_weights: List[float],
        feature_list: List[str],
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
        feature_dir_list: Optional[List[str]] = None,
    ):
        """Supports base class for Seamless Communication dataset classes. In loads a list of jsonls and weights for each,
        along with a list of requested features (raw audio, raw visual, smplh etc) and applies some transforms to the data
        Args:
            jsonl_list: List of jsonl files containing samples to load
            jsonl_weights: List of weights to use for each jsonl file
            feature_list: List of features to load for each sample
            transforms: List of transforms to apply to each sample
            transforms_kwargs: List of kwargs to pass to the transform fns
            feature_dir_list: List of feature base directories, if given should have the same length as jsonl_list (default to parent directory of jsonl files)
        """
        if len(jsonl_list) != len(jsonl_weights):
            raise ConfigurationError(
                f"Found differing jsonl_list and jsonl_weights lengths: {len(jsonl_list)} != {len(jsonl_weights)}"
            )
        self.jsonl_list = jsonl_list
        self.jsonl_weights = jsonl_weights
        self.feature_dict = self._prepare_feature_dict(feature_list)
        self.transform_fns = self._prepare_transforms(transforms, transforms_kwargs)
        if feature_dir_list is not None:
            if len(feature_dir_list) != len(jsonl_list):
                raise ConfigurationError(
                    f"Found differing feature_dir_list and jsonl_list lengths: {len(feature_dir_list)} != {len(jsonl_list)}"
                )
            self.feature_dir_list = feature_dir_list
        else:
            self.feature_dir_list = [
                str(Path(jsonl_path).parent) for jsonl_path in jsonl_list
            ]

    @override
    def create_reader(
        self,
        gang: Gang,
        npc: int = DEFAULT_NUM_PARALLEL_CALLS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_shuffle_window: int = 1,
        segment_shuffle_window: int = 1,
        num_prefetch: int = 1,
        num_accumulate: int = 1,
        mismatch_tolerance_seconds: float = DEFAULT_DURATION_MISMATCH_SECONDS_TOLERANCE,
        seed: int = DEFAULT_SEED,
    ) -> DataReader[DataBatch]:
        """Create a reader for the dataset. This defines the pipeline that loads and serves batches of data.
        Args:
            gang: Gang to use for the reader
            segment_size: Size of the segments to read
            npc: Number of parallel calls to use (torch num_workers)
            batch_size: Size of the batches to read
            batch_shuffle_window: Size of the sliding window for shuffling batches
            segment_shuffle_window: Size of the sliding window for shuffling segments within each batch.
            num_prefetch: Number of batches to prefetch in background
            num_accumulate: Number of batches to accumulate before yielding
            mismatch_tolerance_seconds: Maximum allowed difference in seconds between the durations of two fields.
            seed: Seed to use for the reader
        """
        self.mismatch_tolerance_seconds = mismatch_tolerance_seconds

        # For each jsonl file, load its data and apply its transforms
        data = [
            self._load_jsonl(Path(jsonl_path), Path(feature_dir), gang, npc)
            for jsonl_path, feature_dir in zip(self.jsonl_list, self.feature_dir_list)
        ]

        # Combine all the jsonl loaded samples into a single dataset, with sample weights `data_weights`
        log.info("[1/4] Loading jsonl samples...")
        dataset_builder = DataPipeline.sample(
            data,
            weights=self.jsonl_weights,
            seed=seed,
            allow_repeats=True,  # repeat loading shorter data when finished
        )

        # Shuffle segments
        log.info("[2/4] Shuffling individual data segments...")
        seed, dataset_builder = shuffle(seed, segment_shuffle_window, dataset_builder)

        # Bucket batch and shuffle batches
        log.info("[3/4] Creating and shuffling batches...")
        dataset_builder.bucket(batch_size, drop_remainder=False)
        seed, dataset_builder = shuffle(seed, batch_shuffle_window, dataset_builder)

        # Prefetch `num_prefetch` examples in background.
        dataset_builder.prefetch(num_prefetch)

        # Collate and serve the batches
        log.info("[4/4] Collating batches...")
        dataset_builder.map(
            partial(
                collate_seamless_communication_data_batch,
                gang=gang,
            ),
            num_parallel_calls=npc,
        )
        pipeline = dataset_builder.and_return()
        return DataPipelineReader[DataBatch](
            pipeline=pipeline,
            num_accumulate=num_accumulate,
            gang=gang,
            drop_remainder=False,
            sync_batches=True,
        )

    def _prepare_feature_dict(self, feature_list: List[str]) -> Dict[str, Set[str]]:
        """Prepare the feature dict for the dataset. Makes sure that each requested feature follows the valid
        formatting convention <collection>:<feature_name> and prepares a dict to load feature_name keys from each h5py collection.
        Args:
            feature_list: List of features to load
        """
        feature_dict: Dict[str, Set[str]] = {}
        for feature in feature_list:
            split_feature_name = feature.split(":")
            if len(split_feature_name) != 2:
                raise ConfigurationError(
                    f"Provided feature name {feature} which does not fit <collection>:<feature_name> convention!"
                )
            collection, feature_name = split_feature_name
            validate_feature_name(collection, feature_name)
            feature_dict.setdefault(collection, set()).add(feature_name)
        return feature_dict

    def _prepare_transforms(
        self,
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
    ) -> List[partial[Any]]:
        """Prepare the transforms for the dataset. Make sure that the transforms are callable and the kwargs are dictionaries
        Args:
            transforms: List of transforms to apply to each sample
            transforms_kwargs: List of kwargs to pass to the transform fns
        """
        if len(transforms) != len(transforms_kwargs):
            raise ConfigurationError(
                f"Found differing lengths for transforms and transforms_kwargs ({len(transforms)} != {len(transforms_kwargs)})"
            )

        transform_fns = []
        for transform, kwargs in zip(transforms, transforms_kwargs):
            if not callable(transform):
                raise ConfigurationError(f"Transform {transform} is not callable")
            if not isinstance(kwargs, dict):
                raise ConfigurationError(
                    f"Transform kwargs {kwargs} is not a dictionary"
                )
            try:
                transform_fns.append(partial(transform, **kwargs))
            except Exception as e:
                raise DatasetError(
                    f"Failed to create partial function for transform {transform}: {e}"
                )
        return transform_fns

    def _load_jsonl(
        self, jsonl_path: Path, feature_basepath: Path, gang: Gang, npc: int
    ) -> DataPipeline:
        """Load jsonl files from manifold and apply transforms to each sample
        Args:
            jsonl_path: Path to the jsonl file to load
            feature_basepath: Path to the base directory of the features
            gang: Gang to use for the reader
            npc: Number of parallel calls to use (torch num_workers)
        """
        jsonl_builder = (
            read_text(jsonl_path)
            .shard(gang.rank, gang.size)
            .map(json.loads, num_parallel_calls=npc)
        )

        # Load the data from each jsonl
        log.info(f"[1a] Loading data samples from jsonl `{jsonl_path}`")
        jsonl_builder.map(
            partial(
                self._load_sample,
                feature_basepath=feature_basepath,
            ),
            num_parallel_calls=npc,
        )

        # Apply transforms to each sample
        log.info(
            f"[1b] Applying transforms to loaded features from data, gotten from `{jsonl_path}`"
        )
        for transform_fn in self.transform_fns:
            jsonl_builder.map(
                transform_fn,
                num_parallel_calls=npc,
            )
            jsonl_builder.yield_from(
                lambda x: read_sequence(x).and_return() if isinstance(x, list) else x
            )

        return jsonl_builder.and_return()

    def _load_feature_set(
        self,
        sample: Dict[str, Any],
        collection: str,
        features: Set[str],
        feature_basepath: Path,
    ) -> Dict[str, Any]:
        """Load a single feature set for a sample from a jsonl file
        Args:
            sample: Sample to load
            collection: name of feature collection to load (ie smirk, metadata, smplh etc.)
            features: list of feature key strings to load from each h5py collection
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """
        load_feature_fn = get_load_feature_fn(feature_basepath, collection, sample)
        feature_data: Dict[str, Any] = load_feature_fn(features=features)
        return feature_data


class GenericSeamlessCommunicationMonadicDataset(GenericSeamlessCommunicationDataset):
    """Seamless Communication dataset class for monadic data (ie audio, visual, metadata etc)
    The input jsonl should contain a single id per line.
    """

    def __init__(
        self,
        jsonl_list: List[str],
        jsonl_weights: List[float],
        feature_list: List[str],
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
        feature_dir_list: Optional[List[str]] = None,
    ):
        super().__init__(
            jsonl_list,
            jsonl_weights,
            feature_list,
            transforms,
            transforms_kwargs,
            feature_dir_list,
        )

    @classmethod
    @override
    def from_path(cls, path: Path) -> GenericSeamlessCommunicationMonadicDataset:
        """Load data from yaml file (We intend for each split (train, val, test) to have its own yaml file))
        This is the entrypoint that defines the inputs to the dataset

        Args:
            path: Path to the yaml file containing the dataset configuration
        """
        if not path.suffix == ".yaml":
            raise ValueError(f"Expected a yaml data file, found {str(path)}!")
        with path.open() as fp:
            content = yaml.safe_load(fp)
        return GenericSeamlessCommunicationMonadicDataset(
            jsonl_list=content["jsonl_list"],
            jsonl_weights=content["jsonl_weights"],
            feature_list=content["feature_list"],
            transforms=[
                getattr(transforms, transform) for transform in content["transforms"]
            ],
            transforms_kwargs=content["transforms_kwargs"],
            feature_dir_list=content.get("feature_dir_list", None),
        )

    @override
    def _load_sample(
        self, sample: Dict[str, Any], feature_basepath: Path
    ) -> Dict[str, Any]:
        """Load a single monadic sample from the jsonl file (one line)
        Args:
            sample: Sample to load
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """
        sample_id = sample[JSONL_KEY_ID]
        sample_features = {}
        for collection, features in self.feature_dict.items():
            sample_features.update(
                self._load_feature_set(sample, collection, features, feature_basepath)
            )

        # Assert that the durations for all features are roughly equal
        sample_features = validate_feature_duration_consistency(
            sample_id, sample_features, self.mismatch_tolerance_seconds
        )
        return {
            **sample_features,
            JSONL_KEY_ID: sample_id,
            JSONL_KEY_VISUAL_RATE: sample[JSONL_KEY_VISUAL_RATE],
            JSONL_KEY_SPEECH_TOKEN_RATE: sample[JSONL_KEY_SPEECH_TOKEN_RATE],
        }


class GenericSeamlessCommunicationDyadicDataset(GenericSeamlessCommunicationDataset):
    """Seamless Communication dataset class for dyadic data (ie audio, visual, metadata etc)
    The input jsonl should contain a two ids per line.
    """

    def __init__(
        self,
        jsonl_list: List[str],
        jsonl_weights: List[float],
        feature_list: List[str],
        transforms: List[Callable[..., Any]],
        transforms_kwargs: List[Dict[str, Any]],
        feature_dir_list: Optional[List[str]] = None,
    ):
        super().__init__(
            jsonl_list,
            jsonl_weights,
            feature_list,
            transforms,
            transforms_kwargs,
            feature_dir_list,
        )

    @classmethod
    @override
    def from_path(cls, path: Path) -> GenericSeamlessCommunicationDyadicDataset:
        """Load data from yaml file (We intend for each split (train, val, test) to have its own yaml file))
        This is the entrypoint that defines the inputs to the dataset

        Args:
            path: Path to the yaml file containing the dataset configuration
        """
        if not path.suffix == ".yaml":
            raise ValueError(f"Expected a yaml data file, found {str(path)}!")
        with path.open() as fp:
            content = yaml.safe_load(fp)
        return GenericSeamlessCommunicationDyadicDataset(
            jsonl_list=content["jsonl_list"],
            jsonl_weights=content["jsonl_weights"],
            feature_list=content["feature_list"],
            transforms=content["transforms"],
            transforms_kwargs=content["transforms_kwargs"],
            feature_dir_list=content.get("feature_dir_list", None),
        )

    @override
    def _load_sample(
        self, sample: Dict[str, Any], feature_basepath: Path
    ) -> Dict[str, Any]:
        """Load a single dyadic sample from the jsonl file (one line)
        Args:
            sample: Sample to load
            feature_basepath: Path to the basepath of the features (ie basepath/smirk)
        """
        sample_spkr0 = sample[DYADIC_JSONL_KEY_ID_0]
        sample_spkr1 = sample[DYADIC_JSONL_KEY_ID_1]
        sample_features_spkr0 = {}
        sample_features_spkr1 = {}
        for collection, features in self.feature_dict.items():
            sample_features_spkr0.update(
                self._load_feature_set(
                    sample_spkr0, collection, features, feature_basepath
                )
            )
            sample_features_spkr1.update(
                self._load_feature_set(
                    sample_spkr1, collection, features, feature_basepath
                )
            )

        # Assert that the durations for all features are roughly equal
        sample_features_spkr0 = validate_feature_duration_consistency(
            sample_spkr0[JSONL_KEY_ID],
            sample_features_spkr0,
            self.mismatch_tolerance_seconds,
        )
        sample_features_spkr1 = validate_feature_duration_consistency(
            sample_spkr1[JSONL_KEY_ID],
            sample_features_spkr1,
            self.mismatch_tolerance_seconds,
        )
        return {
            DYADIC_JSONL_KEY_ID_0: {
                **sample_features_spkr0,
                JSONL_KEY_ID: sample_spkr0[JSONL_KEY_ID],
                JSONL_KEY_VISUAL_RATE: sample_spkr0[JSONL_KEY_VISUAL_RATE],
                JSONL_KEY_SPEECH_TOKEN_RATE: sample_spkr0[JSONL_KEY_SPEECH_TOKEN_RATE],
            },
            DYADIC_JSONL_KEY_ID_1: {
                **sample_features_spkr1,
                JSONL_KEY_ID: sample_spkr1[JSONL_KEY_ID],
                JSONL_KEY_VISUAL_RATE: sample_spkr1[JSONL_KEY_VISUAL_RATE],
                JSONL_KEY_SPEECH_TOKEN_RATE: sample_spkr1[JSONL_KEY_SPEECH_TOKEN_RATE],
            },
        }


@final
class GenericSeamlessCommunicationMonadicDatasetLoader(
    AbstractDatasetLoader[GenericSeamlessCommunicationMonadicDataset]
):
    @override
    def _load(
        self, path: Path, card: AssetCard
    ) -> GenericSeamlessCommunicationMonadicDataset:
        try:
            return GenericSeamlessCommunicationMonadicDataset.from_path(path=path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


@final
class GenericSeamlessCommunicationDyadicDatasetLoader(
    AbstractDatasetLoader[GenericSeamlessCommunicationDyadicDataset]
):
    @override
    def _load(
        self, path: Path, card: AssetCard
    ) -> GenericSeamlessCommunicationDyadicDataset:
        try:
            return GenericSeamlessCommunicationDyadicDataset.from_path(path=path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_seamless_communication_monadic_dataset = DelegatingDatasetLoader[
    GenericSeamlessCommunicationMonadicDataset
]()

load_generic_seamless_communication_monadic_dataset = (
    GenericSeamlessCommunicationMonadicDatasetLoader()
)

load_seamless_communication_monadic_dataset.register(
    "generic_seamless_communication_monadic",
    load_seamless_communication_monadic_dataset,
)

load_seamless_communication_dyadic_dataset = DelegatingDatasetLoader[
    GenericSeamlessCommunicationDyadicDataset
]()

load_generic_seamless_communication_dyadic_dataset = (
    GenericSeamlessCommunicationDyadicDatasetLoader()
)

load_seamless_communication_dyadic_dataset.register(
    "generic_seamless_communication_dyadic", load_seamless_communication_dyadic_dataset
)
