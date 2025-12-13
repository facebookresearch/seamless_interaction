# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import wget
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from seamless_interaction.constants import ALL_FEATURES
from seamless_interaction.utils import recursively_cast_to_float32, setup_logging

logger = setup_logging(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset access and processing."""

    # Data source settings
    label: str = "improvised"
    split: str = "dev"
    preferred_vendors_only: bool = True

    # Processing settings
    num_workers: int | None = None  # Will be auto-detected if None
    local_dir: str | None = None  # Will use default if None

    # Sampling settings
    seed: int | None = None

    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = min(10, max(1, os.cpu_count() - 2))
        if self.local_dir is None:
            self.local_dir = str(Path.home() / "datasets/seamless_interaction")


@dataclass
class InteractionKey:
    """Structured representation of interaction identifiers."""

    vendor: str
    session: str
    interaction: str | None = None
    participant: str | None = None

    @classmethod
    def from_file_id(cls, file_id: str) -> "InteractionKey":
        """Parse file ID into structured components."""
        match = re.match(r"V(\d+)_S(\d+)_I(\d+)_P(\d+)", file_id)
        if not match:
            raise ValueError(f"Invalid file ID format: {file_id}")
        vendor, session, interaction, participant = match.groups()
        return cls(vendor, session, interaction, participant)

    @classmethod
    def from_interaction_key(cls, interaction_key: str) -> "InteractionKey":
        """Parse interaction key into structured components."""
        match = re.match(r"V(\d+)_S(\d+)_I(\d+)", interaction_key)
        if not match:
            raise ValueError(f"Invalid interaction key format: {interaction_key}")
        vendor, session, interaction = match.groups()
        return cls(vendor, session, interaction)

    @classmethod
    def from_session_key(cls, session_key: str) -> "InteractionKey":
        """Parse session key into structured components."""
        match = re.match(r"V(\d+)_S(\d+)", session_key)
        if not match:
            raise ValueError(f"Invalid session key format: {session_key}")
        vendor, session = match.groups()
        return cls(vendor, session, None)

    @property
    def file_id(self) -> str:
        """Generate full file ID."""
        if not self.participant or not self.interaction:
            raise ValueError("Participant required for file ID")
        return f"V{self.vendor}_S{self.session}_I{self.interaction}_P{self.participant}"

    @property
    def interaction_key(self) -> str:
        """Generate interaction key."""
        if not self.interaction:
            raise ValueError("Interaction required for interaction key")
        return f"V{self.vendor}_S{self.session}_I{self.interaction}"

    @property
    def session_key(self) -> str:
        """Generate session key."""
        return f"V{self.vendor}_S{self.session}"


class SeamlessInteractionFS:
    """
    Filesystem interface for the Seamless Interaction Dataset <-> S3 & HF.
    """

    # S3 configuration
    _bucket: Final[str] = "dl.fbaipublicfiles.com"
    _prefix: Final[str] = "seamless_interaction"

    # HuggingFace configuration
    _hf_api: HfApi = HfApi()
    _hf_fs: HfFileSystem | None = None
    _hf_repo_id: Final[str] = "facebook/seamless-interaction"
    _hf_repo_type: Final[str] = "dataset"

    # Default configuration
    _default_config = DatasetConfig()
    _filelist_path: str = str(
        Path(__file__).parent.parent.parent / "assets/filelist.csv"
    )

    # Cache
    _cached_filelist: pd.DataFrame = None
    _cached_file_id_to_label_split: dict = {}
    _dry_run: bool = False

    def __init__(
        self,
        *,
        config: DatasetConfig | None = None,
        local_dir: str | None = None,
        filelist_path: str | None = None,
        dry_run: bool = False,
        num_workers: int | None = None,
    ) -> None:
        """
        Initialize the filesystem interface.

        :param config: Dataset configuration object
        :param local_dir: Local directory for downloads
        :param filelist_path: Path to dataset filelist
        :param dry_run: Whether to run in dry-run mode
        :param num_workers: Number of parallel workers
        """
        # Use provided config or create default
        self.config = config or DatasetConfig()

        # Override config values with explicit parameters
        if local_dir:
            self.config.local_dir = local_dir
        if num_workers:
            self.config.num_workers = num_workers

        # Set other configuration
        self._filelist_path = filelist_path or self._filelist_path
        self._dry_run = dry_run

        # Create local directory
        try:
            if self.config.local_dir is None:
                raise ValueError("Please configure local_dir")
            os.makedirs(self.config.local_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create local directory: {e}")
            raise e

        # Initialize cache
        self._load_filelist_cache()

    def _load_filelist_cache(self) -> None:
        """Load and cache the filelist for faster operations."""
        try:
            self._cached_filelist = pd.read_csv(self._filelist_path)
            logger.info(f"Loaded filelist with {len(self._cached_filelist)} entries")
        except Exception as e:
            logger.warning(f"Could not load filelist: {e}")
            self._cached_filelist = pd.DataFrame()

    @property
    def num_workers(self) -> int:
        return self.config.num_workers or 0

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self.config.num_workers = value

    def _filter_candidates(
        self,
        label: str | None = None,
        split: str | None = None,
        preferred_vendors_only: bool | None = None,
    ) -> pd.DataFrame:
        """
        Common filtering logic for dataset candidates.

        :param label: Dataset label (uses config default if None)
        :param split: Dataset split (uses config default if None)
        :param preferred_vendors_only: Filter by preferred vendors (uses config default if None)
        :return: Filtered DataFrame of candidates
        """
        if self._cached_filelist is None or self._cached_filelist.empty:
            self._load_filelist_cache()

        label = label or self.config.label
        split = split or self.config.split
        preferred_vendors_only = (
            preferred_vendors_only
            if preferred_vendors_only is not None
            else self.config.preferred_vendors_only
        )

        # Filter by label and split
        candidates = self._cached_filelist[
            (self._cached_filelist["label"] == label)
            & (self._cached_filelist["split"] == split)
        ]

        # Filter by preferred vendors if requested
        if preferred_vendors_only:
            preferred_vendors = self._get_preferred_vendors()
            vendor_mask = candidates["file_id"].str.startswith(tuple(preferred_vendors))
            candidates = candidates[vendor_mask]

        return candidates

    def _get_preferred_vendors(self) -> list[str]:
        """Get preferred vendors with smaller file sizes."""
        return ["V00", "V01"]

    def _group_by_pattern(
        self, candidates: pd.DataFrame, pattern: str
    ) -> pd.core.groupby.DataFrameGroupBy:
        """
        Group candidates by a regex pattern.

        :param candidates: DataFrame of file candidates
        :param pattern: Regex pattern for grouping
        :return: Grouped DataFrame
        """
        return candidates.groupby(candidates["file_id"].str.extract(pattern)[0])

    def get_available_keys(
        self,
        key_type: Literal["session", "interaction", "file"],
        label: str | None = None,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """
        Get available keys for exploration (useful for streamlit dropdowns).

        :param key_type: Type of keys to retrieve
        :param label: Dataset label
        :param split: Dataset split
        :param limit: Maximum number of keys to return
        :return: List of available keys
        """
        candidates = self._filter_candidates(label, split, preferred_vendors_only=False)

        if key_type == "session":
            keys = (
                candidates["file_id"].str.extract(r"(V\d+_S\d+)_I\d+_P\d+")[0].unique()
            )
        elif key_type == "interaction":
            keys = (
                candidates["file_id"].str.extract(r"(V\d+_S\d+_I\d+)_P\d+")[0].unique()
            )
        elif key_type == "file":
            keys = candidates["file_id"].unique()
        else:
            raise ValueError(f"Invalid key_type: {key_type}")

        keys = sorted(keys)
        return keys[:limit] if limit else keys

    def get_interaction_metadata(self, interaction_key: str) -> dict:
        """
        Get metadata for a specific interaction (useful for streamlit display).

        :param interaction_key: Interaction key (V00_S0809_I00000582)
        :return: Dictionary with interaction metadata
        """
        key_obj = InteractionKey.from_interaction_key(interaction_key)

        # Find all files for this interaction
        candidates = self._filter_candidates(preferred_vendors_only=False)
        interaction_files = candidates[
            candidates["file_id"].str.startswith(interaction_key)
        ]

        if interaction_files.empty:
            return {}

        # Extract metadata
        sample_row = interaction_files.iloc[0]
        participants = (
            interaction_files["file_id"]
            .str.extract(r"V\d+_S\d+_I\d+_P(\d+)")[0]
            .tolist()
        )

        return {
            "interaction_key": interaction_key,
            "session_key": key_obj.session_key,
            "vendor": key_obj.vendor,
            "session": key_obj.session,
            "interaction": key_obj.interaction,
            "participants": participants,
            "num_participants": len(participants),
            "label": sample_row["label"],
            "split": sample_row["split"],
            "batch_idx": sample_row["batch_idx"],
            "archive_idx": sample_row["archive_idx"],
            "file_ids": interaction_files["file_id"].tolist(),
        }

    def sample_random_file_ids(
        self,
        num_samples: int = 1,
        label: str | None = None,
        split: str | None = None,
        preferred_vendors_only: bool | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """
        Sample random file IDs with intelligent vendor preference.

        :param num_samples: Number of samples to return
        :param label: Dataset label
        :param split: Dataset split
        :param preferred_vendors_only: Whether to prefer smaller vendors
        :param seed: Random seed for reproducibility
        :return: List of sampled file IDs
        """
        candidates = self._filter_candidates(label, split, preferred_vendors_only)

        if candidates.empty:
            raise ValueError("No candidates found with given criteria")

        # Set seed for reproducibility
        import random

        if seed is not None:
            random.seed(seed)

        # Sample from available file IDs
        available_ids = candidates["file_id"].tolist()
        sampled = random.sample(available_ids, min(num_samples, len(available_ids)))

        return sampled

    def _keys_to_file_ids(
        self,
        keys: list[str],
        key_type: Literal["interaction", "session"],
        label: str | None = None,
        split: str | None = None,
    ) -> list[str]:
        """
        Convert interaction or session keys to file IDs.

        :param keys: List of keys to convert
        :param key_type: Type of keys (interaction or session)
        :param label: Dataset label
        :param split: Dataset split
        :return: List of file IDs
        """
        candidates = self._filter_candidates(label, split, preferred_vendors_only=False)
        all_file_ids = []

        for key in keys:
            # Validate key format
            if key_type == "interaction":
                InteractionKey.from_interaction_key(key)  # Will raise if invalid
            else:
                InteractionKey.from_session_key(key)  # Will raise if invalid

            # Find matching files
            matching_files = candidates[candidates["file_id"].str.startswith(key)]
            all_file_ids.extend(matching_files["file_id"].tolist())

        return all_file_ids

    def get_interaction_pairs(
        self,
        interaction_keys: str | list[str] | None = None,
        num_pairs: int = 1,
        label: str | None = None,
        split: str | None = None,
        preferred_vendors_only: bool | None = None,
    ) -> list[list[str]]:
        """
        Get interaction pairs using interaction keys or auto-sampling.

        :param interaction_keys: Interaction key(s) in format V00_S0809_I00000582
        :param num_pairs: Number of interaction pairs to return
        :param label: Dataset label
        :param split: Dataset split
        :param preferred_vendors_only: Whether to prefer smaller vendors
        :return: List of interaction pairs, each pair is a list of file IDs
        """
        if interaction_keys is not None:
            # Handle specific interaction keys
            if isinstance(interaction_keys, str):
                interaction_keys = [interaction_keys]

            all_file_ids = self._keys_to_file_ids(
                interaction_keys, "interaction", label, split
            )

            # Group by interaction and create pairs
            return self._group_files_into_pairs(all_file_ids, num_pairs)
        else:
            # Auto-sample interaction pairs
            return self._sample_interaction_pairs(
                num_pairs, label, split, preferred_vendors_only
            )

    def get_session_groups(
        self,
        session_keys: str | list[str] | None = None,
        num_sessions: int = 1,
        interactions_per_session: int = 4,
        label: str | None = None,
        split: str | None = None,
        preferred_vendors_only: bool | None = None,
    ) -> list[list[str]]:
        """
        Get session groups using session keys or auto-sampling.

        :param session_keys: Session key(s) in format V00_S0809
        :param num_sessions: Number of sessions to return
        :param interactions_per_session: Target interactions per session
        :param label: Dataset label
        :param split: Dataset split
        :param preferred_vendors_only: Whether to prefer smaller vendors
        :return: List of session groups, each group is a list of file IDs
        """
        if session_keys is not None:
            # Handle specific session keys
            if isinstance(session_keys, str):
                session_keys = [session_keys]

            groups = []
            for session_key in session_keys:
                session_file_ids = self._keys_to_file_ids(
                    [session_key], "session", label, split
                )
                if interactions_per_session > 0:
                    session_file_ids = session_file_ids[:interactions_per_session]
                if session_file_ids:
                    groups.append(session_file_ids)

            return groups[:num_sessions]
        else:
            # Auto-sample session groups
            return self._sample_session_groups(
                num_sessions,
                interactions_per_session,
                label,
                split,
                preferred_vendors_only,
            )

    def _group_files_into_pairs(
        self, file_ids: list[str], num_pairs: int
    ) -> list[list[str]]:
        """Group file IDs into interaction pairs."""
        pairs = []
        interaction_dict: dict[str, list[str]] = {}

        for file_id in file_ids:
            key_obj = InteractionKey.from_file_id(file_id)
            interaction_key = key_obj.interaction_key
            if interaction_key not in interaction_dict:
                interaction_dict[interaction_key] = []
            interaction_dict[interaction_key].append(file_id)

        for interaction_files in interaction_dict.values():
            if len(interaction_files) >= 2:
                pairs.append(interaction_files[:2])

        return pairs[:num_pairs]

    def _sample_interaction_pairs(
        self,
        num_pairs: int,
        label: str | None = None,
        split: str | None = None,
        preferred_vendors_only: bool | None = None,
    ) -> list[list[str]]:
        """Auto-sample interaction pairs from the dataset."""
        candidates = self._filter_candidates(label, split, preferred_vendors_only)

        # Group by interaction and filter for pairs
        interaction_groups = self._group_by_pattern(
            candidates, r"(V\d+_S\d+_I\d+)_P\d+"
        )
        paired_interactions = interaction_groups.filter(lambda x: len(x) >= 2)
        unique_interactions = (
            paired_interactions["file_id"]
            .str.extract(r"(V\d+_S\d+_I\d+)_P\d+")[0]
            .unique()
        )

        if len(unique_interactions) == 0:
            raise ValueError("No interaction pairs found")

        sampled_interactions = random.sample(
            list(unique_interactions), min(num_pairs, len(unique_interactions))
        )

        # Get file IDs for sampled interactions
        sampled_file_ids = []
        for interaction in sampled_interactions:
            interaction_files = candidates[
                candidates["file_id"].str.startswith(interaction)
            ]["file_id"].tolist()
            sampled_file_ids.extend(interaction_files[:2])

        return self._group_files_into_pairs(sampled_file_ids, num_pairs)

    def _sample_session_groups(
        self,
        num_sessions: int,
        interactions_per_session: int,
        label: str | None = None,
        split: str | None = None,
        preferred_vendors_only: bool | None = None,
    ) -> list[list[str]]:
        """Auto-sample session groups from the dataset."""
        candidates = self._filter_candidates(label, split, preferred_vendors_only)

        # Group by session and filter for sufficient interactions
        session_groups = self._group_by_pattern(candidates, r"(V\d+_S\d+)_I\d+_P\d+")
        rich_sessions = session_groups.filter(
            lambda x: len(x) >= interactions_per_session
        )
        unique_sessions = (
            rich_sessions["file_id"].str.extract(r"(V\d+_S\d+)_I\d+_P\d+")[0].unique()
        )

        if len(unique_sessions) == 0:
            raise ValueError("No sessions with sufficient interactions found")

        import random

        sampled_sessions = random.sample(
            list(unique_sessions), min(num_sessions, len(unique_sessions))
        )

        # Get file IDs for sampled sessions
        groups = []
        for session in sampled_sessions:
            session_files = candidates[candidates["file_id"].str.startswith(session)][
                "file_id"
            ].tolist()
            if interactions_per_session > 0:
                session_files = session_files[:interactions_per_session]
            groups.append(session_files)

        return groups

    # === File Access Methods ===

    def list_batches(
        self, label: str | None = None, split: str | None = None
    ) -> list[int]:
        """List available batches for a given label and split."""
        candidates = self._filter_candidates(label, split, preferred_vendors_only=False)
        batches = candidates["batch_idx"].unique().tolist()
        batches.sort()
        return batches

    def list_archives(
        self, label: str | None = None, split: str | None = None, batch: int = 0
    ) -> list[int]:
        """List available tar archives in a batch."""
        candidates = self._filter_candidates(label, split, preferred_vendors_only=False)
        archives = (
            candidates[candidates["batch_idx"] == batch]["archive_idx"]
            .unique()
            .tolist()
        )
        if len(archives) == 0:
            logger.error(f"No archives found for {label}/{split}/{batch}")
            return []
        archives.sort()
        return archives

    def get_tar_archive_size(
        self,
        label: str | None = None,
        split: str | None = None,
        batch: int = 0,
        archive: int = 0,
    ) -> float:
        """Get the size of a tar archive in GB."""
        label = label or self.config.label
        split = split or self.config.split

        hf_path = f"{label}/{split}/{batch:04d}/{archive:04d}.tar"
        info = self._hf_api.get_paths_info(
            repo_id=self._hf_repo_id,
            paths=hf_path,
            repo_type=self._hf_repo_type,
        )
        # convert size from bytes to GB
        size_in_gb = info[0].size / 1024 / 1024 / 1024  # type: ignore[union-attr]
        return size_in_gb

    def get_path_list_for_file_id_s3(self, file_id: str) -> list[str]:
        """Get S3 paths for all modalities of a file ID."""
        if self._cached_filelist is None:
            self._load_filelist_cache()

        file_entry = self._cached_filelist[self._cached_filelist["file_id"] == file_id]
        if file_entry.empty:
            raise ValueError(f"File ID {file_id} not found in dataset")

        label, split = file_entry.iloc[0][["label", "split"]]

        # Build S3 URLs for all modalities
        base_url = f"https://{self._bucket}/{self._prefix}"
        paths = []

        # Audio and video
        paths.extend(
            [
                f"{base_url}/{label}/{split}/audio/{file_id}.wav",
                f"{base_url}/{label}/{split}/video/{file_id}.mp4",
            ]
        )

        # Features (NPY files)
        for feature_group, features in ALL_FEATURES.items():
            if feature_group in [
                "smplh",
                "boxes_and_keypoints",
                "movement",
                "movement_v4",
            ]:
                for feature in features:
                    paths.append(
                        f"{base_url}/{label}/{split}/{feature_group}/{feature}/{file_id}.npy"
                    )

        # Metadata (JSONL files)
        for metadata_type in ALL_FEATURES["metadata"]:
            paths.append(
                f"{base_url}/{label}/{split}/metadata/{metadata_type}/{file_id}.jsonl"
            )

        # Annotations (JSON files) - optional
        for annotation_type in ALL_FEATURES["annotations"]:
            paths.append(
                f"{base_url}/{label}/{split}/annotations/{annotation_type}/{file_id}.json"
            )

        return paths

    def get_path_list_for_file_id_local(self, file_id: str) -> list[str]:
        """Get local paths for all modalities of a file ID."""
        if self._cached_filelist is None:
            self._load_filelist_cache()

        file_entry = self._cached_filelist[self._cached_filelist["file_id"] == file_id]
        if file_entry.empty:
            raise ValueError(f"File ID {file_id} not found in dataset")

        label, split, batch_idx, archive_idx = file_entry.iloc[0][
            ["label", "split", "batch_idx", "archive_idx"]
        ]

        # Return expected local paths
        if self.config.local_dir is None:
            raise ValueError("Please configure local_dir")
        base_path = os.path.join(
            self.config.local_dir,
            label,
            split,
            f"{batch_idx:04d}",
            f"{archive_idx:04d}",
        )
        path_list = glob.glob(os.path.join(base_path, f"{file_id}*"))

        if len(path_list) == 0:
            logger.warning(f"No local files found for {file_id} in {base_path}")

        return path_list

    # === Download Methods ===

    def gather_file_id_data_from_s3(
        self,
        file_id: str,
        *,
        num_workers: int | None = None,
        local_dir: str | None = None,
    ) -> None:
        """Download and organize all data for a file ID from S3."""
        if num_workers is None:
            num_workers = self.config.num_workers
        if local_dir is None:
            local_dir = self.config.local_dir
        if local_dir is None:
            raise ValueError("Please configure local_dir")

        files = self.get_path_list_for_file_id_s3(file_id)
        logger.info(f"Found {len(files)} files for {file_id}")

        file_entry = self._cached_filelist[self._cached_filelist["file_id"] == file_id]
        label, split, batch_idx, archive_idx = file_entry.iloc[0][
            ["label", "split", "batch_idx", "archive_idx"]
        ]

        target_path = os.path.join(
            local_dir, label, split, f"{batch_idx:04d}", f"{archive_idx:04d}"
        )
        os.makedirs(target_path, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with mp.Manager() as manager:
                # Create managed dictionaries for proper multiprocessing synchronization
                shared_np_data = manager.dict()
                shared_json_data = manager.dict()
                shared_json_data["id"] = file_id

                # Use a lock for thread-safe operations
                lock = manager.Lock()

                with mp.Pool(processes=num_workers) as pool:
                    # Pass shared dictionaries and lock to each worker
                    pool.starmap(
                        self._wget_download_from_s3,
                        [
                            (
                                f,
                                tmp_dir,
                                target_path,
                                shared_np_data,
                                shared_json_data,
                                lock,
                            )
                            for f in files
                        ],
                    )

                # Convert managed dicts to regular dicts for JSON serialization
                np_data = dict(shared_np_data)
                json_data = dict(shared_json_data)

        # Save JSON data
        json_file_path = os.path.join(target_path, f"{file_id}.json")
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # Save NPZ data
        if np_data:
            sorted_np_data = {k: np_data[k] for k in sorted(np_data.keys())}
            npz_file_path = os.path.join(target_path, f"{file_id}.npz")
            if os.path.exists(npz_file_path):
                os.remove(npz_file_path)
            np.savez(npz_file_path, **sorted_np_data)
            logger.info(f"Saved {len(sorted_np_data)} numpy arrays to {npz_file_path}")

        logger.info(f"Successfully processed file {file_id} to {target_path}")

    def download_batch_from_s3(
        self,
        batch: list[str],
        local_dir: str | None = None,
        num_workers: int | None = None,
    ) -> bool:
        """Download a batch of file IDs from S3."""
        if local_dir is None:
            local_dir = self.config.local_dir
        if num_workers is None:
            num_workers = self.config.num_workers

        logger.info(f"Downloading batch of {len(batch)} files")

        for file_id in batch:
            self.gather_file_id_data_from_s3(
                file_id, num_workers=num_workers, local_dir=local_dir
            )

        logger.info(f"Completed downloading batch of {len(batch)} files")
        return True

    def download_archive_from_hf(
        self,
        archive: int,
        label: str | None = None,
        split: str | None = None,
        batch: int = 0,
        local_dir: str | None = None,
        extract: bool = True,
    ) -> tuple[bool, str]:
        """Download and optionally extract a tar archive from HuggingFace."""
        if local_dir is None:
            local_dir = self.config.local_dir
        if local_dir is None:
            raise ValueError("Please configure local_dir")
        if label is None:
            label = self.config.label
        if split is None:
            split = self.config.split

        hf_path = f"{label}/{split}/{batch:04d}/{archive:04d}.tar"
        local_archive_dir = os.path.join(local_dir, label, split, f"{batch:04d}")
        os.makedirs(local_archive_dir, exist_ok=True)
        local_tar_path = os.path.join(local_archive_dir, f"{archive:04d}.tar")

        if os.path.exists(local_tar_path):
            logger.info(f"Archive already exists: {local_tar_path}")
        else:
            logger.info(f"Downloading {hf_path} to {local_tar_path}")
            try:
                hf_hub_download(
                    repo_id=self._hf_repo_id,
                    filename=hf_path,
                    local_dir=local_dir,
                    repo_type=self._hf_repo_type,
                )
            except Exception as e:
                logger.error(f"Failed to download {hf_path}: {e}")
                return False, local_tar_path

        if extract:
            extract_dir = os.path.join(local_archive_dir, f"{archive:04d}")
            if os.path.exists(extract_dir):
                logger.info(f"Archive already extracted: {extract_dir}")
                return True, extract_dir
            else:
                logger.info(f"Extracting {local_tar_path} to {extract_dir}")
                try:
                    with tarfile.open(local_tar_path, "r") as tar:
                        tar.extractall(extract_dir)
                    # remove the tar file
                    os.remove(local_tar_path)
                    return True, extract_dir
                except Exception as e:
                    logger.error(f"Failed to extract {local_tar_path}: {e}")
                    return False, local_tar_path

        return True, local_tar_path

    def download_batch_from_hf(
        self,
        label: str | None = None,
        split: str | None = None,
        batch_idx: int | list[int] | None = None,
        *,
        local_dir: str | None = None,
        num_workers: int | None = None,
        archive_list: list[int] | None = None,
    ) -> bool:
        """Download batch(es) from HuggingFace with parallel processing."""
        if local_dir is None:
            local_dir = self.config.local_dir
        if num_workers is None:
            num_workers = self.config.num_workers
        if label is None:
            label = self.config.label
        if split is None:
            split = self.config.split

        # Handle batch indices
        if batch_idx is None:
            batch_indices = self.list_batches(label, split)
        elif isinstance(batch_idx, int):
            batch_indices = [batch_idx]
        else:
            batch_indices = batch_idx

        success = True
        for batch in batch_indices:
            # Get archives for this batch
            if archive_list is None:
                archives = self.list_archives(label, split, batch)
            else:
                archives = archive_list

            logger.info(f"Downloading {len(archives)} archives for batch {batch}")

            # Download archives in parallel
            with mp.Pool(processes=num_workers) as pool:
                try:
                    download_func = partial(
                        self.download_archive_from_hf,
                        label=label,
                        split=split,
                        batch=batch,
                        local_dir=local_dir,
                        extract=True,
                    )
                    results = pool.map(download_func, archives)
                finally:
                    # Ensure proper cleanup
                    pool.close()
                    pool.join()

            # Check if all downloads succeeded
            batch_success = all(result[0] for result in results)
            if not batch_success:
                logger.error(f"Some downloads failed for batch {batch}")
                success = False

        return success

    def _wget_download_from_s3(
        self,
        url: str,
        tmp_dir: str,
        target_path: str,
        shared_np_data: dict,
        shared_json_data: dict,
        lock,
    ) -> None:
        """Download a single file from S3 and process it."""
        try:
            # Extract filename from URL
            filename = "_".join(url.split("/")[-2:])  # include also subfeature name
            tmp_file_path = os.path.join(tmp_dir, filename)

            # Download file
            if not self._dry_run:
                try:
                    wget.download(url, tmp_file_path, bar=None)
                except Exception as e:
                    if isinstance(e, HTTPError) and e.code == 403:
                        logger.info(
                            f"Skipping optional file {'/'.join(url.split('/')[-3:])}"
                        )
                    else:
                        logger.error(f"Failed to download {url}: {e}")
                    return

            # Process based on file type
            if filename.endswith(".wav"):
                # Copy audio file directly
                target_file = os.path.join(target_path, filename.replace("audio_", ""))
                if not self._dry_run:
                    shutil.move(tmp_file_path, target_file)

            elif filename.endswith(".mp4"):
                # Copy video file directly
                target_file = os.path.join(target_path, filename.replace("video_", ""))
                if not self._dry_run:
                    shutil.move(tmp_file_path, target_file)

            elif filename.endswith(".npy"):
                # Load and store numpy data
                if not self._dry_run:
                    data = np.load(tmp_file_path)
                    data = recursively_cast_to_float32(data)
                    # Extract feature name from path
                    path_parts = url.split("/")
                    feature_group = path_parts[-3]  # e.g., "smplh"
                    feature_name = path_parts[-2]  # e.g., "body_pose"
                    key = f"{feature_group}:{feature_name}"

                    with lock:
                        shared_np_data[key] = data

            elif filename.endswith(".jsonl"):
                # Process JSONL metadata
                if not self._dry_run:
                    with open(tmp_file_path, "r") as f:
                        lines = [json.loads(line) for line in f]

                    # Extract metadata type from path
                    path_parts = url.split("/")
                    metadata_type = path_parts[-2]  # e.g., "transcript" or "vad"

                    with lock:
                        shared_json_data[f"metadata:{metadata_type}"] = lines

            elif filename.endswith(".json"):
                # These annotation files are actually multi-line JSON (JSONL)
                if not self._dry_run:
                    with open(tmp_file_path, "r") as f:
                        # Load line-by-line — each line is a JSON object
                        data = [json.loads(line) for line in f if line.strip()]
                    
                    # Extract annotation type from path
                    path_parts = url.split("/")
                    annotation_type = path_parts[-2]  # e.g., "1P-IS"

                    with lock:
                        shared_json_data[f"annotations:{annotation_type}"] = data

        except Exception as e:
            if isinstance(e, HTTPError) and e.code == 403:
                logger.info(f"Skipping {url} (data not available)")
                return
            else:
                logger.error(f"Failed to download {url}: {e}")
                raise

    def check_movement_v4_availability(
        self, file_id: str, timeout: int = 10
    ) -> dict[str, bool]:
        """
        Check which movement_v4 features are available for a given file ID.

        Args:
            file_id: The file ID to check
            timeout: Request timeout in seconds

        Returns:
            Dict mapping feature names to availability (True/False)

        Note:
            Movement v4 data is being gradually rolled out. This method helps
            determine which features are currently available for a specific file.
        """
        import requests

        paths = self.get_path_list_for_file_id_s3(file_id)
        movement_v4_paths = [p for p in paths if "movement_v4" in p]

        availability = {}
        for feature in ALL_FEATURES["movement_v4"]:
            feature_path = next(
                (p for p in movement_v4_paths if f"movement_v4/{feature}/" in p), None
            )
            if feature_path is None:
                availability[feature] = False
            else:
                try:
                    response = requests.head(feature_path, timeout=timeout)
                    availability[feature] = response.status_code == 200
                except requests.RequestException:
                    availability[feature] = False

        return availability
