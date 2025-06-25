# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from functools import cache, partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import wget
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from tqdm import tqdm

from seamless_interaction.constants import (
    ALL_LABELS,
    ALL_SPLITS,
    ALL_FEATURES,
    FILE_ID_REGEX,
)
from seamless_interaction.utils import setup_logging, recursively_cast_to_float32

logger = setup_logging(__name__)


class SeamlessInteractionFS:
    """
    The s3 bucket has the following layout (note that `list_objects_v2` is not supported for this bucket for security reasons):
    s3://dl.fbaipublicfiles.com/seamless_interaction/
    ├── improvised
    │   ├── dev
    │   │   ├── audio
    │   │   ├── annotations
    │   │   ├── metadata
    |   |   |   ├── trascription
    │   │   |   └── vad
    │   │   ├── smplh  # visual (npy)
    │   │   │   ├── body_pose
    │   │   │   ├── global_orient
    │   │   │   ├── is_valid
    │   │   │   ├── left_hand_pose
    │   │   │   ├── right_hand_pose
    │   │   │   └── translation
    │   │   ├── movement  # imitator encoder (npy)
    │   │   │   ├── EmotionArousalToken
    │   │   │   ├── emotion_valence
    │   │   │   ├── EmotionValenceToken
    │   │   │   ├── expression
    │   │   │   ├── FAUToken
    │   │   │   ├── frame_latent
    │   │   │   ├── FAUValue
    │   │   │   ├── gaze_encodings
    │   │   │   ├── alignment_head_rotation
    │   │   │   ├── head_encodings
    │   │   │   ├── alignment_translation
    │   │   │   ├── hypernet_features
    │   │   │   ├── emotion_arousal
    │   │   │   ├── is_valid
    │   │   │   └── emotion_scores
    │   │   ├── video
    │   │   └── boxes_and_keypoints
    │   │       ├── box
    │   │       ├── is_valid_box
    │   │       └── keypoints
    │   ├── train
    │   └── test
    └── naturalistic
        ├── dev
        ├── train
        └── test

    The HuggingFace dataset has the following layout:
    datasets/facebook/seamless-interaction/
    ├── improvised/
    │   ├── train/
    │   │   ├── 0000/
    │   │   │   ├── 0000.tar
    │   │   │   ├── 0001.tar
    │   │   │   └── ...
    │   │   ├── 0001/
    │   │   └── ...
    │   ├── dev/
    │   └── test/
    └── naturalistic/
        ├── train/
        ├── dev/
        └── test/

    Each tar file contains:
    - .mp4 files (video)
    - .wav files (audio)
    - .json files (annotations, metadata)
    - .npz files (movement, smplh, keypoints data)
    """

    # s3
    _bucket: str = "dl.fbaipublicfiles.com"
    _prefix: str = "seamless_interaction"
    # HuggingFace
    _hf_api: HfApi = HfApi()
    _hf_fs: HfFileSystem = None
    _hf_repo_id: str = "facebook/seamless-interaction"
    _hf_repo_type: str = "dataset"
    # local
    _local_dir: str = Path.home() / "datasets/seamless_interaction"  # YMMV
    _ckpt_file: str = "filelists_checkpoint.text"
    _cached_filelist: pd.DataFrame = None
    _cached_file_id_to_label_split: dict = {}
    _cached_batches: dict = {}  # Cache for batch listings
    _completed_files: set = set()
    _dry_run: bool = False
    _num_workers: int = min(10, max(1, os.cpu_count() - 2))

    def __init__(
        self,
        *,
        local_dir: str | None = None,
        ckpt_file: str | None = None,
        hf_repo_id: str | None = None,
        hf_repo_type: Literal["dataset", "model"] = "dataset",
        dry_run: bool = False,
        num_workers: int | None = None,
    ) -> None:
        if local_dir:
            self._local_dir = local_dir
        if hf_repo_id:
            self._hf_repo_id = hf_repo_id
        self._hf_repo_type = hf_repo_type
        self._hf_fs = HfFileSystem()
        self._dry_run = dry_run
        try:
            os.makedirs(self._local_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create local directory {self._local_dir}: {e}")
            raise e
        self._ckpt_file = ckpt_file or self._ckpt_file
        self._load_checkpoint()
        # to populate the cached file list
        _ = self._fetch_filelist("improvised", "train")
        if num_workers:
            self._num_workers = num_workers

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value: int) -> None:
        self._num_workers = value

    def _load_checkpoint(self) -> None:
        """
        Load the checkpoint file to restore cached file lists and completed files.
        """
        if os.path.exists(self._ckpt_file):
            with open(self._ckpt_file, "r") as f:
                self._completed_files = set(f.read().splitlines())

    def _save_checkpoint(self) -> None:
        """
        Save the current state of cached file lists and completed files to a checkpoint file.
        """
        with open(self._ckpt_file, "w") as f:
            f.write("\n".join(self._completed_files))

    def _fetch_filelist(self, label: str, split: str) -> list[str]:
        """
        Fetch the file list for the specified label and split.
        """
        if self._cached_filelist is None:
            repo_root_dir = (
                os.getcwd().split("seamless_interaction")[0] + "seamless_interaction"
            )
            filelist_path = f"{repo_root_dir}/assets/metadata.csv"
            df = pd.read_csv(filelist_path)
            if df.empty:
                raise ValueError(
                    f"No files found for label '{label}' and split '{split}'."
                )

            self._cached_filelist = df
        return self._cached_filelist.loc[
            (self._cached_filelist["label"] == label)
            & (self._cached_filelist["split"] == split),
            "file_id",
        ].to_list()

    def fetch_all_filelist(self) -> list[str]:
        """
        Fetch the file list for all labels and splits.
        """
        all_filelists = []
        for label in ALL_LABELS:
            for split in ALL_SPLITS:
                try:
                    all_filelists.extend(self._fetch_filelist(label, split))
                except Exception as e:
                    logger.warning(f"Could not fetch filelist for {label}/{split}: {e}")
        return all_filelists

    def list_batches(self, label: str, split: str) -> list[str]:
        """
        List available batches for a given label and split.
        """
        if self._cached_filelist is None:
            _ = self._fetch_filelist(label, split)

        batches = self._cached_filelist.loc[
            (self._cached_filelist["label"] == label)
            & (self._cached_filelist["split"] == split),
            "batch_idx",
        ].to_list()
        batches.sort()
        return batches

    def list_archives(self, label: str, split: str, batch: int) -> list[int]:
        """
        List available tar archives in a batch.

        :param label: The label (improvised or naturalistic)
        :param split: The split (train, dev, test)
        :param batch: The batch index as integer
        :return: List of unique archive indices as integers
        """
        if self._cached_filelist is None:
            _ = self._fetch_filelist(label, split)

        archives = (
            self._cached_filelist.loc[
                (self._cached_filelist["label"] == label)
                & (self._cached_filelist["split"] == split)
                & (self._cached_filelist["batch_idx"] == batch),
                "archive_idx",
            ]
            .unique()
            .tolist()
        )  # Use unique() to avoid duplicates
        if len(archives) == 0:
            logger.error(f"No archives found for {label}/{split}/{batch}")
            return []
        archives.sort()
        return archives

    def get_tar_archive_size(
        self, label: str, split: str, batch: int, archive: int
    ) -> float:
        """
        Get the size of a tar archive.
        """
        hf_path = f"{label}/{split}/{batch:04d}/{archive:04d}.tar"
        info = self._hf_api.get_paths_info(
            repo_id=self._hf_repo_id,
            paths=hf_path,
            repo_type=self._hf_repo_type,
        )
        size_in_gb = info[0].size / 1024 / 1024 / 1024
        return size_in_gb

    def gather_file_id_data_from_s3(
        self,
        file_id: str,
        *,
        num_workers: int | None = None,
        local_dir: str | None = None,
    ) -> None:
        """
        Given a file ID, gather all the audio, video, annotations, metadata, and npy data from s3.

        Reorganize and save them into:
        <label>/<split>/<batch_idx>/<archive_idx>/<file_id>[.wav, .mp4, .json, .npz].

        :param file_id: The file ID to gather data for
        :param num_workers: Number of parallel workers to use
        :param local_dir: Local directory to download to
        """
        if num_workers is None:
            num_workers = self._num_workers
        if local_dir is None:
            local_dir = self._local_dir
        files = self.get_path_list_for_file_id_s3(file_id)
        logger.info(f"Found {len(files)} files for {file_id}")

        label, split, batch_idx, archive_idx = self._cached_filelist.loc[
            self._cached_filelist["file_id"] == file_id,
            ["label", "split", "batch_idx", "archive_idx"],
        ].values[0]
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
            shutil.rmtree(json_file_path)
        with open(json_file_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # Reorder the np_data to ensure consistent order and save as .npz
        if np_data:
            sorted_np_data = {k: np_data[k] for k in sorted(np_data.keys())}
            print(sorted_np_data)
            npz_file_path = os.path.join(target_path, f"{file_id}.npz")
            np.savez(npz_file_path, **sorted_np_data)
            logger.info(f"Saved {len(sorted_np_data)} numpy arrays to {npz_file_path}")

        logger.info(f"Successfully processed file {file_id} to {target_path}")

    def download_archive_from_hf(
        self,
        idx: int,
        archive: int,
        label: str,
        split: str,
        batch: int,
        local_dir: str | None = None,
        extract: bool = True,
    ) -> tuple[bool, str]:
        """
        Download and optionally extract a tar archive.

        :param label: The label (improvised or naturalistic)
        :param split: The split (train, dev, test)
        :param batch: The batch index
        :param archive: The archive index
        :param local_dir: Local directory to download to
        :param extract: Whether to extract the tar file
        :return: Tuple of (success, extracted_dir_or_tar_path)
        """
        if local_dir is None:
            local_dir = self._local_dir

        # Create target directory
        target_dir = os.path.join(local_dir, label, split, f"{batch:04d}")
        os.makedirs(target_dir, exist_ok=True)

        # Check if extraction directory already exists
        extract_dir = os.path.join(target_dir, f"{archive:04d}")
        if extract and os.path.exists(extract_dir) and os.listdir(extract_dir):
            logger.info(f"Archive {archive:04d} already extracted at {extract_dir}")
            return True, extract_dir

        # Check if tar file already exists
        local_tar_path = os.path.join(target_dir, f"{archive:04d}.tar")
        if not extract and os.path.exists(local_tar_path):
            logger.info(f"Archive {archive:04d} already downloaded at {local_tar_path}")
            return True, local_tar_path

        try:
            # Download the tar file
            hf_path = f"{label}/{split}/{batch:04d}/{archive:04d}.tar"
            logger.info(f"Downloading {hf_path} to {local_tar_path}")

            # Check the file size using hf api
            size_in_gb = self.get_tar_archive_size(label, split, batch, archive)
            logger.info(f"File size: {size_in_gb:.2f} GB")

            hf_hub_download(
                repo_id=self._hf_repo_id,
                repo_type=self._hf_repo_type,
                filename=hf_path,
                local_dir=local_dir,
            )

            if not os.path.exists(local_tar_path):
                logger.error(f"Failed to download {hf_path} to {local_tar_path}")
                return False, ""

        except Exception as e:
            logger.error(
                f"Error downloading archive {label}/{split}/{batch:04d}/{archive:04d}: {e}"
            )
            return False, ""

        if extract:
            # Extract the tar file to target dir
            logger.info(f"Extracting {local_tar_path} to {extract_dir}")
            try:
                os.makedirs(extract_dir, exist_ok=True)
                with tarfile.open(local_tar_path, "r") as tar:
                    tar.extractall(extract_dir)

                # Remove the tar file after successful extraction
                os.remove(local_tar_path)
                logger.info(f"Successfully extracted and removed {local_tar_path}")

            except Exception as e:
                logger.error(f"Error extracting {local_tar_path} to {extract_dir}: {e}")
                return False, ""

            return True, extract_dir
        else:
            return True, local_tar_path

    def get_path_list_for_file_id_s3(self, file_id: str) -> list[str]:
        """
        Get the path list for a given file ID from s3 for wget download.
        """
        label, split = self._cached_filelist.loc[
            self._cached_filelist["file_id"] == file_id,
            ["label", "split"],
        ].values[0]

        path_list = [
            # audio
            f"{self._bucket}/{self._prefix}/{label}/{split}/audio/{file_id}.wav",
            # video
            f"{self._bucket}/{self._prefix}/{label}/{split}/video/{file_id}.mp4",
        ]
        for feature, subfeatures in ALL_FEATURES.items():
            if feature == "annotations":
                # Annotations are JSON files, we need to try glob it
                path_list.extend(
                    [
                        f"{self._bucket}/{self._prefix}/{label}/{split}/annotations/{subfeature}/{file_id}.json"
                        for subfeature in subfeatures
                    ]
                )
            elif feature == "metadata":
                # Metadata is a JSON file, we need to try glob in transcript which is optional
                path_list.extend(
                    [
                        f"{self._bucket}/{self._prefix}/{label}/{split}/metadata/{subfeature}/{file_id}.jsonl"
                        for subfeature in subfeatures
                    ]
                )
            else:
                # For other features, we assume they are numpy files
                path_list.extend(
                    [
                        f"{self._bucket}/{self._prefix}/{label}/{split}/{feature}/{subfeature}/{file_id}.npy"
                        for subfeature in subfeatures
                    ]
                )

        return path_list

    def get_path_list_for_file_id_local(self, file_id: str) -> list[str]:
        """
        Get the file paths for a given file ID.
        """
        label, split, batch_idx = self._cached_filelist.loc[
            self._cached_filelist["file_id"] == file_id, ["label", "split", "batch_idx"]
        ].values[0]
        if label not in ALL_LABELS or split not in ALL_SPLITS:
            raise ValueError(f"Invalid label '{label}' or split '{split}'.")

        # Return expected local paths - we'll need to search through extracted archives
        base_path = os.path.join(self._local_dir, label, split, f"{batch_idx:04d}")
        path_list = glob.glob(os.path.join(base_path, f"{file_id}*"))
        if len(path_list) == 0:
            logger.error(f"No local files found for {file_id}")
        elif len(path_list) < 4:
            logger.warning(
                f"Expected 4 files (.wav, .mp4, .json, .npz) for {file_id}, found {len(path_list)} ({[p.split('.')[-1] for p in path_list]})"
            )

        return path_list

    def download_batch_from_hf(
        self,
        label: str,
        split: str,
        batch_idx: int | list[int] | None = None,
        *,
        local_dir: str | None = None,
        num_workers: int | None = None,
        archive_list: list[int] | None = None,
    ) -> bool:
        """
        Download a batch of tar archives from HuggingFace and extract them.

        Each worker downloads a different archive in parallel to avoid duplication.

        :param label: The label (improvised or naturalistic)
        :param split: The split (train, dev, test)
        :param batch_idx: The batch index or list of batch indices, or None to download all batches
        :param local_dir: Local directory to download to
        :param num_workers: Number of parallel workers
        :return: True if all downloads succeeded, False otherwise
        """
        if num_workers is None:
            num_workers = self._num_workers
        if local_dir is None:
            local_dir = self._local_dir

        archives = self.list_archives(label, split, batch_idx)
        if archive_list is not None:
            archives = [a for a in archives if a in archive_list]
        if not archives:
            logger.warning(f"No archives found for {label}/{split}/{batch_idx}")
            return False

        logger.info(
            f"Starting download of {len(archives)} archives from batch {batch_idx} using {num_workers} workers"
        )

        if batch_idx is None:
            batch_idx = self.list_batches(label, split)

        if isinstance(batch_idx, list):
            for batch in batch_idx:
                self.download_batch_from_hf(
                    label=label,
                    split=split,
                    batch_idx=batch,
                    local_dir=local_dir,
                    num_workers=num_workers,
                    archive_list=archive_list,
                )
            return True

        success_count = 0
        if num_workers > 1:
            # Use multiprocessing for parallel processing
            logger.info(f"Starting processing with {num_workers} processes")
            with mp.Pool(processes=num_workers) as pool:
                process_batch_partial = partial(
                    self.download_archive_from_hf,
                    label=label,
                    split=split,
                    batch=batch_idx,
                    local_dir=local_dir,
                    extract=True,
                )

                pool.starmap(process_batch_partial, enumerate(archives))

        else:
            # Serial processing
            logger.info("Starting processing in single process mode")
            for archive in tqdm(archives, desc="Processing batches"):
                self.download_archive_from_hf(
                    label=label,
                    split=split,
                    batch=batch_idx,
                    archive=archive,
                    local_dir=local_dir,
                    extract=True,
                )

        logger.info(
            f"Completed batch {batch_idx}: {success_count}/{len(archives)} archives successful"
        )
        return success_count == len(archives)

    def _wget_download_from_s3(
        self,
        url: str,
        tmp_dir: str,
        target_path: str,
        shared_np_data: dict,
        shared_json_data: dict,
        lock,
    ) -> None:
        """
        Download a file from a URL to a local path and store data in shared dictionaries.

        :param url: The URL to download from
        :param tmp_dir: Temporary directory for intermediate files
        :param target_path: Target directory for final files
        :param shared_np_data: Shared dictionary for numpy data
        :param shared_json_data: Shared dictionary for JSON data
        :param lock: Multiprocessing lock for thread-safe operations
        """
        local_path = os.path.join(tmp_dir, "_".join(url.split("/")[-2:]))
        file_id = shared_json_data["id"]

        try:
            if url.endswith(".npy"):
                try:
                    logger.info(f"Downloading numpy file: {url}")
                    wget.download(f"https://{url}", out=local_path)
                    feature = ":".join(url.split("/")[-3:-1])
                    loaded = np.load(local_path, allow_pickle=True)

                    # Use lock for thread-safe dictionary access
                    with lock:
                        shared_np_data[feature] = recursively_cast_to_float32(loaded)
                except Exception:
                    logger.info(f"{url} does not exist, skipping")

            elif url.endswith(".jsonl"):
                try:
                    logger.info(f"Downloading JSONL file: {url}")
                    wget.download(f"https://{url}", out=local_path)
                    feature = ":".join(url.split("/")[-3:-1])

                    with open(local_path, "r") as f:
                        data = [json.loads(line.strip()) for line in f if line.strip()]

                    with lock:
                        shared_json_data[feature] = data
                except Exception:
                    logger.info(f"{url} does not exist, skipping")

            elif url.endswith(".json"):
                logger.info(f"Downloading JSON file: {url}")
                try:
                    wget.download(f"https://{url}", out=local_path)
                    feature = ":".join(url.split("/")[-3:-1])

                    with open(local_path, "r") as f:
                        data = json.load(f)

                    with lock:
                        shared_json_data[feature] = data
                except Exception:
                    logger.info(f"{url} does not exist, skipping")

            elif url.endswith(".wav"):
                final_path = os.path.join(target_path, f"{file_id}.wav")
                if not os.path.exists(final_path):
                    logger.info(f"Downloading audio file: {url}")
                    wget.download(f"https://{url}", out=final_path)
                else:
                    logger.info(f"Audio file already exists: {final_path}")

            elif url.endswith(".mp4"):
                final_path = os.path.join(target_path, f"{file_id}.mp4")
                if not os.path.exists(final_path):
                    logger.info(f"Downloading video file: {url}")
                    wget.download(f"https://{url}", out=final_path)
                else:
                    logger.info(f"Video file already exists: {final_path}")

            else:
                logger.warning(f"Unknown file type for URL: {url}")

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            raise

    def download_batch_from_s3(
        self,
        batch: list[str],
        local_dir: str | None = None,
        num_workers: int | None = None,
    ) -> bool:
        """
        Download files for a batch of file IDs from s3.
        Using multiprocessing to download files in parallel.
        Track the progress of the download w/ tqdm.
        """
        if num_workers is None:
            num_workers = self._num_workers

        for file in tqdm(batch, desc="Downloading files"):
            self.gather_file_id_data_from_s3(
                file, num_workers=num_workers, local_dir=local_dir
            )
        return True

    def partition_filelist(
        self,
        file_list: list[str],
        partition_by: Literal[
            "vendor-session", "vendor-session-interaction"
        ] = "vendor-session",
    ) -> dict[str, list[str]]:
        """
        Partition the file list by vendor and session.
        """
        partitions = defaultdict(list)
        for file in file_list:
            match = re.match(FILE_ID_REGEX, file)
            if not match:
                logger.info(f"Skipping file with invalid format: {file}")
                continue
            vendor_id, session_id, interaction_id, _ = match.groups()
            # Use vendor, session and interaction as the partition key
            if partition_by == "vendor-session":
                key = f"V{vendor_id}_S{session_id}"
            elif partition_by == "vendor-session-interaction":
                key = f"V{vendor_id}_S{session_id}_I{interaction_id}"

            partitions[key].append(file)

        return partitions

    def create_batches(
        self, partitions: dict[str, list[str]], batch_size: int = 32
    ) -> list[list[str]]:
        """
        Create balanced batches of files for uploading.

        Args:
            partitions: Dictionary mapping partition keys to lists of files
            batch_size: Target number of files per batch

        Returns:
            List of batches, where each batch is a list of files
        """
        batches = []
        current_batch = []

        # Sort partitions by size for better balancing
        sorted_partitions = sorted(
            partitions.items(), key=lambda x: len(x[1]), reverse=True
        )

        for _, files in sorted_partitions:
            # If adding this partition would exceed batch size, start a new batch
            if len(current_batch) + len(files) > batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []

            # If the partition itself exceeds batch size, split it
            if len(files) > batch_size:
                for i in range(0, len(files), batch_size):
                    batches.append(files[i : i + batch_size])
            else:
                current_batch.extend(files)

        # Add the final batch if there's anything left
        if current_batch:
            batches.append(current_batch)

        return batches

    @cache
    def _get_glob_prefix(self, label: str, split: str) -> dict[str, str]:
        """
        Get the list of prefix for a given label and split.
        """
        if label not in ALL_LABELS or split not in ALL_SPLITS:
            raise ValueError(f"Invalid label '{label}' or split '{split}'.")

        _res = {
            "audio": f"{label}/{split}/audio/",
            "video": f"{label}/{split}/video/",
        }
        _res.update(
            {
                feature: f"{label}/{split}/{feature}/*/"
                for feature in ALL_FEATURES.keys()
            }
        )
        return _res

    def process_batch(
        self,
        batch_idx: int,
        batch: list[str],
        *,
        local_dir: str = None,
        continuation: bool = True,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """
        Process a batch of files, downloading them and updating the checkpoint.
        """
        if continuation:
            # Load the checkpoint to get completed files
            self._load_checkpoint()
            batch = [file for file in batch if file not in self._completed_files]

        if not batch:
            logger.info(f"Batch {batch_idx} has no new files to upload, skipping")
            return

        # Download the batch
        success = self.download_batch(
            batch, local_dir or self._local_dir, overwrite=overwrite, dry_run=dry_run
        )
        if not success:
            logger.error(f"Failed to download batch {batch_idx + 1}")
            return
        else:
            # Update completed files
            self._completed_files.update(batch)
            self._save_checkpoint()
            return
