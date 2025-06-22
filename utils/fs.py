import logging
import os
import re
from collections import defaultdict
from typing import Final, Literal
from functools import cache

import fsspec
import pandas as pd
import s3fs
import tqdm
from huggingface_hub import HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("s3fs.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
    "annotations": [],  # json (optional)
}
# Regular expression to parse file IDs from filenames
FILE_ID_REGEX = r"V(\d+)_S(\d+)_I(\d+)_P(\d+)"


class SeamlessInteractionFS:
    """
    The s3 bucket has the following layout:
    s3://fairusersglobal/tmp/seamless/MOSAIC/
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
    """

    # s3
    _bucket: str = "fairusersglobal"
    _prefix: str = "tmp/seamless/MOSAIC"
    _s3fs: s3fs
    # hf
    _hf_api: HfApi = HfApi()
    _hf_repo_id: str = "facebook/seamless_interaction"
    _hf_repo_type: str = "dataset"
    # local
    _local_dir: str = "/checkpoint/seamless/yaoj/data/seamless_interaction/dataset/"
    _cached_filelists: dict = {}
    _cached_file_id_to_label_split: dict = {}
    _ckpt_file: str = "filelists_checkpoint.text"
    _completed_files: set = set()
    _dry_run: bool = False

    def __init__(
        self,
        *,
        bucket: str | None = None,
        prefix: str | None = None,
        local_dir: str | None = None,
        ckpt_file: str | None = None,
        hf_repo_id: str | None = None,
        hf_repo_type: Literal["dataset", "model"] = "dataset",
        dry_run: bool = False,
    ) -> None:
        if bucket:
            self._bucket = bucket
        if prefix:
            self._prefix = prefix
        if local_dir:
            self._local_dir = local_dir
        if hf_repo_id:
            self._hf_repo_id = hf_repo_id
        self._hf_repo_type = hf_repo_type
        self._s3fs = fsspec.filesystem(
            "s3",
            anon=False,
            default_block_size=10_000_000,
            config_kwargs={"max_pool_connections": 50},
        )
        self._dry_run = dry_run
        os.makedirs(self._local_dir, exist_ok=True)
        self._ckpt_file = ckpt_file or self._ckpt_file
        self._load_checkpoint()

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
        with open(self.ckpt_file, "w") as f:
            f.write("\n".join(self._completed_files))

    def _fetch_filelist(self, label: str, split: str) -> list[str]:
        """
        Fetch the file list for the specified label and split.
        """
        if (label, split) in self._cached_filelists:
            return self._cached_filelists[(label, split)]

        filelist_path = f"filelists/{label}/{split}/metadata.csv"
        df = pd.read_csv(filelist_path)
        if df.empty:
            raise ValueError(f"No files found for label '{label}' and split '{split}'.")

        filelist = df["participant_0_id"].to_list() + df["participant_1_id"].to_list()
        self._cached_filelists[(label, split)] = filelist
        for file in filelist:
            self._cached_file_id_to_label_split[file] = (label, split)
        return filelist

    def fetch_all_filelist(self) -> list[str]:
        """
        Fetch the file list for all labels and splits.
        """
        all_filelists = []
        for label in ALL_LABELS:
            for split in ALL_SPLITS:
                all_filelists.extend(self._fetch_filelist(label, split))
        return all_filelists

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
            else:
                raise ValueError(f"Invalid partition_by value: {partition_by}")

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
        _res.update({
            feature: f"{label}/{split}/{feature}/*/"
            for feature in ALL_FEATURES.keys()
        })
        return _res

    def get_path_list_for_file_id(self, file_id: str) -> list[str]:
        """
        Get the suffix for a given file ID based on its type.
        """
        label, split = self._cached_file_id_to_label_split.get(file_id, (None, None))
        if not label or not split:
            raise ValueError(f"File ID '{file_id}' not found in cached file lists.")
        if label not in ALL_LABELS or split not in ALL_SPLITS:
            raise ValueError(f"Invalid label '{label}' or split '{split}'.")

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
                    self._s3fs.glob(
                        f"{self._bucket}/{self._prefix}/{label}/{split}/annotations/{file_id}*"
                    )
                )
            elif feature == "metadata":
                # Metadata is a JSON file, we need to try glob in transcript which is optional
                path_list.extend(
                    self._s3fs.glob(
                        f"{self._bucket}/{self._prefix}/{label}/{split}/metadata/transcript/{file_id}*"
                    )
                )
                path_list.append(
                    f"{self._bucket}/{self._prefix}/{label}/{split}/metadata/vad/{file_id}.jsonl"
                )
            else:
                # For other features, we assume they are numpy files
                if feature == "movement" and not self._s3fs.exists(
                    f"{self._bucket}/{self._prefix}/{label}/{split}/movement/FAUToken/{file_id}.npy"
                ):
                    # skip movement if it doesn't exist
                    continue
                if subfeatures:
                    for subfeature in subfeatures:
                        path_list.append(
                            f"{self._bucket}/{self._prefix}/{label}/{split}/{feature}/{subfeature}/{file_id}.npy"
                        )
                else:
                    path_list.append(
                        f"{self._bucket}/{self._prefix}/{label}/{split}/{feature}/{file_id}.npy"
                    )
        return path_list

    def download_batch(
        self,
        batch: list[str],
        local_dir: str = None,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """
        Download a batch of files to the local directory.
        """
        try:
            if local_dir is None:
                local_dir = self._local_dir

            _total_files = 0
            with tqdm.tqdm(batch, desc="Downloading files") as pbar:
                for file_id in pbar:
                    paths = self.get_path_list_for_file_id(file_id)
                    for path in paths:
                        local_path = os.path.join(local_dir, path.replace(f"{self._bucket}/{self._prefix}/", ""))
                        if not overwrite and os.path.exists(local_path):
                            logger.info(f"Skipping {local_path} as it already exists.")
                            continue
                        if dry_run:
                            logger.info(
                                f"Dry run: would download {path} to {local_path}"
                            )
                            continue
                        logger.info(f"Downloading {path} to {local_path}")
                        self._s3fs.get(path, local_path)
                    _total_files += len(paths)
            logger.info(f"Downloaded the {_total_files} files")
            return True
        except Exception as e:
            logger.error(f"Error downloading batch: {e}")
            return False

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

    def upload_batch_to_hf(
        self,
        batch_idx: int,
        batch: list[str],
        label: str,
        split: str,
        modality: str,
        *,
        ignore_patterns: list[str] = None,
    ) -> None:
        """
        Upload a file or folder to the Hugging Face Hub.
        """
        suffix = f"{label}/{split}/{modality}"
        local_path = f"{self._local_dir}{suffix}"
        if modality not in ["audio", "video"]:
            allow_patterns = [f"**/{file_id}.*" for file_id in batch]
        else:
            allow_patterns = [f"{file_id}.*" for file_id in batch]

        ignore_patterns = ignore_patterns or [".git", ".gitattributes", "README.md"]
        
        self._hf_api.upload_folder(
            folder_path=local_path,
            path_in_repo=suffix,
            repo_id=self._hf_repo_id,
            repo_type=self._hf_repo_type,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
