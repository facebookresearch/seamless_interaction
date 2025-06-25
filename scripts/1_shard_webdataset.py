import os
import logging
import sys
import numpy as np
import os
import multiprocessing as mp
from glob import glob
import tqdm
from functools import partial

import webdataset as wds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("upload.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_arg_parser():
    """Set up command line argument parser"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorganize Seamless Interaction dataset files"
    )
    parser.add_argument(
        "--label",
        choices=["improvised", "naturalistic"],
        default="improvised",
        help="Dataset category to reorganize",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="dev",
        help="Dataset split to reorganize",
    )
    parser.add_argument(
        "--batch-start",
        type=int,
        default=0,
        help="Starting index for batch processing",
    )
    parser.add_argument(
        "--batch-end",
        type=int,
        default=-1,
        help="Ending index for batch processing",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=8,
        help="Maximum number of processes to run in parallel",
    )
    return parser


def handle_sigint(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    logger.info("Received interrupt signal. Exiting gracefully...")
    sys.exit(1)


def data_iter(file_id_list: list[str], input_dir: str):
    for file_id in file_id_list:
        npz_path = os.path.join(input_dir, f"{file_id}.npz")
        json_path = os.path.join(input_dir, f"{file_id}.json")
        audio_path = os.path.join(input_dir, f"{file_id}.wav")
        video_path = os.path.join(input_dir, f"{file_id}.mp4")
        if (
            not os.path.exists(npz_path)
            or not os.path.exists(json_path)
            or not os.path.exists(audio_path)
            or not os.path.exists(video_path)
        ):
            continue
        sample = {
            "__key__": file_id,
            "npz": dict(np.load(npz_path, allow_pickle=True)),
            "json": open(json_path, "r", encoding="utf-8").read(),
            "wav": open(audio_path, "rb").read(),
            "mp4": open(video_path, "rb").read(),
        }
        yield sample


def process_batch(
    i: int, batch_idx: int, input_dir: str, target_dir: str, label: str, split: str
) -> list[str]:
    input_dir = f"{input_dir}/{label}-{split}-{batch_idx:04d}"
    logger.info(f"Processing input directory: {input_dir} for batch index {batch_idx}")
    files = os.listdir(input_dir)
    file_id_list = set(
        [
            f.split("/")[-1].split(".")[0]
            for f in files
            if f.endswith((".npz", ".json", ".wav", ".mp4"))
        ]
    )
    os.makedirs(f"{target_dir}/{label}", exist_ok=True)
    os.makedirs(f"{target_dir}/{label}/{split}", exist_ok=True)
    os.makedirs(f"{target_dir}/{label}/{split}/{batch_idx:04d}", exist_ok=True)
    out_pattern = f"{target_dir}/{label}/{split}/{batch_idx:04d}/%04d.tar"
    sind = wds.ShardWriter(out_pattern, maxcount=1000, maxsize=1_000_000_000)
    for sample in data_iter(file_id_list=file_id_list, input_dir=input_dir):
        sind.write(sample)
    sind.close()
    return file_id_list


def main():
    # Set up signal handler for graceful shutdown
    import signal

    signal.signal(signal.SIGINT, handle_sigint)

    # Parse command line arguments
    args = setup_arg_parser().parse_args()

    label, split = args.label, args.split
    max_processes = args.max_processes
    data_dir_root = "/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_raw"
    target_dir = f"/checkpoint/seamless_fs2/yaoj/data/seamless_interaction_webdataset_sharded_tar_0623_{split}"
    input_dirs = glob(f"{data_dir_root}/{label}-{split}-*")
    batches = [int(f.split("-")[-1]) for f in input_dirs]
    batch_end = args.batch_end if args.batch_end != -1 else max(batches) + 1
    batch_start = args.batch_start
    batches = [
        b
        for b in batches
        if int(b) >= batch_start and (batch_end == -1 or int(b) < batch_end)
    ]

    newly_processed_files = []
    if max_processes > 1:
        # Use multiprocessing for parallel processing
        logger.info(f"Starting processing with {max_processes} processes")
        with mp.Pool(processes=max_processes) as pool:
            process_batch_partial = partial(
                process_batch,
                input_dir=data_dir_root,
                target_dir=target_dir,
                label=label,
                split=split,
            )

            for i, batch_result in enumerate(
                pool.starmap(process_batch_partial, enumerate(batches))
            ):
                newly_processed_files.extend(batch_result)

    else:
        # Serial processing
        logger.info("Starting processing in single process mode")
        for batch_idx in tqdm(batches, desc="Processing batches"):
            batch_result = process_batch(
                batch_idx=batch_idx,
                input_dir=data_dir_root,
                target_dir=target_dir,
                label=label,
                split=split,
                batch_start=batch_start,
            )
            newly_processed_files.extend(batch_result)

    logger.info(
        f"Processing complete! Successfully processed {len(newly_processed_files)} files"
    )


if __name__ == "__main__":
    main()
