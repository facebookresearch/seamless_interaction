import os
import argparse
import logging
import sys
import time
from utils.fs import SeamlessInteractionFS
import multiprocessing as mp
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("upload.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Upload Seamless Interaction dataset to HuggingFace"
    )
    parser.add_argument(
        "--label",
        choices=["improvised", "naturalistic"],
        default="improvised",
        help="Dataset category to upload",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="dev",
        help="Dataset split to upload",
    )
    parser.add_argument(
        "--modality",
        choices=[
            "audio",
            "video",
            "metadata",
            "movement",
            "boxes_and_keypoints",
            "smplh",
            "annotations",
        ],
        default="metadata",
        help="Data modality to upload",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of files to upload in each batch",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=24,
        help="Maximum number of upload processes to run in parallel",
    )
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Continue from a previous upload (skip files already uploaded)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Interval (in minutes) between saving upload checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually uploading",
    )
    return parser


def handle_sigint(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    logger.info("Received interrupt signal. Exiting gracefully...")
    sys.exit(1)


def process_batch(
    batch_idx: int,
    batch: list[str],
    fs: SeamlessInteractionFS,
    label: str,
    split: str,
    modality: str,
) -> list[str]:
    # 100 files
    logger.info(f"Processing batch {batch_idx + 1}: {len(batch)} files")
    if fs._dry_run:
        logger.info(f"Dry run: Skipping upload for batch {batch_idx + 1}")
        return []
    try:
        fs.upload_batch_to_hf(
            batch_idx=batch_idx,
            batch=batch,
            label=label,
            split=split,
            modality=modality,
        )
    except Exception as e:
        logger.error(f"Error processing batch {batch_idx + 1}: {e}")
        return []
    return batch


def save_checkpoint(
    completed_files: list[str], checkpoint_file: str = "upload_checkpoint.txt"
):
    """Save a checkpoint of completed file uploads"""
    with open(checkpoint_file, "w") as f:
        for file_path in completed_files:
            f.write(f"{file_path}\n")


def load_checkpoint(checkpoint_file: str = "upload_checkpoint.txt") -> list[str]:
    """Load checkpoint of completed file uploads"""
    completed_files = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_files = [line.strip() for line in f]
    return completed_files


def main() -> None:
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Set up signal handler for graceful shutdown
    import signal

    signal.signal(signal.SIGINT, handle_sigint)

    # Log the arguments
    logger.info(f"Arguments: {args}")

    # get arguments
    label = args.label
    split = args.split
    modality = args.modality
    batch_size = args.batch_size
    dry_run = args.dry_run
    max_processes = args.max_processes
    ckpt_file = f"ckpt_{label}_{split}_{modality}.txt"
    completed_files = load_checkpoint(ckpt_file) if args.continuation else []
    logger.info(f"Loaded {len(completed_files)} completed files from checkpoint")

    fs = SeamlessInteractionFS(dry_run=dry_run)
    all_files = fs._fetch_filelist(label=label, split=split)
    partitions = fs.partition_filelist(
        all_files, partition_by="vendor-session-interaction"
    )
    batches = fs.create_batches(partitions, batch_size=batch_size)
    print(
        f"[{label}/{split}/{modality}] Total batches created: {len(batches)} for {len(all_files)} files"
    )

    # Setup for checkpointing
    last_checkpoint_time = time.time()
    newly_completed_files = []

    if max_processes > 1:
        # Use multiprocessing for parallel uploads
        logger.info(f"Starting uploads with {max_processes} processes")
        with mp.Pool(processes=max_processes) as pool:
            process_batch_partial = partial(
                process_batch,
                fs=fs,
                label=label,
                split=split,
                modality=modality,
            )

            for i, batch_result in enumerate(
                pool.starmap(process_batch_partial, enumerate(batches))
            ):
                newly_completed_files.extend(batch_result)

                # Checkpoint periodically
                current_time = time.time()
                if (
                    current_time - last_checkpoint_time
                ) / 60 >= args.checkpoint_interval:
                    save_checkpoint(completed_files + newly_completed_files, ckpt_file)
                    last_checkpoint_time = current_time
                    logger.info(
                        f"Saved checkpoint with {len(completed_files) + len(newly_completed_files)} files"
                    )
    else:
        # Serial processing
        logger.info("Starting uploads in single process mode")
        for batch_idx, batch in enumerate(batches):
            batch_result = process_batch(
                batch_idx=batch_idx,
                batch=batch,
                fs=fs,
                label=label,
                split=split,
                modality=modality,
            )
            newly_completed_files.extend(batch_result)

            # Checkpoint periodically
            current_time = time.time()
            if (current_time - last_checkpoint_time) / 60 >= args.checkpoint_interval:
                save_checkpoint(completed_files + newly_completed_files)
                last_checkpoint_time = current_time
                logger.info(
                    f"Saved checkpoint with {len(completed_files) + len(newly_completed_files)} files"
                )

    # Save final checkpoint
    save_checkpoint(completed_files + newly_completed_files)
    logger.info(
        f"Upload complete! Successfully uploaded {len(newly_completed_files)} files"
    )
    logger.info(
        f"Total files uploaded: {len(completed_files) + len(newly_completed_files)}"
    )


if __name__ == "__main__":
    main()
