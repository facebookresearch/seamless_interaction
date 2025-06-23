from utils.fs import SeamlessInteractionFS
import shutil
import os
import logging
import sys
import numpy as np
import json
from tqdm import tqdm

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
        "--batch-size",
        type=int,
        default=500,
        help="Number of files to process in each batch",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=24,
        help="Maximum number of processes to run in parallel",
    )
    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Continue from a previous run (skip files already processed)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Interval (in minutes) between saving checkpoints",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_raw/",
        help="Target directory for reorganized files",
    )
    parser.add_argument(
        "--batch-start",
        type=int,
        default=0,
        help="Starting index for batch processing (useful for resuming from a specific batch)",
    )
    parser.add_argument(
        "--batch-end",
        type=int,
        default=-1,
        help="Ending index for batch processing (use -1 for all batches)",
    )
    
    return parser


def handle_sigint(signum, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    logger.info("Received interrupt signal. Exiting gracefully...")
    sys.exit(1)


def recursively_cast_to_float32(data):
    if isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            return data.astype(np.float32)
        else:
            return data
    elif isinstance(data, list):
        return [recursively_cast_to_float32(item) for item in data]
    elif isinstance(data, dict):
        return {k: recursively_cast_to_float32(v) for k, v in data.items()}
    else:
        return data  # Keep other types unchanged


def gather_file(fs: SeamlessInteractionFS, file_id: str, batch_idx: int, target_dir: str, local: bool = True) -> None:
    files = fs.get_path_list_for_file_id(file_id, local=local)
    label, split = fs._cached_file_id_to_label_split[file_id]
    target_path = os.path.join(target_dir, f"{label}-{split}-{batch_idx:04d}")
    os.makedirs(target_path, exist_ok=True)
    np_data = {}
    json_data = {"id": file_id}
    for f in files:
        if not local:
            local_path = os.path.join(fs._local_dir, f.split(fs._prefix)[-1].strip("/"))
        else:
            local_path = f
        
        if f.endswith(".npy"):
            feature = ":".join(f.split("/")[-3:-1])
            loaded = np.load(local_path, allow_pickle=True)
            np_data[feature] = recursively_cast_to_float32(loaded)

        elif f.endswith(".jsonl"):
            feature = ":".join(f.split("/")[-3:-1])
            with open(local_path, "r") as f_json:
                json_data[feature] = [json.loads(line) for line in f_json]
        
        elif f.endswith(".json"):
            feature = ":".join(f.split("/")[-3:-1])
            with open(local_path, "r") as f_json:
                annotation = []
                for line in f_json:
                    annotation.append(json.loads(line))
            json_data[feature] = annotation
        elif f.endswith(".wav"):
            if os.path.exists(local_path):
                shutil.move(local_path, f"{target_path}/{file_id}.wav")
        elif f.endswith(".mp4"):
            if os.path.exists(local_path):
                shutil.move(local_path, f"{target_path}/{file_id}.mp4")
    json.dump(json_data, open(f"{target_path}/{file_id}.json", "w"), indent=4)
    # reorder the np_data to ensure consistent order
    np_data = {k: np_data[k] for k in sorted(np_data.keys())}
    # Save the numpy data as .npz file
    np.savez(f"{target_path}/{file_id}.npz", **np_data)


def process_batch(batch_idx: int, batch: list[str], fs: SeamlessInteractionFS, target_dir: str, batch_start: int = 0) -> list[str]:
    """Process a batch of files"""
    logger.info(f"Processing batch {batch_idx + 1} with {len(batch)} files")
    processed_files = []
    
    for file_id in tqdm(batch, desc=f"Processing batch {batch_idx + 1}", leave=False, position=batch_idx - batch_start):
        try:
            gather_file(fs, file_id, batch_idx=batch_idx, target_dir=target_dir)
            processed_files.append(file_id)
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}")
    
    return processed_files


def save_checkpoint(completed_files: list[str], checkpoint_file: str = "reorganize_checkpoint.txt"):
    """Save a checkpoint of completed file processing"""
    with open(checkpoint_file, "w") as f:
        for file_id in completed_files:
            f.write(f"{file_id}\n")


def load_checkpoint(checkpoint_file: str = "reorganize_checkpoint.txt") -> list[str]:
    """Load checkpoint of completed file processing"""
    completed_files = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            completed_files = [line.strip() for line in f]
    return completed_files


def main():
    # Set up signal handler for graceful shutdown
    import signal
    import time
    import multiprocessing as mp
    from functools import partial

    signal.signal(signal.SIGINT, handle_sigint)
    
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Log the arguments
    logger.info(f"Arguments: {args}")
    
    # Get arguments
    label = args.label
    split = args.split
    batch_size = args.batch_size
    max_processes = args.max_processes
    target_dir = args.target_dir
    batch_start = args.batch_start
    batch_end = args.batch_end if args.batch_end != -1 else None
    
    # Setup checkpoint handling
    ckpt_file = f"reorganize_ckpt_{label}_{split}.txt"
    completed_files = load_checkpoint(ckpt_file) if args.continuation else []
    logger.info(f"Loaded {len(completed_files)} completed files from checkpoint")
    
    fs = SeamlessInteractionFS(skip_s3=True)
    all_files = fs._fetch_filelist(label=label, split=split)
    
    # Filter out already processed files if continuing from checkpoint
    if args.continuation and completed_files:
        all_files = [f for f in all_files if f not in completed_files]
        logger.info(f"Filtered to {len(all_files)} remaining files to process")
    
    partitions = fs.partition_filelist(all_files, partition_by="vendor-session-interaction")
    batches = fs.create_batches(partitions, batch_size=batch_size)
    
    logger.info(f"[{label}/{split}] Total batches created: {len(batches)} for {len(all_files)} files")
    
    # Setup for checkpointing
    last_checkpoint_time = time.time()
    newly_processed_files = []
    
    if max_processes > 1:
        # Use multiprocessing for parallel processing
        logger.info(f"Starting processing with {max_processes} processes")
        with mp.Pool(processes=max_processes) as pool:
            process_batch_partial = partial(
                process_batch,
                fs=fs,
                target_dir=target_dir,
                batch_start=batch_start,
            )
            
            for i, batch_result in enumerate(
                pool.starmap(process_batch_partial, enumerate(batches))
            ):
                newly_processed_files.extend(batch_result)
                
                # Checkpoint periodically
                current_time = time.time()
                if (current_time - last_checkpoint_time) / 60 >= args.checkpoint_interval:
                    save_checkpoint(completed_files + newly_processed_files, ckpt_file)
                    last_checkpoint_time = current_time
                    logger.info(f"Saved checkpoint with {len(completed_files) + len(newly_processed_files)} files")
    else:
        # Serial processing
        logger.info("Starting processing in single process mode")
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            if batch_idx < batch_start:
                continue
            if batch_end is not None and batch_idx >= batch_end:
                break
            batch_result = process_batch(
                batch_idx=batch_idx,
                batch=batch,
                fs=fs,
                target_dir=target_dir,
            )
            newly_processed_files.extend(batch_result)
            
            # Checkpoint periodically
            current_time = time.time()
            if (current_time - last_checkpoint_time) / 60 >= args.checkpoint_interval:
                save_checkpoint(completed_files + newly_processed_files, ckpt_file)
                last_checkpoint_time = current_time
                logger.info(f"Saved checkpoint with {len(completed_files) + len(newly_processed_files)} files")
    
    # Save final checkpoint
    save_checkpoint(completed_files + newly_processed_files, ckpt_file)
    logger.info(f"Processing complete! Successfully processed {len(newly_processed_files)} files")
    logger.info(f"Total files processed: {len(completed_files) + len(newly_processed_files)}")

if __name__ == "__main__":
    main()