import os
import logging
import sys
import numpy as np
import os
from glob import glob

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
        if not os.path.exists(npz_path):
            continue
        sample = {
            "__key__": file_id,
            "npz": dict(np.load(npz_path, allow_pickle=True)),
            "json": open(json_path, "r", encoding="utf-8").read(),
            "wav": open(audio_path, "rb").read(),
            "mp4": open(video_path, "rb").read(),
        }
        yield sample

def main():
    # Set up signal handler for graceful shutdown
    import signal

    signal.signal(signal.SIGINT, handle_sigint)

    # Parse command line arguments
    args = setup_arg_parser().parse_args()

    label, split = args.label, args.split
    data_dir_root = "/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_raw"
    target_dir = "/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_sharded_tar"
    input_dirs = glob(f"{data_dir_root}/{label}-{split}-*")
    for input_dir in input_dirs:
        # this could be the work of a single worker
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory {input_dir} does not exist. Skipping.")
            continue
        files = os.listdir(input_dir)
        file_id_list = set([f.split("/")[-1].split(".")[0] for f in files if f.endswith((".npz", ".json", ".wav", ".mp4"))])
        batch_idx = int(input_dir.split("-")[-1])
        os.makedirs(f"{target_dir}/{label}", exist_ok=True)
        os.makedirs(f"{target_dir}/{label}/{split}", exist_ok=True)
        os.makedirs(f"{target_dir}/{label}/{split}/{batch_idx:04d}", exist_ok=True)
        out_pattern = f"{target_dir}/{label}/{split}/{batch_idx:04d}/%04d.tar"
        sind = wds.ShardWriter(out_pattern, maxcount=1000, maxsize=1_000_000_000)
        for sample in data_iter(file_id_list=file_id_list, input_dir=input_dir):
            sind.write(sample)
        sind.close()

if __name__ == "__main__":
    main()
