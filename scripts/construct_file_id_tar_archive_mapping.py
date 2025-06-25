#!/usr/bin/env python3
"""
Script to construct file ID to tar archive mapping from webdataset sharded
tar files. Supports multiprocessing for improved performance.
"""

import tarfile
import os
import glob
import pandas as pd
import argparse
import logging
import json
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the script"""
    split_tar_dirs: List[Tuple[str, str]] = None
    labels: List[str] = None
    output_file: str = "all_file_id_to_hf_paths.csv"
    num_processes: Optional[int] = None
    
    def __post_init__(self):
        if self.split_tar_dirs is None:
            # Default configuration
            self.split_tar_dirs = [
                ("test", 
                 "/checkpoint/seamless/data/"
                 "seamless_interaction_webdataset_sharded_tar_0623_test"),
                ("dev", 
                 "/checkpoint/seamless/data/"
                 "seamless_interaction_webdataset_sharded_tar_0623_dev"),
                ("train", 
                 "/checkpoint/seamless/data/"
                 "seamless_interaction_webdataset_sharded_tar_0623_train"),
            ]
        if self.labels is None:
            self.labels = ["improvised", "naturalistic"]
        if self.num_processes is None:
            # Cap at 8 to avoid overwhelming system
            self.num_processes = min(cpu_count(), 8)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def process_archive(archive_path: str) -> List[Dict[str, str]]:
    """
    Process a single tar archive and extract file IDs with metadata.
    
    Args:
        archive_path: Path to the tar archive
        
    Returns:
        List of dictionaries containing file metadata
    """
    logger = logging.getLogger(__name__)
    results = []
    
    try:
        with tarfile.open(archive_path, "r") as tar:
            file_id_list = set([f.split(".")[0] for f in tar.getnames()])
            
            # Extract metadata from path
            path_parts = archive_path.split("/")
            archive_filename = path_parts[-1].split(".")[0]
            batch = path_parts[-2]
            split = path_parts[-3]
            label = path_parts[-4]
            
            for file_id in file_id_list:
                results.append({
                    "file_id": file_id,
                    "label": label,
                    "split": split,
                    "batch_idx": batch,
                    "archive_idx": archive_filename,
                    "archive_path": archive_path
                })
                 
        logger.debug(f"Processed {archive_path}: {len(results)} files")
        
    except Exception as e:
        logger.error(f"Error processing archive {archive_path}: {e}")
        
    return results


def process_batch_worker(
    batch_info: Tuple[str, str, str, str]
) -> List[Dict[str, str]]:
    """
    Worker function to process a single batch (collection of archives).
    
    Args:
        batch_info: Tuple of (label, split, batch_path, tar_dir)
        
    Returns:
        List of dictionaries containing file metadata for all archives in batch
    """
    label, split, batch_path, tar_dir = batch_info
    logger = logging.getLogger(__name__)
    
    archives = glob.glob(os.path.join(batch_path, "*.tar"))
    if not archives:
        logger.warning(f"No tar files found in {batch_path}")
        return []
    
    logger.info(f"Processing {len(archives)} archives in batch {batch_path}")
    
    all_results = []
    for archive in archives:
        results = process_archive(archive)
        all_results.extend(results)
    
    return all_results


def collect_batch_info(config: Config) -> List[Tuple[str, str, str, str]]:
    """
    Collect information about all batches to process.
    
    Args:
        config: Configuration object
        
    Returns:
        List of batch information tuples
    """
    logger = logging.getLogger(__name__)
    batch_info_list = []
    
    for split, tar_dir in config.split_tar_dirs:
        if not os.path.exists(tar_dir):
            logger.warning(
                f"Directory {tar_dir} for split '{split}' does not exist. "
                f"Skipping."
            )
            continue
            
        logger.info(f"Processing split '{split}' from directory: {tar_dir}")
        
        for label in config.labels:
            if split == "train":
                # Handle train split with bucket structure
                buckets = glob.glob(
                    os.path.join(tar_dir, f"{label}_{split}_*")
                )
                for bucket_path in buckets:
                    if not os.path.isdir(bucket_path):
                        continue
                    batches = glob.glob(
                        os.path.join(bucket_path, label, split, "*")
                    )
                    for batch_path in batches:
                        if os.path.isdir(batch_path):
                            batch_info_list.append(
                                (label, split, batch_path, tar_dir)
                            )
            else:
                # Handle dev/test splits
                batches = glob.glob(os.path.join(tar_dir, label, split, "*"))
                for batch_path in batches:
                    if os.path.isdir(batch_path):
                        batch_info_list.append(
                            (label, split, batch_path, tar_dir)
                        )
    
    logger.info(f"Found {len(batch_info_list)} batches to process")
    return batch_info_list


def process_batches_parallel(
    batch_info_list: List[Tuple[str, str, str, str]], 
    num_processes: int
) -> pd.DataFrame:
    """
    Process all batches in parallel using multiprocessing.
    
    Args:
        batch_info_list: List of batch information tuples
        num_processes: Number of processes to use
        
    Returns:
        DataFrame containing all file metadata
    """
    logger = logging.getLogger(__name__)
    
    if not batch_info_list:
        logger.warning("No batches to process")
        return pd.DataFrame()
    
    logger.info(
        f"Processing {len(batch_info_list)} batches using "
        f"{num_processes} processes"
    )
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_batch_worker, batch_info_list)
    
    # Flatten results
    all_file_data = []
    for batch_results in results:
        all_file_data.extend(batch_results)
    
    logger.info(f"Processed {len(all_file_data)} total files")
    return pd.DataFrame(all_file_data)


def parse_split_dirs(split_dirs_str: str) -> List[Tuple[str, str]]:
    """
    Parse the split directories string into a list of tuples.
    
    Args:
        split_dirs_str: JSON string or comma-separated list
        
    Returns:
        List of (split, tar_dir) tuples
    """
    try:
        # Try to parse as JSON first
        return json.loads(split_dirs_str)
    except json.JSONDecodeError:
        # Fall back to simple comma-separated parsing
        # Format: split1:dir1,split2:dir2
        pairs = []
        for pair in split_dirs_str.split(','):
            if ':' in pair:
                split, tar_dir = pair.split(':', 1)
                pairs.append((split.strip(), tar_dir.strip()))
        return pairs


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Construct file ID to tar archive mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories
  python script.py
  
  # Custom output and processes
  python script.py -o my_output.csv -p 4
  
  # Specify custom split directories (JSON format)
  python script.py --split-dirs '[["dev", "/path/to/dev"], ["test", "/path/to/test"]]'
  
  # Specify custom split directories (simple format)
  python script.py --split-dirs 'dev:/path/to/dev,test:/path/to/test'
        """
    )
    parser.add_argument(
        "--output", "-o", default="all_file_id_to_hf_paths.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--processes", "-p", type=int, 
        help="Number of processes to use (default: auto)"
    )
    parser.add_argument(
        "--split-dirs", type=str,
        help=(
            "Split directories as JSON list of [split, dir] pairs or "
            "comma-separated 'split:dir' format. "
            "Example: '[\"dev\", \"/path/to/dev\"], "
            "[\"test\", \"/path/to/test\"]'"
        )
    )
    parser.add_argument(
        "--labels", nargs='+', default=["improvised", "naturalistic"],
        help="Labels to process (default: improvised naturalistic)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Parse split directories if provided
    split_tar_dirs = None
    if args.split_dirs:
        try:
            split_tar_dirs = parse_split_dirs(args.split_dirs)
            logger.info(f"Using custom split directories: {split_tar_dirs}")
        except Exception as e:
            logger.error(f"Error parsing split directories: {e}")
            logger.error("Expected format: JSON list or 'split:dir,split:dir'")
            return 1
    
    # Create configuration
    config = Config(
        split_tar_dirs=split_tar_dirs,
        labels=args.labels,
        output_file=args.output,
        num_processes=args.processes
    )
    
    if not split_tar_dirs:
        logger.info("Using default split directories:")
        for split, tar_dir in config.split_tar_dirs:
            logger.info(f"  {split}: {tar_dir}")
    
    # Collect batch information
    batch_info_list = collect_batch_info(config)
    
    if not batch_info_list:
        logger.error("No batches found to process")
        return 1
    
    # Process batches in parallel
    df = process_batches_parallel(batch_info_list, config.num_processes)
    
    if df.empty:
        logger.error("No data processed")
        return 1
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully saved {len(df)} records to {output_path}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Splits processed: {df['split'].value_counts().to_dict()}")
    logger.info(f"Labels processed: {df['label'].value_counts().to_dict()}")
    
    return 0


if __name__ == "__main__":
    exit(main())