# python scripts/0_reorganize_dataset.py --label improvised --split train
import logging
import shutil
import os
import glob

"""
This script scans the target folder for batches and upload them to the Hugging Face dataset repository.
Each upload session ideally will upload 500GB to 1TB of data, depending on the number of shards and their sizes.
For example, if each shard is around 50GB, then 10 shards will be uploaded in one session (500GB).
The target_folder should have the following structure:
- target_folder/
    - improvised/
        - train/
            - 0000/
                0000.tar
                0001.tar
                ...
            - 0001/
                0000.tar
                0001.tar
                ...
            ...            
    - naturalistic/
    ...
The script will first scan the target folder for all subfolders, then it will create new folders for each batch
and upload them to the Hugging Face dataset repository.
For example, given the following input arguments:
--label improvised --split train --target-folder /path/to/target_folder
We will create a list of folders to upload:
- /path/to/{target_folder}_improvised_train_by_{number_of_shards}_0/
    - improvised/
        - train/
            - 0000/
            ...
            - 0009/
- /path/to/{target_folder}_improvised_train_by_{number_of_shards}_1/
    - improvised/
        - train/
            - 0010/
            ...
            - 0019/
...
- /path/to/{target_folder}_improvised_train_by_{number_of_shards}_9/
    - improvised/
        - train/
            - 0090/
            ...
            - 0097/  # if this is the last batch, it might have fewer than 10 shards
"""

target_folder = "/checkpoint/seamless_fs2/yaoj/data/seamless_interaction_webdataset_sharded_tar_0623_train/"
number_of_shards = 10

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
        "--target-folder",
        type=str,
        default=target_folder,
        help="Path to the target folder containing the dataset",
    )
    return parser

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    label = args.label
    split = args.split
    target_folder = args.target_folder
    
    all_batches = sorted(set([i[:-1] for i in os.listdir(os.path.join(target_folder, label, split))]))
    
    for i, batch_prefix in enumerate(all_batches):
        batch_folder = os.path.join(target_folder, f"{label}_{split}_{i}")
        target_label_split_folder = os.path.join(target_folder, label, split)
        batched_label_split_folder = os.path.join(batch_folder, label, split)
        os.makedirs(batch_folder, exist_ok=True)
        os.makedirs(os.path.join(batch_folder, label), exist_ok=True)
        os.makedirs(batched_label_split_folder, exist_ok=True)
        
        # Find all subfolders for the current batch
        subfolders = sorted(glob.glob(os.path.join(target_label_split_folder, f"{batch_prefix}*")))
        
        # Copy the subfolders to the batch folder
        for subfolder in subfolders:
            if os.path.isdir(subfolder):
                shutil.move(subfolder, batched_label_split_folder)
                # print(f"Moving {subfolder} to {batched_label_split_folder}")
        
        logger.info(f"Created batch folder: {batch_folder}")

if __name__ == "__main__":
    main()