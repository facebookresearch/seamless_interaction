from huggingface_hub import HfApi

# python scripts/3_hf_upload_webdataset.py --label naturalistic --split train --workers 190 --batch-id 17

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
        "--batch-id",
        type=int,
        default=0,
        help="Batch ID for the dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=98,
        help="Number of worker processes to use for uploading",
    )
    return parser

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    label = args.label
    split = args.split
    batch_id = args.batch_id
    workers = args.workers
    
    api = HfApi()
    destination_dataset = "facebook/seamless-interaction"

    target_folder = f"/checkpoint/seamless_fs2/yaoj/data/seamless_interaction_webdataset_sharded_tar_0623_train/{label}_{split}_{batch_id}"
    # api.upload_folder(
    #     folder_path="/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_sharded_tar/naturalistic/test/0003/",
    #     path_in_repo="naturalistic/test/0003/",
    #     repo_id=destination_dataset,
    #     repo_type="dataset",
    # )

    api.upload_large_folder(
        repo_id=destination_dataset,
        repo_type="dataset",
        folder_path=target_folder,
        num_workers=workers,
    )


if __name__ == "__main__":
    main()
