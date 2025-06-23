from huggingface_hub import HfApi

# done
# naturalistic/dev/0002/
# naturalistic/dev/0001/
# naturalistic/dev/0000/
# improvised/dev/0000/
# improvised/dev/0001/
# improvised/test/*
# naturalistic/test/0000/
# naturalistic/test/0001/
# naturalistic/test/0002/
# naturalistic/test/0003/

# to be sharded
# naturalistic/train/0170..0183/

# to be reorganized
# naturalistic/train/0000..0030/
# naturalistic/train/0030..0060/
# naturalistic/train/0060..0080/
# naturalistic/train/0080..0100/
# naturalistic/train/0100..0130/
# naturalistic/train/0130..0150/
# naturalistic/train/0150..0170/
# improvised/train/0000..0020/
# improvised/train/0020..0040/
# improvised/train/0040..0060/
# improvised/train/0060..0070/

# python scripts/0_reorganize_dataset.py --label improvised --split train --max-processes 1 --batch-start 40 --batch-end 60

api = HfApi()
destination_dataset = "facebook/seamless-interaction-webdataset"
# api.upload_folder(
#     folder_path="/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_sharded_tar/naturalistic/test/0003/",
#     path_in_repo="naturalistic/test/0003/",
#     repo_id=destination_dataset,
#     repo_type="dataset",
# )

api.upload_large_folder(
    repo_id=destination_dataset,
    repo_type="dataset",
    folder_path="/checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_sharded_tar_dev_test/",
)


if __name__ == "__main__":
    main()