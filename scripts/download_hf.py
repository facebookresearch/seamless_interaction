# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from seamless_interaction.fs import SeamlessInteractionFS


def main():
    fs = SeamlessInteractionFS()
    # to download a single batch
    fs.download_batch_from_hf(
        label="improvised",
        split="dev",
        batch_idx=0,
        num_workers=None,  # auto-detect the number of workers
        archive_list=[0, 23]
    )
    
    # to download a subset of batches
    # fs.download_batch_from_hf(
    #     label="improvised",
    #     split="dev",
    #     batch_idx=[0, 1, 2],
    # )
    
    # to download all the batches given a split
    # fs.download_batch_from_hf(
    #     label="improvised",
    #     split="dev",
    #     batch_idx=None,
    # )
    
    # to download all the datasets, simply use huggingface_hub
    # beware that this will download the entire dataset, which is 4000+ hours of data (~27TB)
    # from huggingface_hub import snapshot_download
    # snapshot_download(repo_id="facebook/seamless-interaction", repo_type="dataset")


if __name__ == "__main__":
    main()
