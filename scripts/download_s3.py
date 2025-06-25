# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from seamless_interaction.fs import SeamlessInteractionFS


def main():
    fs = SeamlessInteractionFS()
    # download a single datapoint
    fs.gather_file_id_data_from_s3(
        "V00_S0809_I00000582_P0947",
        local_dir=None,  # local directory to download to, default is ~/datasets/seamless_interaction
        num_workers=None,  # auto-detect the number of workers
    )

    # download a list of datapoints
    fs.download_batch_from_s3(
        ["V00_S0809_I00000582_P0947", "V00_S0809_I00000582_P0947"],
        local_dir=None,  # local directory to download to, default is ~/datasets/seamless_interaction
    )


if __name__ == "__main__":
    main()
