from datasets import load_dataset
from utils.fs import SeamlessInteractionFS
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--label", type=str, default="improvised")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--archive_idx", type=int, default=23)
    args = parser.parse_args()

    fs = SeamlessInteractionFS()
    local_dir = Path.home() / "datasets/seamless_interaction"
    mode = args.mode
    label = args.label
    split = args.split
    batch_idx = args.batch_idx
    archive_idx = args.archive_idx

    fs.download_archive_from_hf(
        idx=batch_idx,
        archive=archive_idx,
        label=label,
        split=split,
        batch=batch_idx,
        local_dir=local_dir,
        extract=False,
    )

    if mode == "local":
        local_path = (
            local_dir / f"{label}/{split}/{batch_idx:04d}/{archive_idx:04d}.tar"
        )
        dataset = load_dataset(
            "webdataset", data_files={split: local_path}, split=split, streaming=True
        )
    elif mode == "hf":
        base_url = f"https://huggingface.co/datasets/facebook/seamless-interaction/resolve/main/{label}/{split}/{batch_idx:04d}/{archive_idx:04d}.tar"
        urls = [base_url.format(batch_idx=batch_idx, archive_idx=archive_idx)]
        dataset = load_dataset(
            "webdataset", data_files={split: urls}, split=split, streaming=True
        )

    for item in dataset:
        break

    print(item.keys())


if __name__ == "__main__":
    main()
