from datasets import load_dataset
from utils.fs import SeamlessInteractionFS
from pathlib import Path

fs = SeamlessInteractionFS()
local_dir = Path.home() / "datasets/seamless_interaction"

fs.download_archive_from_hf(
    idx=0,
    archive=23,
    label="improvised",
    split="dev",
    batch=0,
    local_dir=local_dir,
    extract=False,
)

local_path = local_dir / "improvised/dev/0000/0023.tar"
dataset_local = load_dataset("webdataset", data_files={"dev": local_path}, split="dev", streaming=True)

for item in dataset_local:
    break