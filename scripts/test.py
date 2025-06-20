from huggingface_hub import snapshot_download

from typing import Final

repo_id: Final = "rajjanardhan00/Seamless_Dummy_Dataset_Fixed"
repo_type: Final = "dataset"

snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir="data", local_dir_use_symlinks=False, allow_patterns=["**/dev/*", "**/*.csv"])
