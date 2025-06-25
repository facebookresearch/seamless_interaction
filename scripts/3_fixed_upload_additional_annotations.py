"""Example cli command:
python scripts/3_fixed_upload_additional_annotations.py --annotations_path /home/jasonz/captions_corrected_segmented_v4/updated_annotations_mosaic_6_23_25_v4/ \
    --shard_path_prefix /checkpoint/seamless/data/seamless_interaction_webdataset_sharded_tar_0623 \
    --raw_dataset_path /checkpoint/seamless/yaoj/data/seamless_interaction_webdataset_raw/ \
    --hf_dataset facebook/seamless-interaction --dry_run True --test True
"""

from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from huggingface_hub import HfApi
from pathlib import Path

import tarfile
import argparse
import os
import glob
import json
import io


_DATASET_LABEL = "dataset"

def gen_updated_tars(annotations_path_list: List[str], base_raw_dataset_path: str, shard_dataset_prefix: str, hf_api: HfApi, hf_dataset: str, dry_run: bool, test: bool):
    """Given a list of annotations to update, identifies the sharded tars to update and creates and 
    submits a new tar file to upload to hf
    """
    for annotation_path in tqdm(annotations_path_list):
        acting_type, split, annotation_type, sample_id = _get_annotation_identifiers(annotation_path)
        shard = _get_update_shard(
            base_raw_dataset_path,
            acting_type,
            split,
            sample_id
        )
        tar_file_path = _get_update_tar(
            shard_dataset_prefix,
            split,
            acting_type,
            shard,
            sample_id
        )
        if tar_file_path is None:
            print(f'ERROR: Found None tarfile path for {annotation_path}')
            continue
        annotation_content = _get_annotation_content(annotation_path)
        if test:
            print('================================= Sample Log =================================')
            print(tar_file_path)
            print(f"Annotation path: {annotation_path}")
            print(annotation_content)
            _print_tarfile_contents(tar_file_path)

        out_tar_file_path = gen_append_tar(sample_id, tar_file_path, annotation_content, annotation_type)
        if test:
            _print_tarfile_contents(out_tar_file_path)
            print('==============================================================================')

        if not dry_run:
            gen_upload_to_hf(
                acting_type,
                split,
                shard,
                out_tar_file_path,
                hf_api=hf_api,
                hf_dataset=hf_dataset
            )


def gen_append_tar(sample_id: str, tar_file_path: str, annotation_content: List[Dict[str, Any]], annotation_type: str):
    """Generates a single tar file, which includes only the updated entry that we want to append to 
    the existing tar
    """
    out_tar_file_path = tar_file_path.replace('/data/', '/jasonz/')
    os.makedirs(Path(out_tar_file_path).parent, exist_ok=True)
    
    with tarfile.open(tar_file_path, 'r:*') as rt:
        with tarfile.open(out_tar_file_path, 'a') as wt:
            for member in rt.getmembers():
                file_obj = rt.extractfile(member)
                existing_members = [m.name for m in wt.getmembers()]
                # If we need to append an annotation to this sample_id, do so, even if another annotation has already been appended
                if member.name == f'{sample_id}.json' and file_obj is not None and member.name not in existing_members:
                    content = json.load(file_obj)
                    content[f"annotations:{annotation_type}"] = annotation_content
                    json_bytes = json.dumps(content, indent=2).encode('utf-8')
                    new_file_obj = io.BytesIO(json_bytes)
                    member.size = len(json_bytes)
                    wt.addfile(member, new_file_obj)

                # We only want to copy over the rest if it hasn't been copied already from a previous annotation addition
                elif member.name not in existing_members:
                    if file_obj is not None:
                        wt.addfile(member, file_obj)
                    else:
                        wt.addfile(member)
    return out_tar_file_path


def gen_upload_to_hf(
    acting_type: str, 
    split: str, 
    shard: str, 
    output_tar_path: str, 
    hf_api: HfApi, 
    hf_dataset: str
):
    tar_file_path = os.path.basename(output_tar_path)
    upload_path = os.path.join(acting_type, split, shard, tar_file_path)
    hf_api.upload_file(
        path_or_fileobj=output_tar_path,
        path_in_repo=upload_path,
        repo_id=hf_dataset,
        repo_type=_DATASET_LABEL,
    )


def _get_annotation_content(annotation_path: str) -> List[Dict[str, Any]]:
    with open(annotation_path, "r") as f_json:
        annotation_content = [json.loads(line) for line in f_json]
        if not annotation_content:
            print(f"Empty annotation content for: {annotation_path}!")
    return annotation_content


def _get_update_tar(shard_dataset_prefix: str, split: str, acting_type: str, shard: str, sample_id: str):
    tar_dir = os.path.join(f"{shard_dataset_prefix}_{split}", acting_type, split, shard)
    for tar_file in os.listdir(tar_dir):
        full_path = os.path.join(tar_dir, tar_file)
        with tarfile.open(full_path, "r:*") as t:
            for member in t.getmembers():
                if member.name == f'{sample_id}.json':
                    return full_path
    return None


def _get_update_shard(
    base_raw_dataset_path: str,
    acting_type: str, 
    split: str, 
    sample_id: str, 
) -> str:
    """Given an annotation path identifiers, identify the tar shard that needs to be updated
    """
    # Hardcoded based on current subpath pattern
    subpath_pattern = f"{acting_type}-{split}-*/{sample_id}.json"
    candidate_paths = glob.glob(os.path.join(base_raw_dataset_path, subpath_pattern))
    assert len(candidate_paths) == 1, f"Found != 1 candidate paths for {subpath_pattern}"

    shard = os.path.basename(Path(candidate_paths[0]).parent).split('-')[-1]
    return shard


def _get_annotation_identifiers(annotation_path: str) -> Tuple[str, str, str, str]:
    path_items = annotation_path.split('/')
    acting_type = path_items[5]
    split = path_items[6]
    annotation_type = path_items[8]
    sample_id = os.path.splitext(path_items[9])[0]

    return acting_type, split, annotation_type, sample_id


def _get_annotations_path_list(annotations_path: str) -> List[str]:
    """Given a root directory containing annotations, obtain a list of all the paths to annotations
    """
    return glob.glob(f"{annotations_path}/**/*.json", recursive=True)


def _print_tarfile_contents(tar_file_path: str) -> None:
    print(f'INFO: printing tarfile contents for {tar_file_path}')
    with tarfile.open(tar_file_path, 'r:*') as rt:
        print(f"Number of members: {len(rt.getmembers())}")
        for member in rt.getmembers():
            print(member.name)
            if member.name.endswith('.json'):
                file_obj = rt.extractfile(member)
                print(json.load(file_obj).keys())
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update and upload tar files given annotations to HuggingFace")
    parser.add_argument('--annotations_path', type=str, required=True)
    parser.add_argument('--shard_path_prefix', type=str, required=True)
    parser.add_argument('--raw_dataset_path', type=str, required=True)
    parser.add_argument('--hf_dataset', type=str, required=True)
    parser.add_argument('--dry_run', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    annotations_path_list = _get_annotations_path_list(args.annotations_path)

    if args.test:
        annotations_path_list = annotations_path_list[:10]

    hf_api = HfApi()
    gen_updated_tars(annotations_path_list, args.raw_dataset_path, args.shard_path_prefix, hf_api=hf_api, hf_dataset=args.hf_dataset, dry_run=args.dry_run, test=args.test)