# Combine Datasets
from lerobot.datasets.dataset_tools import (
    merge_datasets,
)
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
hub_api = HfApi()
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

HF_LEROBOT_ROOT = "/Users/david/dev/lerobot/datasets"
# HF_LEROBOT_ROOT = "/ssd1t/david/lerobot/datasets"
ds_0 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_0", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_0")
ds_1 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_1", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_1")
ds_2 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_2", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_2")
ds_3 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_3", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_3")
out_repo_id = "davidlau90/lekiwi_toy_0123"
merged_ds = merge_datasets(
    datasets = [ds_0, ds_1, ds_2, ds_3],
    output_repo_id = Path(out_repo_id),
    output_dir = Path(f"{HF_LEROBOT_ROOT}/{out_repo_id}"),
)

# Verify
df_out = LeRobotDataset(repo_id=out_repo_id, root=HF_LEROBOT_ROOT+f"/{out_repo_id}")
df_out.push_to_hub()
hub_api.create_tag(out_repo_id, tag="v3.0", repo_type="dataset")