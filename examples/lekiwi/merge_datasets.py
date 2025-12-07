# Combine Datasets
from lerobot.datasets.dataset_tools import merge_datasets
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
hub_api = HfApi()
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/Volumes/KIOXIA2T/huggingface"
os.environ["HF_HOME_HUB"] = "/Volumes/KIOXIA2T/huggingface/hub"
HF_DATASET_ROOT = "/Volumes/KIOXIA2T/huggingface/lerobot"

ds_1 = LeRobotDataset(repo_id="davidlau90/basket_1", root=HF_DATASET_ROOT+"/davidlau90/basket_1")
ds_2 = LeRobotDataset(repo_id="davidlau90/basket_2", root=HF_DATASET_ROOT+"/davidlau90/basket_2")
ds_3 = LeRobotDataset(repo_id="davidlau90/basket_3", root=HF_DATASET_ROOT+"/davidlau90/basket_3")
ds_4 = LeRobotDataset(repo_id="davidlau90/basket_4", root=HF_DATASET_ROOT+"/davidlau90/basket_4")
ds_5 = LeRobotDataset(repo_id="davidlau90/basket_5", root=HF_DATASET_ROOT+"/davidlau90/basket_5")
out_repo_id = "davidlau90/basket_12345"
merged_ds = merge_datasets(
    datasets = [ds_1, ds_2, ds_3, ds_4, ds_5],
    output_repo_id = Path(out_repo_id),
    output_dir = Path(f"{HF_DATASET_ROOT}/{out_repo_id}"),
)

# Verify and tag as v3.0 format
df_out = LeRobotDataset(repo_id=out_repo_id, root=HF_DATASET_ROOT+f"/{out_repo_id}")
hub_api.create_tag(out_repo_id, tag="v3.0", repo_type="dataset")

# Push to hub
df_out.push_to_hub()