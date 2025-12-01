# Combine Datasets
from lerobot.datasets.dataset_tools import merge_datasets
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
hub_api = HfApi()
import os
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

os.environ["HF_HOME"] = "/ssd1t/david/huggingface"
os.environ["HF_HOME_HUB"] = "/ssd1t/david/huggingface/hub"
HF_DATASET_ROOT = "/ssd1t/david/datasets"

ds_0 = LeRobotDataset(repo_id="davidlau90/pickup_toy_457", root=HF_DATASET_ROOT+"/davidlau90/pickup_toy_457")
ds_1 = LeRobotDataset(repo_id="davidlau90/grab_toy_1")
out_repo_id = "davidlau90/grab_and_pickup"
merged_ds = merge_datasets(
    datasets = [ds_0, ds_1],
    output_repo_id = Path(out_repo_id),
    output_dir = Path(f"{HF_DATASET_ROOT}/{out_repo_id}"),
)

# Verify
df_out = LeRobotDataset(repo_id=out_repo_id, root=HF_DATASET_ROOT+f"/{out_repo_id}")
df_out.push_to_hub()
hub_api.create_tag(out_repo_id, tag="v3.0", repo_type="dataset")