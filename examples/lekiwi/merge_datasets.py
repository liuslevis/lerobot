# Combine Datasets
from lerobot.datasets.dataset_tools import (
    merge_datasets,
)
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
HF_LEROBOT_ROOT = "/Users/david/.cache/huggingface/lerobot"
# HF_LEROBOT_ROOT = "/ssd1t/david/lerobot/datasets"

ds4 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_pickup_4", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_pickup_4")
ds5 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_pickup_5", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_pickup_5")
# bad ds6 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_pickup_6", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_pickup_6")
ds7 = LeRobotDataset(repo_id="davidlau90/lekiwi_toy_pickup_7", root=HF_LEROBOT_ROOT+"/davidlau90/lekiwi_toy_pickup_7")
out_repo_id = "davidlau90/lekiwi_toy_pickup_457"
merged_ds = merge_datasets(
    datasets = [ds4, ds5, ds7],
    output_repo_id = Path(out_repo_id),
    output_dir = Path(f"{HF_LEROBOT_ROOT}/{out_repo_id}"),
)

ds_simple = lekiwi_toy_pickup_simple = LeRobotDataset(repo_id=out_repo_id, root=HF_LEROBOT_ROOT+f"/{out_repo_id}")
ds_simple.push_to_hub()