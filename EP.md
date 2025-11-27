
# TODO
[] Ubuntu Record / Inference video Play Slow, maybe try cuda decode https://github.com/huggingface/lerobot/pull/913/files
[] Learn 1 Grab -> 2 Placement

# Calibration

```
/Users/david/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/di.json # Mac
/root/.cache/huggingface/lerobot/calibration/robots/lekiwi/didi.json # RP5
```


# RP5 Sync
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=didi --host.connection_time_s=36000 --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\" , width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 30}}"

# PC Sync
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python examples/lekiwi/evaluate.py 


# PC Async
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=9999

# RP Async

Notes: X observation per chunk 
python -m lerobot.async_inference.robot_client     \
    --robot.type=lekiwi     \
    --robot.port=/dev/ttyACM0     \
    --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\" , width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 30}}"     \
    --robot.id=didi     \
    --task="pick up toys\n"     \
    --server_address=192.168.0.78:9999     \
    --policy_type=pi05     \
    --pretrained_name_or_path=/ssd1t/david/lerobot/outputs/pi05_toy_457/checkpoints/003000/pretrained_model     \
    --policy_device=cuda     \
    --actions_per_chunk=50     \
    --chunk_size_threshold=0.5     \
    --aggregate_fn_name=weighted_average   \
    --debug_visualize_queue_size=True


python -m lerobot.async_inference.robot_client     \
    --robot.type=lekiwi     \
    --robot.port=/dev/ttyACM0     \
    --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\" , width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 30}}"     \
    --robot.id=didi     \
    --task="pick up toys\n"     \
    --server_address=192.168.0.78:9999     \
    --policy_type=pi05     \
    --pretrained_name_or_path=/ssd1t/david/lerobot/outputs/pi05_toy_457/checkpoints/003000/pretrained_model     \
    --policy_device=cuda     \
    --actions_per_chunk=10     \
    --chunk_size_threshold=0.5     \
    --aggregate_fn_name=weighted_average   \
    --debug_visualize_queue_size=False


```python
import pickle
from PIL import Image
import numpy as np

def save(observation, obs='wrist', i=0):
    img_array = observation.observation[obs]
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'obs/{obs}_{i}.png')  


for i in range(0, 44):
    with open(f'obs/obs_{i}.pkl', 'rb') as f:  # 注意：必须用 'rb'（读二进制）
        observation = pickle.load(f)
        save(observation, 'wrist', i)
        save(observation, 'front', i)




print(type(observation))
print(observation)
# TimedObservation(timestamp=1763884202.1972432, timestep=125, observation={'arm_shoulder_pan.pos': -16.34980988593155, 'arm_shoulder_lift.pos': -3.2230703986429177, 'arm_elbow_flex.pos': 20.667870036101093, 'arm_wrist_flex.pos': 48.84341637010675, 'arm_wrist_roll.pos': 2.124542124542117, 'arm_gripper.pos': 19.579945799457995, 'x.vel': np.float64(0.01771288441634978), 'y.vel': np.float64(0.010226538585904275), 'theta.vel': np.float64(4.6875), 'front': array([[[ 43, 105, 130],
#         [ 43, 105, 130],
#         [ 47, 103, 123],
#         ...,
```



LeKiwi Setup


具身智能π0.5(pi0.5)模型在lerobot机械臂上复现 知乎 笔者就采了50个任务，积木放在5个固定的点位，平均每点位10个任务，每个点位积木的朝向在360度范围内随机均匀摆放。lerobot文档建议实验时不要引入太多的变量，所以我就只在积木摆放的点位上有一些多样性，积木和盘子用的是固定的。



Setup
【5-lerobot 校准、遥操及异常处理-哔哩哔哩】 https://b23.tv/ev3KDvP


hostname -I
192.168.0.207 198.18.0.1

chmod +x ~/Downloads/Miniforge3-Darwin-arm64.sh
~/Downloads/Miniforge3-Darwin-arm64.sh
conda init
bash
conda create -y -n lerobot python=3.10
conda activate lerobot
# conda install feetech-servo-sdk==1.0.0
conda install -c conda-forge ffmpeg=6.1.1 -y


git clone https://github.com/huggingface/lerobot
git clone https://github.com/liuslevis/lerobot
vim pyproject.toml # https://github.com -> https://gh-proxy.com/https://github.com  
pip install -e ".[lekiwi]"
pip install -e ".[pi]"
pip install -e ".[async]"



# Test Camera (RP5)
* cd ~/lerobot/tests && python test_cam.py
* vim src/lerobot/robots/lekiwi/config_lekiwi.py

# Calibration
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5AB01813381 --teleop.id=di
lerobot-calibrate --robot.type=lekiwi --robot.id=didi

# Start Tele Op (PC leader arm)
python examples/lekiwi/teleoperate.py 

# Start Host (RP)
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=didi --host.connection_time_s=36000 --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\" , width: 640, height: 480, fps: 15}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 15}}"


# Dataset Record 
python -i examples/lekiwi/record_toy.py

- Right arrow key pressed. Exiting loop...
- Left arrow key pressed. Exiting loop and rerecord the last episode
- Escape key pressed. Stopping data recording
- F: slower
- R: faster
- ASDW: move
- Z X: rotate left right
- Q: quit tele op

# Dataset Upload 
hf upload davidlau90/lekiwi_toy_pickup_2 ~/.cache/huggingface/lerobot/davidlau90/lekiwi_toy_pickup_2 --repo-type dataset

训练数据路径：
~/.cache/huggingface/lerobot/davidlau90/lekiwi_toy_pickup_1 # 10 次 移动抓放
~/.cache/huggingface/lerobot/davidlau90/lekiwi_toy_pickup # 20 次 固定位置抓放后移动
~/.cache/huggingface/lerobot/davidlau90/lekiwi_toy_pickup_simple # 单一物品固定抓放，不移动




# Replay Rec.
python examples/lekiwi/replay.py

# Visualize Dataset (HF -> rdd -> rerun.io)
python lerobot_dataset_viz.py




# PC 5080 sm120 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128




# Async Policy Server Start PC
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=9999

# Async Client Start RP5
export ACT_PER_CHUNK=50
export CKPT=/ssd1t/david/lerobot/outputs/pi05_toy_457/checkpoints/003000/pretrained_model

export ACT_PER_CHUNK=2
export CKPT=/ssd1t/david/lerobot/outputs/pi05_toy_0123_again/checkpoints/000200/pretrained_model # grab robot itself
export CKPT=/ssd1t/david/lerobot/outputs/pi05_toy_0123_again/checkpoints/001000/pretrained_model # movement ok, but cannot grab

python -m lerobot.async_inference.robot_client \
    --robot.type=lekiwi \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\"
, width: 640, height: 480, fps: 15}, wrist: {type: opencv, index_or_path: \"/dev/video2\"
, width: 640, height: 480, fps: 15}}" \
    --robot.id=didi \
    --task="pick up toys\n" \
    --server_address=192.168.0.78:9999 \
    --policy_type=pi05 \
    --pretrained_name_or_path=$CKPT \
    --policy_device=cuda \
    --actions_per_chunk=${ACT_PER_CHUNK} \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True


# BitaHub A100: 
```
pytorch:2.3.1-cuda12.1-cudnn8-py310-ubuntu22.04

# mkdir -p /ssd1t/david/opt/
# cd /ssd1t/david/opt
# tar -cvf opt_conda.tar /opt/conda
# tar -xvf opt_conda.tar
mv /opt/conda /opt/conda_bak
ln -s /ssd1t/david/opt/conda /opt/conda


echo "envs_dirs:" >> ~/.condarc
echo "  - /ssd1t/david/conda/env" >> ~/.condarc
echo "pkgs_dirs:" >> ~/.condarc
echo "  - /ssd1t/david/conda/pkgs" >> ~/.condarc
conda init
bash
cd /ssd1t/david/lerobot
# conda create -y -n lerobot python=3.10
conda activate lerobot
# conda install -c conda-forge ffmpeg=6.1.1 -y &
# pip install -e ".[lekiwi,pi]"
# git config --global credential.helper store
export HUGGINGFACE_TOKEN=
export WANDB_TOKEN=
export HF_HOME=/ssd1t/david/huggingface
export HF_HOME_HUB=/ssd1t/david/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
wandb login --relogin ${WANDB_TOKEN}

# hf download hxdoso/new_frame
# hf download lerobot/pi05_base 
# hf download google/paligemma-3b-pt-224 --repo-type model
```

# Train: π0.5 A100
Notes: 
- pi05 batch_size=32, gpu_mem=41GB
- pi05 batch_size=64, gpu_mem=51GB
- pi05 batch_size=128, gpu_mem=61GB 
- pi05 batch_size=168, gpu_mem=73GB 
- pi05 batch_size=256, gpu_mem>80GB OOM

```
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/ssd1t/david/huggingface
export HF_HOME_HUB=/ssd1t/david/huggingface/hub
```

## dataset grab_toy_1 (grab 1 car)
```
screen
rm -rf outputs/pi05_grab_toy_1 && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=/ssd1t/david/lerobot/datasets/davidlau90/grab_toy_1 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_grab_toy_1 \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/pi05_grab_toy_1 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=60000 \
    --log_fre=50 \
    --batch_size=168 \
    --save_freq=100 \
    > logs/train_grab_toy_1.txt 2>&1 &
tail -f logs/train_grab_toy_1.txt

INFO 2025-11-27 01:35:39 ot_train.py:262 cfg.steps=60000 (60K)
INFO 2025-11-27 01:35:39 ot_train.py:263 dataset.num_frames=22452 (22K)
INFO 2025-11-27 01:35:39 ot_train.py:264 dataset.num_episodes=50
INFO 2025-11-27 01:35:39 ot_train.py:267 Effective batch size: 168 x 1 = 168
INFO 2025-11-27 01:35:39 ot_train.py:268 num_learnable_params=3616757520 (4B)
INFO 2025-11-27 01:35:39 ot_train.py:269 num_total_params=3616757520 (4B)
INFO 2025-11-27 01:35:39 ot_train.py:324 Start offline training on a fixed dataset
```

## dataset 0123 (1 car, clear background, 85 episodes)
```
# eval on 4090 
python src/lerobot/scripts/lerobot_train_eval_only.py \
    --dataset.repo_id=/ssd1t/david/lerobot/datasets/davidlau90/lekiwi_toy_0123 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_0123_again_eval \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/pi05_toy_0123_again \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=60000 \
    --log_fre=500 \
    --batch_size=16 \
    --save_freq=200000 \
    --resume=true \
    --config_path=outputs/pi05_toy_0123_again/checkpoints/001200/pretrained_model/train_config.json 
INFO 2025-11-26 11:35:18 ot_train.py:351 step:2.0K smpl:32K ep:47 epch:0.55 loss:0.880 grdn:0.000 lr:2.5e-05 updt_s:0.522 data_s:0.020 <- 001800
INFO 2025-11-26 11:42:01 val_only.py:351 step:1.5K smpl:24K ep:35 epch:0.41 loss:0.885 grdn:0.000 lr:2.5e-05 updt_s:0.521 data_s:0.017 <- 001200


screen
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=/ssd1t/david/lerobot/datasets/davidlau90/lekiwi_toy_0123 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_0123_again \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/pi05_toy_0123_again \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=60000 \
    --log_fre=50 \
    --batch_size=168 \
    --save_freq=200 \
    --resume=true \
    --config_path=outputs/pi05_toy_0123_again/checkpoints/last/pretrained_model/train_config.json 

INFO 2025-11-25 20:36:46 ot_train.py:324 Start offline training on a fixed dataset
INFO 2025-11-25 20:50:39 ot_train.py:351 step:50.0 smpl:8K ep:12 epch:0.14 loss:4.138 grdn:10.525 lr:6.6e-07 updt_s:16.329 data_s:0.335
INFO 2025-11-25 21:04:16 ot_train.py:351 step:100.0 smpl:17K ep:25 epch:0.29 loss:4.162 grdn:10.594 lr:1.9e-06 updt_s:16.254 data_s:0.075
INFO 2025-11-25 21:17:52 ot_train.py:351 step:150.0 smpl:25K ep:37 epch:0.43 loss:3.901 grdn:9.910 lr:3.2e-06 updt_s:16.254 data_s:0.074
INFO 2025-11-25 21:17:52 ot_train.py:351 step:150.0 smpl:25K ep:37 epch:0.43 loss:3.901 grdn:9.910 lr:3.2e-06 updt_s:16.254 data_s:0.074
INFO 2025-11-25 21:31:29 ot_train.py:351 step:200.0 smpl:34K ep:49 epch:0.58 loss:3.764 grdn:9.701 lr:4.4e-06 updt_s:16.257 data_s:0.075 --ckpg
INFO 2025-11-25 21:45:30 ot_train.py:351 step:250.0 smpl:42K ep:62 epch:0.72 loss:3.367 grdn:8.707 lr:5.7e-06 updt_s:16.256 data_s:0.075
INFO 2025-11-25 21:59:07 ot_train.py:351 step:300.0 smpl:50K ep:74 epch:0.87 loss:2.808 grdn:7.386 lr:6.9e-06 updt_s:16.258 data_s:0.073
INFO 2025-11-25 22:12:47 ot_train.py:351 step:350.0 smpl:59K ep:86 epch:1.01 loss:2.121 grdn:5.418 lr:8.2e-06 updt_s:16.057 data_s:0.340
INFO 2025-11-25 22:26:24 ot_train.py:351 step:400.0 smpl:67K ep:98 epch:1.16 loss:1.586 grdn:3.698 lr:9.4e-06 updt_s:16.258 data_s:0.074 --ckpt
INFO 2025-11-25 22:40:26 ot_train.py:351 step:450.0 smpl:76K ep:111 epch:1.30 loss:1.262 grdn:2.664 lr:1.1e-05 updt_s:16.253 data_s:0.076
INFO 2025-11-25 22:54:02 ot_train.py:351 step:500.0 smpl:84K ep:123 epch:1.45 loss:1.046 grdn:1.398 lr:1.2e-05 updt_s:16.255 data_s:0.074
INFO 2025-11-25 23:20:46 ot_train.py:351 step:450.0 smpl:76K ep:111 epch:1.30 loss:1.249 grdn:2.572 lr:1.1e-05 updt_s:16.359 data_s:0.275 --retry
INFO 2025-11-25 23:34:25 ot_train.py:351 step:500.0 smpl:84K ep:123 epch:1.45 loss:1.039 grdn:1.399 lr:1.2e-05 updt_s:16.289 data_s:0.075

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=/ssd1t/david/lerobot/datasets/davidlau90/lekiwi_toy_0123 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_0123 \
    --job_name=pi05_toy_0123 \
    --policy.repo_id=davidlau90/pi05_toy_0123 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=60000 \
    --log_fre=50 \
    --batch_size=168 \
    --save_freq=200 \
    --resume=true \
    --config_path=outputs/pi05_toy_0123/checkpoints/last/pretrained_model/train_config.json \
    --optimizer.lr 1.11e-4 
    # --policy.optimizer_lr 2.5e-04 \ # not work
    # --policy.optimizer_lr 2.5e-05 \ # orig 


INFO 2025-11-24 15:06:27 ot_train.py:351 step:200 smpl:6K ep:9 epch:0.11 loss:1.134 grdn:2.458 lr:2.5e-05 updt_s:3.293 data_s:0.015 <- batch size 32
INFO 2025-11-24 15:23:01 ot_train.py:351 step:500 smpl:16K ep:23 epch:0.28 loss:0.929 grdn:1.496 lr:2.4e-05 updt_s:3.292 data_s:0.016
INFO 2025-11-24 16:34:45 ot_train.py:351 step:2K smpl:58K ep:84 epch:0.99 loss:0.873 grdn:1.461 lr:1.1e-05 updt_s:3.291 data_s:0.016
INFO 2025-11-24 19:08:40 ot_train.py:351 step:3K smpl:109K ep:159 epch:1.88 loss:0.874 grdn:1.493 lr:2.4e-05 updt_s:3.319 data_s:0.016
INFO 2025-11-24 20:04:15 ot_train.py:351 step:4K smpl:141K ep:206 epch:2.43 loss:0.873 grdn:1.517 lr:2.4e-05 updt_s:3.316 data_s:0.015
INFO 2025-11-24 20:59:54 ot_train.py:351 step:5K smpl:173K ep:253 epch:2.98 loss:0.875 grdn:1.526 lr:2.3e-05 updt_s:3.316 data_s:0.016
INFO 2025-11-24 21:11:01 ot_train.py:351 step:6K smpl:179K ep:263 epch:3.09 loss:0.852 grdn:1.477 lr:2.3e-05 updt_s:3.317 data_s:0.016
INFO 2025-11-24 21:55:54 ot_train.py:351 step:6K smpl:205K ep:300 epch:3.53 loss:0.871 grdn:1.518 lr:2.3e-05 updt_s:3.317 data_s:0.016
INFO 2025-11-24 22:34:55 ot_train.py:351 step:6.2K smpl:397K ep:581 epch:6.84 loss:0.873 grdn:1.071 lr:2.3e-05 updt_s:6.374 data_s:0.057 <- batch size 168
INFO 2025-11-24 22:56:14 ot_train.py:351 step:6.4K smpl:410K ep:600 epch:7.06 loss:0.860 grdn:1.095 lr:2.3e-05 updt_s:6.360 data_s:0.030
INFO 2025-11-24 23:17:32 ot_train.py:351 step:6.6K smpl:422K ep:619 epch:7.28 loss:0.866 grdn:1.072 lr:2.2e-05 updt_s:6.360 data_s:0.029
INFO 2025-11-25 00:49:53 ot_train.py:351 step:6.2K smpl:1M ep:2K epch:17.95 loss:0.868 grdn:0.687 lr:2.3e-05 updt_s:16.245 data_s:0.120
INFO 2025-11-25 01:44:12 ot_train.py:351 step:6.4K smpl:1M ep:2K epch:18.53 loss:0.867 grdn:0.700 lr:2.3e-05 updt_s:16.175 data_s:0.117 
INFO 2025-11-25 02:39:54 ot_train.py:351 step:6.6K smpl:1M ep:2K epch:19.11 loss:0.865 grdn:0.698 lr:2.2e-05 updt_s:16.632 data_s:0.074 
```

## dataset hxdoso
```
screen
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=/ssd1t/david/lerobot/datasets/MrXuan/hxdoso \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_hxdoso \
    --job_name=pi05_training \
    --policy.repo_id=lerobot/pi05_base \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=128 \
    --log_freq=10 \

INFO 2025-11-25 18:29:10 ot_train.py:351 step:10.0 smpl:1K ep:3 epch:0.05 loss:0.287 grdn:3.301 lr:1.6e-06 updt_s:13.567 data_s:1.471
INFO 2025-11-25 18:31:18 ot_train.py:351 step:20.0 smpl:3K ep:5 epch:0.10 loss:0.248 grdn:2.559 lr:4.1e-06 updt_s:12.702 data_s:0.060
INFO 2025-11-25 18:33:25 ot_train.py:351 step:30.0 smpl:4K ep:8 epch:0.15 loss:0.195 grdn:1.244 lr:6.6e-06 updt_s:12.703 data_s:0.060
INFO 2025-11-25 18:35:33 ot_train.py:351 step:40.0 smpl:5K ep:10 epch:0.20 loss:0.166 grdn:0.850 lr:9.0e-06 updt_s:12.706 data_s:0.058
INFO 2025-11-25 18:37:41 ot_train.py:351 step:50.0 smpl:6K ep:13 epch:0.25 loss:0.141 grdn:0.861 lr:1.2e-05 updt_s:12.702 data_s:0.056
INFO 2025-11-25 18:39:48 ot_train.py:351 step:60.0 smpl:8K ep:15 epch:0.30 loss:0.110 grdn:0.788 lr:1.4e-05 updt_s:12.704 data_s:0.058
INFO 2025-11-25 18:41:56 ot_train.py:351 step:70.0 smpl:9K ep:18 epch:0.35 loss:0.103 grdn:0.934 lr:1.6e-05 updt_s:12.709 data_s:0.056
INFO 2025-11-25 18:44:04 ot_train.py:351 step:80.0 smpl:10K ep:20 epch:0.40 loss:0.094 grdn:0.923 lr:1.9e-05 updt_s:12.714 data_s:0.056
INFO 2025-11-25 18:46:11 ot_train.py:351 step:90.0 smpl:12K ep:23 epch:0.45 loss:0.084 grdn:0.931 lr:2.1e-05 updt_s:12.705 data_s:0.057
INFO 2025-11-25 18:48:19 ot_train.py:351 step:100.0 smpl:13K ep:25 epch:0.50 loss:0.076 grdn:0.889 lr:2.4e-05 updt_s:12.702 data_s:0.057
INFO 2025-11-25 18:50:27 ot_train.py:351 step:110.0 smpl:14K ep:28 epch:0.55 loss:0.070 grdn:0.829 lr:2.5e-05 updt_s:12.703 data_s:0.057
INFO 2025-11-25 18:52:34 ot_train.py:351 step:120.0 smpl:15K ep:30 epch:0.60 loss:0.067 grdn:0.785 lr:2.5e-05 updt_s:12.705 data_s:0.058
INFO 2025-11-25 18:54:42 ot_train.py:351 step:130.0 smpl:17K ep:33 epch:0.65 loss:0.064 grdn:0.851 lr:2.5e-05 updt_s:12.703 data_s:0.058
INFO 2025-11-25 18:56:50 ot_train.py:351 step:140.0 smpl:18K ep:35 epch:0.70 loss:0.060 grdn:0.734 lr:2.5e-05 updt_s:12.703 data_s:0.056
INFO 2025-11-25 18:58:57 ot_train.py:351 step:150.0 smpl:19K ep:38 epch:0.75 loss:0.058 grdn:0.721 lr:2.5e-05 updt_s:12.702 data_s:0.057
INFO 2025-11-25 19:01:05 ot_train.py:351 step:160.0 smpl:20K ep:40 epch:0.80 loss:0.055 grdn:0.668 lr:2.5e-05 updt_s:12.716 data_s:0.057
INFO 2025-11-25 19:03:13 ot_train.py:351 step:170.0 smpl:22K ep:43 epch:0.85 loss:0.056 grdn:0.622 lr:2.5e-05 updt_s:12.703 data_s:0.057
INFO 2025-11-25 19:05:20 ot_train.py:351 step:180.0 smpl:23K ep:45 epch:0.90 loss:0.055 grdn:0.642 lr:2.5e-05 updt_s:12.701 data_s:0.056
```


## dataset 457 (25)
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_457 \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_457 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_457 \
    --job_name=pi05_toy_457 \
    --policy.repo_id=davidlau90/pi05_toy_457 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --log_fre=50 \
    --batch_size=32

## dataset 7  good
INFO 2025-11-21 13:17:17 ot_train.py:351 step:200 smpl:6K ep:11 epch:1.07 loss:2.404 grdn:6.790 lr:1.9e-05 updt_s:3.305 data_s:0.056
INFO 2025-11-21 14:03:40 ot_train.py:351 step:100 smpl:3K ep:5 epch:0.53 loss:3.235 grdn:9.501 lr:1.9e-05 updt_s:5.944 data_s:0.038
INFO 2025-11-21 14:09:47 ot_train.py:351 step:150 smpl:5K ep:8 epch:0.80 loss:1.533 grdn:4.547 lr:2.5e-05 updt_s:7.297 data_s:0.040
INFO 2025-11-21 14:15:53 ot_train.py:351 step:200 smpl:6K ep:11 epch:1.07 loss:1.011 grdn:2.314 lr:2.5e-05 updt_s:7.225 data_s:0.092
INFO 2025-11-21 14:21:59 ot_train.py:351 step:250 smpl:8K ep:13 epch:1.33 loss:0.873 grdn:1.496 lr:2.5e-05 updt_s:7.281 data_s:0.040
INFO 2025-11-21 14:28:05 ot_train.py:351 step:300 smpl:10K ep:16 epch:1.60 loss:0.852 grdn:1.321 lr:2.5e-05 updt_s:7.279 data_s:0.040
INFO 2025-11-21 14:34:12 ot_train.py:351 step:350 smpl:11K ep:19 epch:1.87 loss:0.859 grdn:1.492 lr:2.4e-05 updt_s:7.283 data_s:0.040
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_7 \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_7 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_7 \
    --job_name=pi05_toy_7 \
    --policy.repo_id=davidlau90/pi05_toy_7 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --log_fre=50 \
    --batch_size=32


## dataset 5  good
INFO 2025-11-21 14:47:06 ot_train.py:351 step:50 smpl:2K ep:2 epch:0.36 loss:3.888 grdn:11.134 lr:6.6e-06 updt_s:7.320 data_s:0.109
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_5 \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_5 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_5 \
    --job_name=pi05_toy_5 \
    --policy.repo_id=davidlau90/pi05_toy_5 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --log_fre=50 \
    --batch_size=32

## dataset 6 bad: 
INFO 2025-11-21 14:24:09 ot_train.py:351 step:200 smpl:6K ep:5 epch:0.16 loss:1741504370847283937280.000 grdn:10679317590.819 lr:1.9e-05 updt_s:7.291 data_s:0.058
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_6 \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_6 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_6 \
    --job_name=pi05_toy_6 \
    --policy.repo_id=davidlau90/pi05_toy_6 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --batch_size=32

## dataset 4 good: INFO 2025-11-21 13:38:11 ot_train.py:351 step:200 smpl:6K ep:6 epch:0.10 loss:509752646492880764928.000 grdn:2301886843.971 lr:1.9e-05 updt_s:4.771 data_s:0.044
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_4 \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_4 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_4 \
    --job_name=pi05_toy_4 \
    --policy.repo_id=davidlau90/pi05_toy_4 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --batch_size=32

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_simple \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_simple \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_toy_simple \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/lekiwi_pi05_toy_simple \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --batch_size=32

python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_1 \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_1 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training_1 \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/lekiwi_pi05_1 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --batch_size=32

rm -rf outputs/pi05_training_2 && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.root=datasets/davidlau90/lekiwi_toy_pickup_2 \
    --dataset.repo_id=davidlau90/lekiwi_toy_pickup_2 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training_2 \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/lekiwi_pi05_2 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --batch_size=32



AutoDL A800 96G:
- A800 03区无科学网络 / 数据盘30G优点不够装 1个CKPT 20G
- pi05_base batch size 3 OK
    - outputs/pi05_training/checkpoints000002/training_state/optimizer_state.safetensors 13G
    - outputs/pi05_training/checkpoints000002/pretrained_model/model.safetensors 7G
    - Connection to huggingface.co timed out. (connect timeout=None)’) 
AutoDL A100 32G: OOM
- pi05_base batch size 32 OOM
- pi05_base batch size 8 OOM 
- pi0_base batch size 8 OOM

AutoDL H20 96G: Too Slow
- pi05_base batch size 32 (GPU RAM:51.9GB; RAM 17GB)





# Clash Proxy (AutoDL NA）
git clone --branch master --depth 1 https://gh-proxy.com/https://github.com/nelvko/clash-for-linux-install.git 
cd clash-for-linux-install
sudo bash install.sh # https://45.137.181.44/link/2b3j4LBfzz25djGQ?clash=1


