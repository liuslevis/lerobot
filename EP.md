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



# 配置测试摄像头 (host)
* cd ~/lerobot/tests && python test_cam.py
* vim src/lerobot/robots/lekiwi/config_lekiwi.py

# Calibration
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5AB01813381 --teleop.id=di
lerobot-calibrate --robot.type=lekiwi --robot.id=didi

# 启动 leader arm
python examples/lekiwi/teleoperate.py 

# 启动 host
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=didi --host.connection_time_s=3600 


# Dataset Record 
python -i examples/lekiwi/record.py

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




# Async Policy Server Start
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=9999

# Async Client Start
python -m lerobot.async_inference.robot_client \
    --robot.type=lekiwi \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\"
, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\"
, width: 640, height: 480, fps: 30}}" \
    --robot.id=didi \
    --task="pick up toys\n" \
    --server_address=192.168.0.78:9999 \
    --policy_type=pi05 \
    --pretrained_name_or_path=/ssd1t/david/lerobot/outputs/pi05_toy_457/checkpoints/003000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True


# BitaHub A100: 
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
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -c conda-forge ffmpeg=6.1.1 -y &

pip install -e ".[lekiwi,pi]"
git config --global credential.helper store
export HUGGINGFACE_TOKEN=
export HF_HOME=/ssd1t/david/huggingface
export HF_HOME_HUB=/ssd1t/david/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
wandb login --relogin 
# hf download lerobot/pi05_base 
# hf download google/paligemma-3b-pt-224 --repo-type model

# Train: π0.5 A100
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/ssd1t/david/huggingface
export HF_HOME_HUB=/ssd1t/david/huggingface/hub


# dataset 457 (25)
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

# dataset 7  good
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


# dataset 5  good
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

# dataset 6 bad: 
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

# dataset 4 good: INFO 2025-11-21 13:38:11 ot_train.py:351 step:200 smpl:6K ep:6 epch:0.10 loss:509752646492880764928.000 grdn:2301886843.971 lr:1.9e-05 updt_s:4.771 data_s:0.044
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





# Clash Proxy (AutoDL 不能用）
git clone --branch master --depth 1 https://gh-proxy.com/https://github.com/nelvko/clash-for-linux-install.git 
cd clash-for-linux-install
sudo bash install.sh # https://45.137.181.44/link/2b3j4LBfzz25djGQ?clash=1


