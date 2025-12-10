
# TODO
[] Ubuntu Record / Inference video Play Slow, maybe try cuda decode https://github.com/huggingface/lerobot/pull/913/files
[] Learn 1 Grab -> 2 Placement

# FAQ

Q: Why huge loss and grad?
A: brasket_1 have longer time and fewer episodes → diffusion/action-chunking may get misaligned → incorrect temporal condition → model diverges?

Q:
A:

# RP5 Sync
```
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=didi --host.connection_time_s=36000 --robot.cameras="{ front: {type: opencv, index_or_path: \"/dev/video0\" , width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: \"/dev/video2\", width: 640, height: 480, fps: 30}}"
```

# PC Sync Evaluate
```
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python examples/lekiwi/evaluate.py 
```

# LeKiwi Setup
```
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
```

# Test Camera (RP5)
```
cd ~/lerobot/tests && python test_cam.py
vim src/lerobot/robots/lekiwi/config_lekiwi.py
```

# Calibration
```
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5AB01813381 --teleop.id=di
lerobot-calibrate --robot.type=lekiwi --robot.id=didi --robot.cameras '{front: {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}, wrist: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30} }' 
# /Users/david/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/di.json # Mac
# /root/.cache/huggingface/lerobot/calibration/robots/lekiwi/didi.json # RP5
```

# Start Tele Op (PC leader arm)
```
python examples/lekiwi/teleoperate.py 
```

# Start Host (RP)
```
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=didi --host.connection_time_s=36000 --robot.cameras='{front: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "rotation": 180},wrist: {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30} }'
```

# Dataset Record 
```
python -i examples/lekiwi/record_toy.py
python -i examples/lekiwi/record_basket.py
```

- Right arrow key pressed. Exiting loop...
- Left arrow key pressed. Exiting loop and rerecord the last episode
- Escape key pressed. Stopping data recording
- F: slower
- R: faster
- ASDW: move
- Z X: rotate left right
- Q: quit tele op

# Dataset Upload 
```
hf upload davidlau90/toy_pickup ~/.cache/huggingface/lerobot/davidlau90/toy_pickup --repo-type dataset
```

训练数据路径：
```
davidlau90/lekiwi_toy_pickup_1 # 10 次 移动抓放
davidlau90/lekiwi_toy_pickup # 20 次 固定位置抓放后移动
davidlau90/lekiwi_toy_pickup_simple # 单一物品固定抓放，不移动
davidlau90/grab_toy_1 # 抓小车 50次 不移动
```


# Replay Recording
```
python examples/lekiwi/replay.py
```

# Dataset
## Visualize Dataset (HF -> rdd -> rerun.io)
```
lerobot-dataset-viz --repo-id /ssd1t/david/datasets/davidlau90/basket_1 --episode-index 0 --num-workers 16 # --save 1 --output-dir /ssd1t/david/datasets/viz_rdd
```

## Split Dataset
```
lerobot-edit-dataset \
    --repo_id /ssd1t/david/datasets/davidlau90/basket_1_bak \
    --operation.type split \
    --operation.splits '{"train": 0.5, "test": 0.5}'

```

# PC 5080 sm120 Support
```
pip uninstall torch torchcodec torchvision
pip install torch==2.7	torchcodec==0.5 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
# LoRA / Frozen Param Support

| Frozen Param    | Learnable / Total Param | Used / Total GPU RAM |
|-----------------|-------------------------|----------------------|
| NA              | 3.6B / 3.6B             | 30GB / 80GB          |
| Vision+LM+Lora8 | 467M / 3.6B             | 10.7GB / 16.3GB      |
|*Vision+LM       | 696M / 3.6B             | 12.3GB / 16.3GB      |
| Vision          | 1.1B / 3.6B             | 15.7GB / 16.3GB      |
| LM              | 3.2B / 3.6B             | OOM /16.3GB          |

Note: batch_size=1

## Downlaod Dataset on PC
```
cd /ssd1t/david/datasets
REPO_ID=basket_1
hf download davidlau90/${REPO_ID} --repo-type dataset --local-dir davidlau90/${REPO_ID}
```

## Train on 5080 
```
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/ssd1t/david/huggingface
export HF_HOME_HUB=/ssd1t/david/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com


export VARIANT=pi0-basket-peft
export REPO_ID=davidlau90/basket_12345
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=${REPO_ID} \
    --dataset.root=/ssd1t/david/datasets/${REPO_ID} \
    --wandb.enable=false \
    --job_name=pi0_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi0 \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --steps=30000 \
    --log_freq=200 \
    --eval_freq=1000 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=10 \
> outputs/logs/${VARIANT}.txt 2>&1
tail -f outputs/logs/${VARIANT}.txt


# train basket pi05 after grab
export VARIANT=pi05-grab-basket-peft
export REPO_ID=davidlau90/basket_12345
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=${REPO_ID} \
    --dataset.root=/ssd1t/david/datasets/${REPO_ID} \
    --wandb.enable=false \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --optimizer.grad_clip_norm=1.0 \
    --steps=300000 \
    --log_fre=200 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=15 \
    --resume=true \
    --config_path=outputs/pi05-grab-peft/checkpoints/last/pretrained_model/train_config.json \
> outputs/logs/${VARIANT}.txt 2>&1 
tail -f outputs/logs/${VARIANT}.txt

# train basket pi05
export VARIANT=pi05-basket-peft
export REPO_ID=davidlau90/basket_12345
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=${REPO_ID} \
    --dataset.root=/ssd1t/david/datasets/${REPO_ID} \
    --wandb.enable=false \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --steps=15000 \
    --log_freq=200 \
    --eval_freq=1000 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=15 \
> outputs/logs/${VARIANT}.txt 2>&1
tail -f outputs/logs/${VARIANT}.txt



# train grab & pick together
export VARIANT=pi05-grab-and-pick-peft
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/grab_and_pickup \
    --dataset.root=/ssd1t/david/datasets/davidlau90/grab_and_pickup \
    --wandb.enable=false \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --steps=27000 \
    --log_fre=500 \
    --eval_freq=1000 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=15 \
> outputs/logs/${VARIANT}.txt 2>&1

# train pick on top of grab 
export VARIANT=pi05-grab-pick-peft
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/pickup_toy_457 \
    --dataset.root=/ssd1t/david/datasets/davidlau90/pickup_toy_457 \
    --wandb.enable=false \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --steps=27000 \
    --log_fre=500 \
    --eval_freq=1000 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=15 \
    --resume=true \
    --config_path=outputs/pi05-grab-peft/checkpoints/last/pretrained_model/train_config.json \
> outputs/logs/${VARIANT}.txt 2>&1
step:12.0K smpl:180K ep:199 epch:7.95 loss:0.035 grdn:0.494 lr:1.6e-05 updt_s:2.445 data_s:0.007
step:15.0K smpl:225K ep:248 epch:9.93 loss:0.027 grdn:0.466 lr:1.2e-05 updt_s:2.445 data_s:0.007
step:18.0K smpl:270K ep:298 epch:11.92 loss:0.023 grdn:0.444 lr:8.4e-06 updt_s:2.445 data_s:0.007
step:21.0K smpl:315K ep:348 epch:13.91 loss:0.021 grdn:0.468 lr:5.3e-06 updt_s:2.445 data_s:0.007
step:24.0K smpl:360K ep:397 epch:15.89 loss:0.020 grdn:0.460 lr:3.3e-06 updt_s:2.445 data_s:0.007
step:27.0K smpl:405K ep:447 epch:17.88 loss:0.019 grdn:0.467 lr:2.5e-06 updt_s:2.478 data_s:0.007

export VARIANT=pi05-pick-peft
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/pickup_toy_457 \
    --dataset.root=/ssd1t/david/datasets/davidlau90/pickup_toy_457 \
    --wandb.enable=true \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --steps=9000 \
    --log_fre=500 \
    --eval_freq=1000 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=15 \
> outputs/logs/${VARIANT}.txt 2>&1
INFO 01:45:16 step:500.0 smpl:8K ep:8 epch:0.33 loss:0.153 grdn:0.972 lr:1.8e-05 updt_s:2.452 data_s:0.012
INFO 02:05:45 step:1.0K smpl:15K ep:17 epch:0.66 loss:0.060 grdn:0.527 lr:2.5e-05 updt_s:2.447 data_s:0.011
INFO 02:26:14 step:1.5K smpl:22K ep:25 epch:0.99 loss:0.052 grdn:0.529 lr:2.4e-05 updt_s:2.446 data_s:0.011
INFO 02:46:43 step:2.0K smpl:30K ep:33 epch:1.32 loss:0.045 grdn:0.497 lr:2.3e-05 updt_s:2.446 data_s:0.012
INFO 03:07:11 step:2.5K smpl:38K ep:41 epch:1.66 loss:0.040 grdn:0.497 lr:2.2e-05 updt_s:2.446 data_s:0.011
INFO 03:27:49 step:3.0K smpl:45K ep:50 epch:1.99 loss:0.037 grdn:0.499 lr:2.0e-05 updt_s:2.463 data_s:0.011

export VARIANT=pi05-grab-peft
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/grab_toy_1 \
    --wandb.enable=true \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=false \
    --steps=9000 \
    --log_fre=500 \
    --eval_freq=1000 \
    --eval_freq=1000 \
    --save_freq=3000 \
    --batch_size=10 \
> outputs/logs/${VARIANT}.txt 2>&1
INFO 2025-11-30 20:09:19 ot_train.py:351 step:500.0 smpl:5K ep:11 epch:0.22 loss:0.195 grdn:1.341 lr:1.8e-05 updt_s:1.690 data_s:0.011
INFO 2025-11-30 20:23:23 ot_train.py:351 step:1.0K smpl:10K ep:22 epch:0.45 loss:0.080 grdn:0.810 lr:2.5e-05 updt_s:1.680 data_s:0.008
INFO 2025-11-30 20:37:27 ot_train.py:351 step:1.5K smpl:15K ep:33 epch:0.67 loss:0.071 grdn:0.834 lr:2.4e-05 updt_s:1.681 data_s:0.008
INFO 2025-11-30 20:51:31 ot_train.py:351 step:2.0K smpl:20K ep:45 epch:0.89 loss:0.061 grdn:0.741 lr:2.3e-05 updt_s:1.680 data_s:0.008
INFO 2025-11-30 21:05:35 ot_train.py:351 step:2.5K smpl:25K ep:56 epch:1.11 loss:0.057 grdn:0.747 lr:2.2e-05 updt_s:1.677 data_s:0.009
INFO 2025-11-30 21:19:40 ot_train.py:351 step:3.0K smpl:30K ep:67 epch:1.34 loss:0.056 grdn:0.763 lr:2.0e-05 updt_s:1.682 data_s:0.008
INFO 2025-11-30 21:37:19 ot_train.py:351 step:3.5K smpl:35K ep:78 epch:1.56 loss:0.051 grdn:0.769 lr:1.8e-05 updt_s:1.689 data_s:0.008
INFO 2025-11-30 21:51:24 ot_train.py:351 step:4.0K smpl:40K ep:89 epch:1.78 loss:0.049 grdn:0.751 lr:1.7e-05 updt_s:1.681 data_s:0.008
INFO 2025-11-30 22:05:28 ot_train.py:351 step:4.5K smpl:45K ep:100 epch:2.00 loss:0.045 grdn:0.735 lr:1.5e-05 updt_s:1.679 data_s:0.009
INFO 2025-11-30 22:19:32 ot_train.py:351 step:5.0K smpl:50K ep:111 epch:2.23 loss:0.043 grdn:0.725 lr:1.3e-05 updt_s:1.680 data_s:0.008
INFO 2025-11-30 22:33:36 ot_train.py:351 step:5.5K smpl:55K ep:122 epch:2.45 loss:0.041 grdn:0.737 lr:1.1e-05 updt_s:1.680 data_s:0.008
INFO 2025-11-30 22:47:40 ot_train.py:351 step:6.0K smpl:60K ep:134 epch:2.67 loss:0.041 grdn:0.776 lr:9.0e-06 updt_s:1.680 data_s:0.008
INFO 2025-11-30 23:04:42 ot_train.py:351 step:6.5K smpl:65K ep:145 epch:2.90 loss:0.040 grdn:0.736 lr:7.3e-06 updt_s:1.680 data_s:0.008
INFO 2025-11-30 23:18:46 ot_train.py:351 step:7.0K smpl:70K ep:156 epch:3.12 loss:0.038 grdn:0.742 lr:5.8e-06 updt_s:1.678 data_s:0.008
INFO 2025-11-30 23:32:50 ot_train.py:351 step:7.5K smpl:75K ep:167 epch:3.34 loss:0.036 grdn:0.728 lr:4.5e-06 updt_s:1.680 data_s:0.008
INFO 2025-11-30 23:46:54 ot_train.py:351 step:8.0K smpl:80K ep:178 epch:3.56 loss:0.036 grdn:0.728 lr:3.6e-06 updt_s:1.680 data_s:0.008
INFO 2025-12-01 00:00:59 ot_train.py:351 step:8.5K smpl:85K ep:189 epch:3.79 loss:0.037 grdn:0.743 lr:2.9e-06 updt_s:1.680 data_s:0.008
INFO 2025-12-01 00:15:02 ot_train.py:351 step:9.0K smpl:90K ep:200 epch:4.01 loss:0.036 grdn:0.727 lr:2.6e-06 updt_s:1.677 data_s:0.008


export VARIANT=grab-lora
rm -rf outputs/${VARIANT} && \
python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=davidlau90/grab_toy_1 \
    --wandb.enable=true \
    --job_name=pi05_training \
    --output_dir=outputs/${VARIANT} \
    --policy.repo_id=davidlau90/${VARIANT} \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_model=true \
    --policy.use_lora=true \
    --policy.lora_rank=8 \
    --steps=3000 \
    --log_fre=200 \
    --eval_fre=200 \
    --eval_freq=1000 \
    --batch_size=10 \
> outputs/logs/${VARIANT}.txt 2>&1
step:3.0K smpl:30K ep:33 epch:1.32 loss:0.093 grdn:3.042 lr:2.6e-06 updt_s:1.682 data_s:0.008

Note: LoRA loss is double

```

# Async Policy Server Start PC
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=9999

# Async Client Start RP5
```
export PROMPT="pickup the toy\n"
export CKPT=/ssd1t/david/lerobot/outputs/pi05_toy_457/checkpoints/003000/pretrained_model
export CKPT=/ssd1t/david/lerobot/outputs/pi05_toy_0123_again/checkpoints/000200/pretrained_model # grab robot itself
export CKPT=/ssd1t/david/lerobot/outputs/pi05_toy_0123_again/checkpoints/001000/pretrained_model # movement ok, but cannot grab
export CKPT=/ssd1t/david/lerobot/outputs/pi05_grab_1/checkpoints/001200/pretrained_model # still cannot grab
export PROMPT="grab the toy\n"

export MODEL=pi05-grab-peft 
export PROMPT="grab the toy\n" # succ grab

export MODEL=pi05-pick-peft 
export PROMPT="pick the toy\n" # can't grab

export MODEL=pi05-grab-and-pick-peft 
export PROMPT="pickup the toy\n" # try to grab but failed


export MODEL="pi05-grab-pick-peft"
export PROMPT="grab the toy\n" # barely move
export PROMPT="pickup the toy\n" # continous try to grab. succ grab and pickup after 5 tries. failed to place. 

export MODEL="pi05-basket-peft"
export PROMPT="grab the toy and put it into basket\n" # failed to catch. have put to basket action

export MODEL="pi05-grab-basket-peft"
export PROMPT="grab the toy and put it into basket\n" # barely move

export CKPT=/ssd1t/david/lerobot/outputs/${MODEL}/checkpoints/last/pretrained_model 
export ACT_PER_CHUNK=50
python -m lerobot.async_inference.robot_client \
    --robot.type=lekiwi \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{front: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "rotation": 180},wrist: {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30} }' \
    --robot.id=didi \
    --task="${PROMPT}" \
    --server_address=192.168.0.78:9999 \
    --policy_type=pi05 \
    --pretrained_name_or_path=${CKPT} \
    --policy_device=cuda \
    --actions_per_chunk=${ACT_PER_CHUNK} \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

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
INFO 2025-11-27 01:49:30 ot_train.py:351 step:50.0 smpl:8K ep:19 epch:0.37 loss:4.726 grdn:11.636 lr:6.6e-07 updt_s:16.322 data_s:0.281
INFO 2025-11-27 02:03:07 ot_train.py:351 step:100.0 smpl:17K ep:37 epch:0.75 loss:4.775 grdn:11.747 lr:1.9e-06 updt_s:16.257 data_s:0.075
INFO 2025-11-27 02:17:55 ot_train.py:351 step:150.0 smpl:25K ep:56 epch:1.12 loss:4.498 grdn:11.069 lr:3.2e-06 updt_s:16.153 data_s:0.256
INFO 2025-11-27 02:31:31 ot_train.py:351 step:200.0 smpl:34K ep:75 epch:1.50 loss:4.321 grdn:10.657 lr:4.4e-06 updt_s:16.251 data_s:0.075
INFO 2025-11-27 02:46:04 ot_train.py:351 step:250.0 smpl:42K ep:94 epch:1.87 loss:3.863 grdn:9.582 lr:5.7e-06 updt_s:16.250 data_s:0.075
INFO 2025-11-27 02:59:44 ot_train.py:351 step:300.0 smpl:50K ep:112 epch:2.24 loss:3.312 grdn:8.300 lr:6.9e-06 updt_s:16.135 data_s:0.248
INFO 2025-11-27 03:14:21 ot_train.py:351 step:350.0 smpl:59K ep:131 epch:2.62 loss:2.539 grdn:6.252 lr:8.2e-06 updt_s:16.246 data_s:0.075
INFO 2025-11-27 03:27:57 ot_train.py:351 step:400.0 smpl:67K ep:150 epch:2.99 loss:1.888 grdn:4.286 lr:9.4e-06 updt_s:16.250 data_s:0.074
INFO 2025-11-27 03:42:41 ot_train.py:351 step:450.0 smpl:76K ep:168 epch:3.37 loss:1.494 grdn:3.143 lr:1.1e-05 updt_s:16.134 data_s:0.276
INFO 2025-11-27 03:56:17 ot_train.py:351 step:500.0 smpl:84K ep:187 epch:3.74 loss:1.197 grdn:1.613 lr:1.2e-05 updt_s:16.252 data_s:0.074
INFO 2025-11-27 04:10:56 ot_train.py:351 step:550.0 smpl:92K ep:206 epch:4.12 loss:1.096 grdn:0.964 lr:1.3e-05 updt_s:16.136 data_s:0.254
INFO 2025-11-27 04:24:32 ot_train.py:351 step:600.0 smpl:101K ep:224 epch:4.49 loss:1.055 grdn:0.788 lr:1.4e-05 updt_s:16.255 data_s:0.074
INFO 2025-11-27 04:39:00 ot_train.py:351 step:650.0 smpl:109K ep:243 epch:4.86 loss:1.040 grdn:0.747 lr:1.6e-05 updt_s:16.254 data_s:0.073
INFO 2025-11-27 04:52:38 ot_train.py:351 step:700.0 smpl:118K ep:262 epch:5.24 loss:1.029 grdn:0.779 lr:1.7e-05 updt_s:16.136 data_s:0.221
INFO 2025-11-27 05:07:06 ot_train.py:351 step:750.0 smpl:126K ep:281 epch:5.61 loss:1.020 grdn:0.737 lr:1.8e-05 updt_s:16.249 data_s:0.075
INFO 2025-11-27 05:20:43 ot_train.py:351 step:800.0 smpl:134K ep:299 epch:5.99 loss:1.005 grdn:0.764 lr:1.9e-05 updt_s:16.258 data_s:0.074
INFO 2025-11-27 05:35:21 ot_train.py:351 step:850.0 smpl:143K ep:318 epch:6.36 loss:0.999 grdn:0.783 lr:2.1e-05 updt_s:16.136 data_s:0.246
INFO 2025-11-27 05:48:57 ot_train.py:351 step:900.0 smpl:151K ep:337 epch:6.73 loss:1.002 grdn:0.788 lr:2.2e-05 updt_s:16.255 data_s:0.074
INFO 2025-11-27 06:03:34 ot_train.py:351 step:950.0 smpl:160K ep:355 epch:7.11 loss:1.004 grdn:0.809 lr:2.3e-05 updt_s:16.138 data_s:0.302
INFO 2025-11-27 06:17:11 ot_train.py:351 step:1.0K smpl:168K ep:374 epch:7.48 loss:0.991 grdn:0.809 lr:2.4e-05 updt_s:16.255 data_s:0.075
INFO 2025-11-27 06:31:44 ot_train.py:351 step:1.1K smpl:176K ep:393 epch:7.86 loss:1.001 grdn:0.845 lr:2.5e-05 updt_s:16.253 data_s:0.074
INFO 2025-11-27 06:45:24 ot_train.py:351 step:1.1K smpl:185K ep:412 epch:8.23 loss:0.992 grdn:0.779 lr:2.5e-05 updt_s:16.141 data_s:0.269
INFO 2025-11-27 06:59:58 ot_train.py:351 step:1.1K smpl:193K ep:430 epch:8.61 loss:0.994 grdn:0.828 lr:2.5e-05 updt_s:16.260 data_s:0.073
INFO 2025-11-27 07:13:35 ot_train.py:351 step:1.2K smpl:202K ep:449 epch:8.98 loss:0.994 grdn:0.807 lr:2.5e-05 updt_s:16.260 data_s:0.072
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
    --dataset.repo_id=/ssd1t/david/lerobot/datasets/davidlau90/lekiwi_toy_pickup_1 \
    --policy.type=pi05 \
    --output_dir=./outputs/test \
    --job_name=pi05_training \
    --policy.repo_id=davidlau90/test \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=3000 \
    --batch_size=1

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


