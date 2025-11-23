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