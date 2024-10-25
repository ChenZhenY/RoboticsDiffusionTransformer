import zmq
import time
import numpy as np

#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
# import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PImage
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import cv2

from scripts.agilex_model import create_model, create_model_franka
# from rdt.scripts.agilex_model import create_model, create_model_franka

# sys.path.append("./")

CAMERA_NAMES = ['agentview_image']

observation_window = None

lang_embeddings = None

# debug
preload_images = None


# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model_franka(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 14,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Apply image transform for the images, and adapt to the images inputs
def update_observation_window(args, config, observation):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None
                    },
            }
        )
    
    images = observation["images"]
    assert len(images)==2, f"Expected 2 images, got {len(images)}"
    img_ext_t0, img_ext_t1 = np.array(images[0]), np.array(images[1])
    img_ext_t0, img_ext_t1 = jpeg_mapping(img_ext_t0), jpeg_mapping(img_ext_t1)

    images_arr = [img_ext_t0, img_ext_t1]
    qpos_arr = observation["qpos"]
    gripper_qpos_arr = observation["gripper_qpos"]

    for qpos, gripper_pos, img in zip(qpos_arr, gripper_qpos_arr, images_arr):
        qpos_gripper = np.array(qpos + gripper_pos) # from list to np array
        assert qpos_gripper.shape == (8,), f"Expected qpos_gripper shape (8,), got {qpos_gripper.shape}"
        qpos_gripper = torch.from_numpy(qpos_gripper).float().cuda()
        observation_window.append(
        {
            'qpos': qpos_gripper,
            'images':
                {
                    config["camera_names"][0]: img,
                },
        }
    )


# RDT inference
def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings
    
    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()     

        # fetch images in sequence [front, right, left]
        img_ext_t0, img_ext_t1 = observation_window[-2]['images'][config['camera_names'][0]], observation_window[-1]['images'][config['camera_names'][0]]
        # background_ext_t0, background_ext_t1 = np.mean(img_ext_t0)*np.ones_like(img_ext_t0), np.mean(img_ext_t1)*np.ones_like(img_ext_t1)
        # NOTE: background is handled in RDT.step(). Need to add background to the images to follow the training data format
        image_arrs = [
            img_ext_t0,
            None, # background_ext_t0,
            None, # background_ext_t0,

            img_ext_t1,
            None, # background_ext_t1,
            None, # background_ext_t1,
        ]
        
        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()
        
        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')
        
        # get last qpos in shape [14, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 14]
        proprio = proprio.unsqueeze(0)
        
        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")
        
        print(f"Model inference time: {time.time() - time1} s")
        
        # print(f"Finish inference_thread_fn: t={t}")
        return actions


# Main loop for the manipulation task
def model_inference(args, config):
    global lang_embeddings

    # Initialize ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP: Reply pattern
    socket.connect("tcp://127.0.0.1:5555")  # Connect to the same address as the server
    
    # Load rdt model
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initialize position of the puppet arm
    input("Initialization complete. Press enter to continue")

    action = None
    # Inference loop: current send out all action chunk at a time, and receive number observation
    # TODO: add two threads for collecting the observation and sending actiosn separately, like ROS
    # TODO: add interpolate like before
    with torch.inference_mode():
        # The current time step
        t = 0
        while True:    
            action_buffer = np.zeros([chunk_size, config['state_dim']])

            observation = socket.recv_json()
            
            # Update observation window
            update_observation_window(args, config, observation)
            
            # When coming to the end of the action chunk
            action_buffer = inference_fn(args, config, policy, t).copy() # np array [64, 8] for Franka
            
            # raw_action = action_buffer[t % chunk_size]
            # action = raw_action
            # # Interpolate the original action sequence
            # if args.use_actions_interpolation:
            #     # print(f"Time {t}, pre {pre_action}, act {action}")
            #     interp_actions = interpolate_action(args, pre_action, action)
            # else:
            #     interp_actions = action[np.newaxis, :]
            # # Execute the interpolated actions one by one
            # for act in interp_actions:
            #     left_action = act[:7]
            #     right_action = act[7:14]
                
            #     if not args.disable_puppet_arm:
            #         ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
            
            #     if args.use_robot_base:
            #         vel_action = act[14:16]
            #         ros_operator.robot_base_publish(vel_action)
            #     rate.sleep()
            #     # print(f"doing action: {act}")
            
            t += 1

            socket.send_json({"action": action_buffer.tolist()})
            
            print("Published Step", t)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)
    
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=20, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    model_inference(args, config)

""" Sample commnad
python ./scripts/franka_sim_inference.py --pretrained_model_name_or_path /home/mbronars/zhenyang/pretrained/rdt-1b --lang_embeddings_path /home/mbronars/zhenyang/RoboticsDiffusionTransformer/lang_encode/pickup_bowl.pt
"""
if __name__ == '__main__':
    main()
    
