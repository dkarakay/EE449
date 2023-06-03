#
# Created on Thu Jun 01 2023
#
# Deniz Karakay 2443307
#
# EE449 HW3
#


import gym_super_mario_bros
import numpy as np

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt

from utils import SaveOnBestTrainingRewardCallback, startGameModel, startGameRand

def create_environment(type,checkpoint):
    #folder_name = f"./video/{name}/{uuid.uuid4()}"
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if(type == 0):
      env = GrayScaleObservation(env, keep_dim=True)
      env = DummyVecEnv([lambda: env])
      env = VecFrameStack(env, n_stack=4, channels_order="last")
      env = VecMonitor(env, f"{checkpoint}/TestMonitor")    
    elif (type == 1):
      env = GrayScaleObservation(env, keep_dim=True)
      env = DummyVecEnv([lambda: env])
      env = VecFrameStack(env, n_stack=4, channels_order="last")
      env = VecMonitor(env, f"{checkpoint}/TestMonitor")    
    return env

CHECKPOINT_DIR_PPO1 = "./train/PPO1/"
CHECKPOINT_DIR_PPO2 = "./train/PPO2/"
CHECKPOINT_DIR_DQN1 = "./train/DQN1/"
LOG_DIR = "./logs/"

env_ppo2 = create_environment(1,CHECKPOINT_DIR_PPO2)
callback_ppo2 = SaveOnBestTrainingRewardCallback(save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_PPO2)

# Learning rate 0.0001
model_ppo2 = PPO('CnnPolicy', env_ppo2, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=512)
model_ppo2.learn(total_timesteps=1000000, callback=callback_ppo2, tb_log_name="PPO2")
