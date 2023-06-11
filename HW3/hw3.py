#
# Created on Thu Jun 01 2023
#
# Deniz Karakay 2443307
#
# EE449 HW3 - Testing
#


import gym_super_mario_bros
import numpy as np

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt

from utils import startGameModel, startGameRand, saveGameModel


def create_environment(type, checkpoint):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if type == 0:
        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4, channels_order="last")
        env = VecMonitor(env, f"{checkpoint}/TestMonitor")
    elif type == 1:
        env = GrayScaleObservation(env, keep_dim=False)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=2)
        env = VecMonitor(env, f"{checkpoint}/TestMonitor")
    elif type == 2:
        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=2, channels_order="first")
        env = VecMonitor(env, f"{checkpoint}/TestMonitor")
    return env


CHECKPOINT_DIR_PPO1 = "./train/PPO1/"
CHECKPOINT_DIR_PPO2 = "./train/PPO2/"
CHECKPOINT_DIR_PPO3 = "./train/PPO3/"
CHECKPOINT_DIR_DQN1 = "./train/DQN1/"
CHECKPOINT_DIR_DQN2 = "./train/DQN2_MLP_2/"
CHECKPOINT_DIR_DQN3 = "./train/DQN2_2/"
CHECKPOINT_DIR_DQN_BEST = "./train/DQN_BEST/"
CHECKPOINT_DIR_PPO_BEST = "./train/PPO_BEST/"

steps = [10000, 100000, 500000, 1000000]

# For testing the best models of PPO and DQN
# Print the scores of the best models
# I modified utils.py to save the scores of the best models

# Fpr PPO1
for s in steps:
    print(f"PPO1 - {s}")
    env = create_environment(0, CHECKPOINT_DIR_PPO1)
    model = PPO.load(f"{CHECKPOINT_DIR_PPO1}/models/iter_{s}")
    if s == 10000 or s == 100000:
        startGameModel(env, model, step_len=2500)
    else:
        startGameModel(env, model, step_len=10000)

# For PPO2
for s in steps:
    print(f"PPO2 - {s}")
    env = create_environment(1, CHECKPOINT_DIR_PPO2)
    model = PPO.load(f"{CHECKPOINT_DIR_PPO2}/models/iter_{s}")
    if s == 10000 or s == 100000:
        startGameModel(env, model, step_len=2500)
    else:
        startGameModel(env, model, step_len=10000)

# For PPO3
for s in steps:
    print(f"PPO3 - {s}")
    env = create_environment(2, CHECKPOINT_DIR_PPO3)
    model = PPO.load(f"{CHECKPOINT_DIR_PPO3}/models/iter_{s}")
    if s == 10000 or s == 100000:
        startGameModel(env, model, step_len=2500)
    else:
        startGameModel(env, model, step_len=10000)

# For DQN1
for s in steps:
    print(f"DQN1 - {s}")
    env = create_environment(0, CHECKPOINT_DIR_DQN1)
    model = DQN.load(f"{CHECKPOINT_DIR_DQN1}/models/iter_{s}")
    if s == 10000 or s == 100000:
        startGameModel(env, model, step_len=2500)
    else:
        startGameModel(env, model, step_len=10000)

# For DQN2
for s in steps:
    print(f"DQN2 - {s}")
    env = create_environment(1, CHECKPOINT_DIR_DQN2)
    model = DQN.load(f"{CHECKPOINT_DIR_DQN2}/models/iter_{s}")
    if s == 10000 or s == 100000:
        startGameModel(env, model, step_len=2500)
    else:
        startGameModel(env, model, step_len=10000)

# For DQN3
for s in steps:
    print(f"DQN3 - {s}")
    env = create_environment(2, CHECKPOINT_DIR_DQN3)
    model = DQN.load(f"{CHECKPOINT_DIR_DQN3}/models/iter_{s}")
    if s == 10000 or s == 100000:
        startGameModel(env, model, step_len=2500)
    else:
        startGameModel(env, model, step_len=10000)

# Record the video gameplay of the best model (PPO2 - 5M training steps)
env = create_environment(1, CHECKPOINT_DIR_PPO_BEST)
model = PPO.load(f"{CHECKPOINT_DIR_PPO_BEST}/best_model")
saveGameModel(env, model)
