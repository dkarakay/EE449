#
# Created on Thu Jun 01 2023
#
# Deniz Karakay 2443307
#
# EE449 HW3 - Training
#


import gym_super_mario_bros
import numpy as np

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from utils import SaveOnBestTrainingRewardCallback, startGameModel, startGameRand


def create_environment(type, checkpoint):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 1st Preprocessing - Type 0 for PPO1 and DQN1
    if type == 0:
        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4, channels_order="last")
        env = VecMonitor(env, f"{checkpoint}/TestMonitor")

    # 2nd Preprocessing - Type 1 for PPO2 and DQN2
    elif type == 1:
        env = GrayScaleObservation(env, keep_dim=False)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=2)
        env = VecMonitor(env, f"{checkpoint}/TestMonitor")

    # 3rd Preprocessing - Type 2 for PPO3 and DQN3
    elif type == 2:
        env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=2, channels_order="first")
        env = VecMonitor(env, f"{checkpoint}/TestMonitor")
    return env


# Checkpoints
CHECKPOINT_DIR_PPO1 = "./train/PPO1/"
CHECKPOINT_DIR_PPO2 = "./train/PPO2/"
CHECKPOINT_DIR_PPO3 = "./train/PPO3/"
CHECKPOINT_DIR_DQN1 = "./train/DQN1/"
CHECKPOINT_DIR_DQN2 = "./train/DQN2/"
CHECKPOINT_DIR_DQN3 = "./train/DQN2_2/"
CHECKPOINT_DIR_DQN2 = "./train/DQN2_MLP_2/"
CHECKPOINT_DIR_DQN3 = "./train/DQN3/"
CHECKPOINT_DIR_DQN_BEST = "./train/DQN_BEST/"
CHECKPOINT_DIR_PPO_BEST = "./train/PPO_BEST/"

LOG_DIR = "./logs/"


# Train PPO1 - Default
env_ppo = create_environment(0, CHECKPOINT_DIR_PPO1)
callback_ppo = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_PPO1
)
model_ppo = PPO(
    "CnnPolicy",
    env_ppo,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000001,
    n_steps=512,
)
model_ppo.learn(total_timesteps=1000000, callback=callback_ppo, tb_log_name="PPO1")


# Train PPO2 - Env 1 - MLP
env_ppo2 = create_environment(1, CHECKPOINT_DIR_PPO2)
callback_ppo2 = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_PPO2
)
model_ppo2 = PPO(
    "MlpPolicy",
    env_ppo2,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000015,
    n_steps=1536,
)

model_ppo2.learn(total_timesteps=1000000, callback=callback_ppo2, tb_log_name="PPO2")


# Train PPO3 - Env 2 - CNN
env_ppo3 = create_environment(2, CHECKPOINT_DIR_PPO3)
callback_ppo3 = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_PPO3
)
model_ppo3 = PPO(
    "CnnPolicy",
    env_ppo3,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000005,
    n_steps=1024,
    ent_coef=0.01,
    gamma=0.99,
    n_epochs=8,
)
model_ppo3.learn(total_timesteps=1000000, callback=callback_ppo3, tb_log_name="PPO3")

# Train DQN1 - Default
env_dqn = create_environment(0, CHECKPOINT_DIR_DQN1)
callback_dqn = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_DQN1
)
model_dqn = DQN(
    "CnnPolicy",
    env_dqn,
    batch_size=192,
    verbose=1,
    learning_starts=10000,
    learning_rate=5e-3,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    train_freq=8,
    buffer_size=10000,
    tensorboard_log=LOG_DIR,
)
model_dqn.learn(total_timesteps=1000000, log_interval=1, callback=callback_dqn)

# Train DQN2 - Env 1 - MLP
env_dqn2 = create_environment(1, CHECKPOINT_DIR_DQN2)
callback_dqn2 = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_DQN2
)
model_dqn2 = DQN(
    "MlpPolicy",
    env_dqn2,
    batch_size=150,
    verbose=1,
    learning_starts=10000,
    learning_rate=10e-3,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    train_freq=4,
    buffer_size=10000,
    tensorboard_log=LOG_DIR,
)
model_dqn2.learn(
    total_timesteps=1000000,
    log_interval=1,
    callback=callback_dqn2,
    tb_log_name="DQN2",
)


# Train DQN3 - Env 2 - CNN
env_dqn3 = create_environment(2, CHECKPOINT_DIR_DQN3)
callback_dqn3 = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_DQN3
)

model_dqn3 = DQN(
    "CnnPolicy",
    env_dqn3,
    batch_size=165,
    verbose=1,
    learning_starts=10000,
    learning_rate=20e-3,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.15,
    train_freq=8,
    buffer_size=10000,
    tensorboard_log=LOG_DIR,
)
model_dqn3.learn(
    total_timesteps=1000000,
    log_interval=1,
    callback=callback_dqn3,
    tb_log_name="DQN3",
)


# Train PPO2 _best - Env 1 - MLP
env_ppo_best = create_environment(1, CHECKPOINT_DIR_PPO_BEST)
callback_ppo_best = SaveOnBestTrainingRewardCallback(
    save_freq=10000, check_freq=1000, chk_dir=CHECKPOINT_DIR_PPO_BEST
)
model_ppo_best = PPO(
    "MlpPolicy",
    env_ppo_best,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.000015,
    n_steps=1536,
)
model_ppo_best.learn(
    total_timesteps=1000000, callback=callback_ppo_best, tb_log_name="PPO_BEST"
)
