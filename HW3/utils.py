import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import RecordVideo
from time import time


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Taken from https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param chk_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, save_freq: int, check_freq: int, chk_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.chk_dir = chk_dir
        self.save_path = os.path.join(chk_dir, "models")
        self.best_mean_reward = -np.inf
        self.save_freq_checkpoints = [0, 10000, 50000, 100000, 250000, 500000, 1000000]


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.n_calls in self.save_freq_checkpoints:
                if self.verbose > 0:
                    print(f"Saving current model to {os.path.join(self.chk_dir, 'models')}")
                self.model.save(os.path.join(self.save_path, f"iter_{self.n_calls}"))

        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.chk_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(
                            f"Saving new best model to {os.path.join(self.chk_dir, 'best_model')}"
                        )
                    self.model.save(os.path.join(self.chk_dir, "best_model"))

        return True


def startGameRand(env):
    fin = True
    for step in range(100000):
        if fin:
            state = env.reset()
        state, reward, fin, info = env.step(env.action_space.sample())
        env.render()
    env.close()


def startGameModel(env, model):
    state = env.reset()
    while True:
        action, _ = model.predict(state)
        state, _, _, _ = env.step(action)
        env.render()


def saveGameRand(env, len=100000, dir="./videos/"):
    env = RecordVideo(env, dir + str(time()) + "/")
    fin = True
    for step in range(len):
        if fin:
            env.reset()
        state, reward, fin, info = env.step(env.action_space.sample())
    env.close()


def saveGameModel(env, model, len=100000, dir="./videos/"):
    env = RecordVideo(env, dir + str(time()) + "/")
    fin = True
    for step in range(len):
        if fin:
            state = env.reset()
        action, _ = model.predict(state)
        state, _, fin, _ = env.step(action)
    env.close()
