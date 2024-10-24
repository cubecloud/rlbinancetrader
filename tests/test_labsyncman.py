import sys
sys.path.insert(0, '/home/cubecloud/Python/projects/rlbinancetrader')
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
# from rllab import SyncManVecEnv
from rllab import LabSubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import shutil as sh
import PIL
import datetime
from pytz import timezone
from multiprocessing import freeze_support

TZ = timezone('Europe/Moscow')


def save_mp4(eps_frames, path_filename, fps=25):
    eps_frame_dir = 'episode_frames'
    os.mkdir(eps_frame_dir)

    for i, frame in enumerate(eps_frames):
        PIL.Image.fromarray(frame).save(os.path.join(eps_frame_dir, f'frame-{i + 1}.png'))

    os.system(f'ffmpeg -v 0 -r {fps} -i {eps_frame_dir}/frame-%1d.png -vcodec libx264 -b 10M -y "{path_filename}"');
    sh.rmtree(eps_frame_dir)


if __name__ == "__main__":
    freeze_support()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('numba').setLevel(logging.INFO)
    logging.getLogger('gymnasium').setLevel(logging.INFO)
    logging.getLogger('LoadDbIndicators').setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create the environment
    # train_vec_env = LabSubprocVecEnv("CartPole-v1", port=5643, authkey=b"password_1", n_envs=120)
    seed = 443
    env_kwargs = dict(render_mode=None,
                      continuous=True,
                      gravity=-9.8,
                      enable_wind=True,
                      wind_power=15.0,
                      turbulence_power=1.5)

    train_vec_env_kwargs = dict(env_id="LunarLander-v2",
                                env_kwargs=env_kwargs,
                                n_envs=2000,
                                seed=seed,
                                vec_env_cls=LabSubprocVecEnv)

    train_vec_env = make_vec_env(**train_vec_env_kwargs)

    model = PPO("MlpPolicy", train_vec_env, verbose=1, device="cuda", n_steps=60, batch_size=2_000, n_epochs=10,
                learning_rate=3e-5, ent_coef=0.01, gamma=0.99, stats_window_size=200, seed=seed)

    # Train the model
    model.learn(total_timesteps=20_000_000, progress_bar=True)
    main_path = "/home/cubecloud/Python/projects/rlbinancetrader/tests/save"
    path_filename = os.path.join(main_path, "lunarlander_v2")
    # Save the model
    model.save(path=path_filename)

    env_kwargs.update(dict(render_mode="rgb_array"))

    eval_vec_env_kwargs = dict(env_id="LunarLander-v2",
                               env_kwargs=env_kwargs,
                               n_envs=1,
                               seed=42,
                               vec_env_cls=DummyVecEnv)
    # Load the model
    eval_vec_env = make_vec_env(**eval_vec_env_kwargs)
    model = PPO.load(path=path_filename, env=eval_vec_env, device="cpu")
    env = model.get_env()

    # Test the model
    obs = env.reset()
    frames = []
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')  # Render the picture as an RGB array
        frames.append(frame)
        if done:
            break

    exp_id = f'{datetime.datetime.now(TZ).strftime("%d%m-%H%M%S")}'
    save_mp4(frames, os.path.join(main_path, f"lunarlander_v2_{exp_id}.mp4"))
