"""Pendulum balancing task"""
import argparse
import sys

import pandas as pd
import tqdm as tqdm
from pathlib import Path
from time import sleep
import gym
from loguru import logger
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import hydra
from omegaconf import DictConfig, OmegaConf
from actors.q_learning_actor import QMatrix, QLearningActor
from gym.envs.classic_control.pendulum import angle_normalize

from utils.plot import save_multiple_formats

DATA_FOLDER = Path(__file__).parents[2] / "data"


def init_env(config, render_mode=None):
    env_conf = dict(config.environment)
    max_torque = env_conf.pop("max_torque")
    env = gym.make(render_mode=render_mode, **env_conf)
    env.env.env.env.max_torque = max_torque
    return env


def train(config: DictConfig, q_matrix, actor):
    env = init_env(config)
    filepath = DATA_FOLDER / config.weights_fpath
    if filepath.exists():
        logger.info(f"Start. Use matrix from {filepath}")
        q_matrix.load(filepath)
    else:
        logger.info("Start. Use all zero matrix")

    info = []
    for i in tqdm.tqdm(range(config.train_params.n_batches)):
        run_simulation(env, actor, n_step=config.train_params.n_step)
        info.append({"iter": i, "zero_percentage": np.sum(q_matrix.matrix == 0) / np.prod(q_matrix.matrix.shape)})
        if i % config.train_params.batch_plot == 0:
            visualize_matrix(q_matrix, title=f"State space exploration, iter {i}")

    env.close()
    visualize_info(info)
    q_matrix.save(filepath)
    logger.info(f"Matrix updated: {filepath}")


def visualize_matrix(q_matrix, title):
    fig = go.Figure(data=go.Heatmap(
        z=np.clip(np.max(q_matrix.matrix, axis=2), -100, None),
        x=q_matrix.state_space[1],
        y=q_matrix.state_space[0],
        colorscale='Viridis'))
    fig.update_layout(title=title, xaxis_title="Angular velocity", yaxis_title="Angle")
    save_multiple_formats(fig, DATA_FOLDER / f"pendulum/matrix_max_updates/{title}")

    for angle_idx in [5, 15]:
        plot_matrix = q_matrix.matrix[angle_idx]
        fig = go.Figure(data=go.Heatmap(
            z=np.clip(plot_matrix, np.quantile(plot_matrix, 0.05), None),
            x=q_matrix.action_space[0],
            y=q_matrix.state_space[1],
            colorscale='Viridis'))
        fig.update_layout(title=f"{title}. Angle {q_matrix.state_space[0][angle_idx]}", yaxis_title="Angular velocity", xaxis_title="Momentum")
        save_multiple_formats(fig, DATA_FOLDER / f"pendulum/angular_idx{angle_idx}_updates/{title}")


def visualize_info(info):
    fig = px.line(pd.DataFrame(info), x="iter", y="zero_percentage")
    save_multiple_formats(fig, DATA_FOLDER / f"pendulum/zero_percentage_progress")


def run_simulation(env, actor, n_step, sleep_seconds: float = 0):
    reward = 0
    env.reset()
    for i in range(n_step):
        state = np.array([angle_normalize(env.state[0]), env.state[1]])
        action = actor.get_action(state, reward)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            logger.debug(f"step {i}: terminated={terminated}, truncated={truncated}")
            env.reset()
        if sleep_seconds:
            sleep(sleep_seconds)


def test(config, q_matrix, actor):
    filepath = DATA_FOLDER / config.weights_fpath
    env = init_env(config, render_mode="human")
    q_matrix.load(filepath)
    run_simulation(env, actor, n_step=config.test_params.n_step, sleep_seconds=config.test_params.sleep_seconds)
    env.close()


@hydra.main(version_base=None, config_path="../../config/pendulum", config_name="q_learning")
def main(config: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    logger.remove()
    logger.add(sys.stderr, level=config.logger_level)

    q_matrix = QMatrix(**config.q_matrix)
    actor = QLearningActor(q_matrix=q_matrix, **config.actor)

    if config.mode == "train":
        train(config, q_matrix, actor)
    elif config.mode == "test":
        test(config, q_matrix, actor)
    else:
        raise ValueError("mode parameter should be train or test")


if __name__ == '__main__':
    main()
