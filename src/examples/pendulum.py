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

from actors.q_learning_actor import QMatrix, QLearningActor
from gym.envs.classic_control.pendulum import angle_normalize

from utils.plot import save_multiple_formats

MAX_EPISODE_STEPS = 300
DATA_FOLDER = Path(__file__).parents[2] / "data/pendulum"


def train(filepath, q_matrix, actor):
    env = gym.make('Pendulum-v1', g=9.81, max_episode_steps=MAX_EPISODE_STEPS)
    if filepath.exists():
        logger.info(f"Start. Use matrix from {filepath}")
        q_matrix.load(filepath)
    else:
        logger.info("Start. Use all zero matrix")

    info = []
    for i in tqdm.tqdm(range(1000)):
        run_simulation(env, actor, n_step=10000)
        info.append({"iter": i, "zero_percentage": np.sum(q_matrix.matrix == 0) / np.prod(q_matrix.matrix.shape)})
        if i % 100 == 0:
            visualize_matrix(q_matrix, title=f"State space exploration, iter {i}")

    env.close()
    visualize_info(info)
    q_matrix.save(filepath)


def visualize_matrix(q_matrix, title):
    fig = go.Figure(data=go.Heatmap(
        z=np.clip(np.max(q_matrix.matrix, axis=2), -100, None),
        x=q_matrix.state_space[1],
        y=q_matrix.state_space[0],
        colorscale='Viridis'))
    fig.update_layout(title=title, xaxis_title="Angular velocity", yaxis_title="Angle")
    save_multiple_formats(fig, DATA_FOLDER / f"matrix_max_updates/{title}")

    for angle_idx in [5, 15]:
        plot_matrix = q_matrix.matrix[angle_idx]
        fig = go.Figure(data=go.Heatmap(
            z=np.clip(plot_matrix, np.quantile(plot_matrix, 0.05), None),
            x=q_matrix.action_space[0],
            y=q_matrix.state_space[1],
            colorscale='Viridis'))
        fig.update_layout(title=f"{title}. Angle {q_matrix.state_space[0][angle_idx]}", yaxis_title="Angular velocity", xaxis_title="Momentum")
        save_multiple_formats(fig, DATA_FOLDER / f"angular_idx{angle_idx}_updates/{title}")


def visualize_info(info):
    fig = px.line(pd.DataFrame(info), x="iter", y="zero_percentage")
    save_multiple_formats(fig, DATA_FOLDER / f"zero_percentage_progress")


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


def test(filepath, q_matrix, actor):
    env = gym.make('Pendulum-v1', g=9.81, render_mode="human", max_episode_steps=MAX_EPISODE_STEPS)
    q_matrix.load(filepath)
    q_matrix.matrix
    run_simulation(env, actor, n_step=1000, sleep_seconds=0.1)
    env.close()


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode')
    args = parser.parse_args()

    learning_rate = 1
    discount_factor = 1
    explore_rate = 0.5
    q_matrix = QMatrix(state_bounds=[(-np.pi / 2, np.pi / 2), (-1, 1)], state_granularity=[0.1, 0.1],
                       action_bounds=[(-2, 2)], action_granularity=[0.1])
    actor = QLearningActor(learning_rate=learning_rate, discount_factor=discount_factor, q_matrix=q_matrix, explore_rate=explore_rate)
    filepath = DATA_FOLDER / "pendulum_q_weights_exprate05.npy"

    if args.mode == 'train':
        train(filepath, q_matrix, actor)
    elif args.mode == 'test':
        test(filepath, q_matrix, actor)
    else:
        raise ValueError("--mode argument should be train or test")
