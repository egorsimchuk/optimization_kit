"""Pendulum balancing task"""
import random
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

from actors.base import BaseActor
from actors.q_learning_actor import QMatrix, QLearningActor
from gym.envs.classic_control.pendulum import angle_normalize


def train():
    # env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
    env = gym.make('Pendulum-v1', g=9.81)

    filepath = Path("/home/esimchuk/repos/optimization_kit/data/pendulum_q_weights.npy")
    q_matrix = QMatrix(state_bounds=[(-np.pi / 2, np.pi / 2), (-1, 1)], state_granularity=[0.1, 0.1],
                       action_bounds=[(-env.max_torque, env.max_torque)], action_granularity=[0.1])
    if filepath.exists():
        q_matrix.load(filepath)
        logger.info(q_matrix.matrix)
    actor = QLearningActor(learning_rate=0.9, discount_factor=0.5, q_matrix=q_matrix, explore_rate=0)

    info = []
    for i in tqdm.tqdm(range(1000)):
        run_simulation(env, actor, n_step=10000)
        info.append({"iter": i, "zero_percentage": np.sum(q_matrix.matrix == 0) / np.prod(q_matrix.matrix.shape)})
        if i % 50 == 0:
            visualize_matrix(q_matrix, title=f"State space exploration, iter {i}")

    env.close()
    visualize_info(info)
    q_matrix.save(filepath)


def visualize_matrix(q_matrix, title):
    fig = go.Figure(data=go.Heatmap(
        z=np.clip(np.sum(q_matrix.matrix, axis=2), -100, None),
        x=q_matrix.state_space[1],
        y=q_matrix.state_space[0],
        colorscale='Viridis'))

    fig.update_layout(title=title, xaxis_title="Angular velocity", yaxis_title="Angle")
    fig.show()


def visualize_info(info):
    fig = px.line(pd.DataFrame(info), x="iter", y="zero_percentage")
    fig.show()


def run_simulation(env, actor, n_step, sleep_seconds: float=0):
    reward = 0
    env.reset()
    for i in range(n_step):
        state = np.array([angle_normalize(env.state[0]), env.state[1]])
        action = actor.get_action(state, reward)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            logger.debug("env was terminated")
            env.reset()
        if sleep_seconds:
            logger.info(f"action={action}")
            sleep(sleep_seconds)



def test():
    env = gym.make('Pendulum-v1', g=9.81, render_mode="human")

    filepath = Path("/home/esimchuk/repos/optimization_kit/data/pendulum_q_weights.npy")
    q_matrix = QMatrix(state_bounds=[(-np.pi / 2, np.pi / 2), (-1, 1)], state_granularity=[0.1, 0.1],
                       action_bounds=[(-env.max_torque, env.max_torque)], action_granularity=[0.1])
    q_matrix.load(filepath)
    actor = QLearningActor(learning_rate=0.9, discount_factor=0.5, q_matrix=q_matrix, explore_rate=0)
    run_simulation(env, actor, n_step=1000, sleep_seconds=0.5)
    env.close()


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    train()
    # test()
