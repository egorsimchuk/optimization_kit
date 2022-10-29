"""Q Learning actor model"""
import math
import random

import pandas as pd
from typing import Tuple, List, Optional

import numpy as np

from actors.base import BaseActor


class QMatrix():

    def __init__(self, state_bounds: List[Tuple[float, float]], state_granularity: List[float],
                 action_bounds: List[Tuple[float, float]], action_granularity: List[float]):
        self.state_bounds = state_bounds
        self.state_granularity = state_granularity
        self.action_bounds = action_bounds
        self.action_granularity = action_granularity
        self.state_space: List[List[float]] = []
        self.action_space: List[List[float]] = []
        self.state_shape: Tuple[int] = ()
        self.action_shape: Tuple[int] = ()
        self._define_matrix()

    def load(self, filepath):
        matrix = np.load(filepath)
        if matrix.shape != self.matrix.shape:
            raise ValueError(f"Can't load q matrix. Shape is {matrix.shape}, but need {self.matrix.shape}")
        self.matrix = matrix

    def save(self, filepath):
        np.save(filepath, self.matrix)

    def _define_matrix(self):
        for bounds, granularity in zip(self.state_bounds, self.state_granularity):
            self.state_space.append(np.arange(bounds[0], bounds[1] + granularity, granularity))

        for bounds, granularity in zip(self.action_bounds, self.action_granularity):
            self.action_space.append(np.arange(bounds[0], bounds[1] + granularity, granularity))

        self.state_shape = tuple(len(dim) for dim in self.state_space)
        self.action_shape = tuple(len(dim) for dim in self.action_space)
        self.matrix = np.zeros(self.state_shape + self.action_shape)

    def get_random_action(self, state: np.ndarray) -> Tuple[np.ndarray, Tuple[int]]:
        idxs = tuple(random.randint(0, dim - 1) for dim in self.action_shape)
        return np.array([space[idx] for space, idx in zip(self.action_space, idxs)]), idxs

    def get_optimal_action(self, state: np.ndarray) -> Tuple[np.ndarray, Tuple[int]]:
        actions = self.matrix[self.find_state_position(state)]
        idxs = np.unravel_index(actions.argmax(), self.action_shape)
        if np.all(np.asarray(idxs) == 0) and actions[idxs] == 0:
            return self.get_random_action(state)
        return np.array([space[idx] for space, idx in zip(self.action_space, idxs)]), idxs

    def find_q_position(self, state: np.ndarray, action_idxs: Tuple[int]) -> Tuple[int]:
        return tuple(find_nearest(space, state_value) for space, state_value in zip(self.state_space, state)) + action_idxs

    def find_state_position(self, state: np.ndarray) -> Tuple[int]:
        return tuple(find_nearest(space, state_value) for space, state_value in zip(self.state_space, state))

    def find_max_q(self, state: np.ndarray) -> Tuple[int]:
        res = self.matrix[self.find_state_position(state)].max()
        return res


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


class QLearningActor(BaseActor):

    def __init__(self, learning_rate: float, discount_factor: float, q_matrix: QMatrix, explore_rate: float):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_matrix = q_matrix
        self.prev_state = None
        self.prev_action_idxs: Tuple[int] = None
        self.explore_rate = explore_rate

    def get_action(self, state: np.ndarray, prev_reward: float) -> np.ndarray:
        self.q_matrix.get_optimal_action(np.array([ 2.32605438, -6.71626693]))
        if random.uniform(0, 1) < self.explore_rate:
            # explore: take random action
            action, action_idxs = self.q_matrix.get_random_action(state)
        else:
            # exploit: take the most optimal action
            action, action_idxs = self.q_matrix.get_optimal_action(state)
        if self.prev_state is not None:
            self._update_q_matrix(state, prev_reward)
        self.prev_state = state
        self.prev_action_idxs = action_idxs
        return action

    def _update_q_matrix(self, state, prev_reward):
        prev_pos = self.q_matrix.find_q_position(self.prev_state, self.prev_action_idxs)
        max_q = self.q_matrix.find_max_q(state)
        self.q_matrix.matrix[prev_pos] += self.learning_rate * (prev_reward + self.discount_factor * max_q - self.q_matrix.matrix[prev_pos])
