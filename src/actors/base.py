"""Base actor model"""
import abc

import numpy as np


class BaseActor():

    @abc.abstractmethod
    def get_action(self, state: np.ndarray, prev_reward: float) -> np.ndarray:
        """Get action for income state"""