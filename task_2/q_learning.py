import random
from abc import abstractmethod

import numpy as np

ACTIONS = [0, 1]

class QLearning():
    @abstractmethod
    def select_action(self, q_table) -> int:
        """Return an action based on the provided q_values."""
        pass

    @abstractmethod
    def get_action_probabilities(self, q_table) -> list[int]:
        """Return the action probabilities based on the provided q_values."""
        pass


class EpsilonGreedyLearning(QLearning):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, q_table):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(ACTIONS)  # Explore
        return np.argmax(q_table)  # Exploit


class BoltzmannPolicyLearning(QLearning):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def softmax(self, q_values):
        """
        Compute softmax values for x with temperature tau.
        
        This was derived from paper 2 p. 5 Equation 4
        """
        # Subtract max value for numerical stability
        q_max = np.max(q_values)
        e_x = np.exp((q_values - q_max) / self.temperature)
        return e_x / np.sum(e_x)

    def choose_action(self, q_values):
        """Select an action using Boltzmann (softmax) distribution."""
        probabilities = self.softmax(q_values)
        return np.random.choice(ACTIONS, p=probabilities)