import random
from abc import abstractmethod

import numpy as np

ACTIONS = [0, 1]

class QLearning():
    @abstractmethod
    def choose_action(self, q_table) -> int:
        """Return an action based on the provided q_values."""
        pass

    @abstractmethod
    def get_action_probabilities(self, q_table) -> list[int]:
        """Return the action probabilities based on the provided q_values."""
        pass
    
    @abstractmethod
    def decay_parameters(self) -> None:
        """Return the action probabilities based on the provided q_values."""
        pass
    
    @abstractmethod
    def reset_parameters(self) -> None:
        """Return the action probabilities based on the provided q_values."""
        pass


class EpsilonGreedyQLearning(QLearning):
    def __init__(self, epsilon, min_epsilon, epsilon_decay):
        self.epsilon = epsilon
        self.starter_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def choose_action(self, q_table):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(ACTIONS)  # Explore
        return np.argmax(q_table)  # Exploit
    
    def decay_parameters(self):
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def reset_parameters(self):
        self.epsilon = self.starter_epsilon


class BoltzmannQLearning(QLearning):
    def __init__(self, temperature, temparature_min, temperature_decay):
        self.temperature = temperature
        self.starter_temperature = temperature
        self.temparature_min = temparature_min
        self.temperature_decay = temperature_decay

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
    
    def get_action_probabilities(self, q_table):
        return self.softmax(q_table)
    
    def decay_parameters(self):
        # Decay temperature
        self.temperature = max(self.temperature * self.temperature_decay, self.temparature_min)
        
    def reset_parameters(self):
        self.temparature = self.starter_temperature