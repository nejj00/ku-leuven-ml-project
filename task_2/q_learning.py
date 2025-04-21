"""
Q-Learning implementations for multi-agent reinforcement learning.

This module provides base class and concrete implementations for:
1. Epsilon-Greedy Q-learning
2. Boltzmann (softmax) Q-learning
"""
import random
from abc import ABC, abstractmethod
import numpy as np
import copy as cp

# Available actions for all agents (assuming 2x2 games)
ACTIONS = [0, 1]

class QLearning(ABC):
    """
    Abstract base class for Q-learning algorithms.
    
    Defines the interface for different Q-learning variants.
    """
    @abstractmethod
    def choose_action(self, q_table) -> int:
        """
        Select an action based on the current Q-values.
        
        Args:
            q_table: The Q-values for each action
            
        Returns:
            int: The selected action index
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, q_table) -> list[float]:
        """
        Calculate the probability distribution over actions.
        
        Args:
            q_table: The Q-values for each action
            
        Returns:
            list[float]: Probability for each action
        """
        pass
    
    @abstractmethod
    def decay_parameters(self) -> None:
        """Decay exploration/exploitation parameters."""
        pass
    
    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset exploration/exploitation parameters to initial values."""
        pass
        
    @abstractmethod
    def copy(self):
        """
        Create a deep copy of this learning algorithm.
        
        Returns:
            QLearning: A new instance with the same parameters
        """
        pass


class EpsilonGreedyQLearning(QLearning):
    """
    Epsilon-Greedy Q-learning implementation.
    
    Balances exploration and exploitation using an epsilon parameter:
    - With probability epsilon: choose random action (exploration)
    - With probability 1-epsilon: choose best action (exploitation)
    """
    def __init__(self, epsilon, min_epsilon, epsilon_decay, alpha=0.1):
        """
        Initialize epsilon-greedy Q-learning parameters.
        
        Args:
            epsilon: Initial exploration rate
            min_epsilon: Minimum exploration rate (floor)
            epsilon_decay: Rate at which epsilon decays over time
            alpha: Learning rate
        """
        self.epsilon = epsilon
        self.starter_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

    def choose_action(self, q_table):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            q_table: Array of Q-values for each action
            
        Returns:
            int: Selected action index
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(ACTIONS)  # Explore
        return np.argmax(q_table)  # Exploit
    
    def get_action_probabilities(self, q_table):
        """
        Calculate action probabilities under epsilon-greedy policy.
        
        Args:
            q_table: Array of Q-values for each action
            
        Returns:
            list[float]: Probability for each action
        """
        num_actions = len(ACTIONS)
        greedy_action = np.argmax(q_table)

        # Assign ε/N probability to all actions (exploration)
        probabilities = np.full(num_actions, self.epsilon / num_actions)
        
        # Add (1-ε) to the greedy action (exploitation)
        probabilities[greedy_action] += 1 - self.epsilon

        return probabilities

    def decay_parameters(self):
        """Decay the exploration rate (epsilon)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def reset_parameters(self):
        """Reset epsilon to its initial value."""
        self.epsilon = self.starter_epsilon
        
    def copy(self):
        """
        Create a deep copy of this learning algorithm.
        
        Returns:
            EpsilonGreedyQLearning: A new instance with the same parameters
        """
        return EpsilonGreedyQLearning(
            epsilon=self.starter_epsilon,
            min_epsilon=self.min_epsilon,
            epsilon_decay=self.epsilon_decay,
            alpha=self.alpha
        )


class BoltzmannQLearning(QLearning):
    """
    Boltzmann (Softmax) Q-learning implementation.
    
    Uses a probabilistic action selection based on Q-values and temperature:
    - Higher temperature: more exploration (more uniform distribution)
    - Lower temperature: more exploitation (more peaked distribution)
    """
    def __init__(self, temperature, temperature_min, temperature_decay, alpha=0.1):
        """
        Initialize Boltzmann Q-learning parameters.
        
        Args:
            temperature: Controls exploration-exploitation balance
            temperature_min: Minimum temperature (floor)
            temperature_decay: Rate at which temperature decays over time
            alpha: Learning rate
        """
        self.temperature = temperature
        self.starter_temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.alpha = alpha

    def softmax(self, q_values):
        """
        Compute softmax probabilities with temperature scaling.
        
        Args:
            q_values: Array of Q-values for each action
            
        Returns:
            np.array: Probability distribution over actions
        """
        # Subtract max value for numerical stability
        q_max = np.max(q_values)
        exp_q = np.exp((q_values - q_max) / self.temperature)
        return exp_q / np.sum(exp_q)

    def choose_action(self, q_values):
        """
        Select action using Boltzmann distribution.
        
        Args:
            q_values: Array of Q-values for each action
            
        Returns:
            int: Selected action index
        """
        probabilities = self.softmax(q_values)
        return np.random.choice(ACTIONS, p=probabilities)
    
    def get_action_probabilities(self, q_table):
        """
        Calculate action probabilities under Boltzmann policy.
        
        Args:
            q_table: Array of Q-values for each action
            
        Returns:
            np.array: Probability for each action
        """
        return self.softmax(q_table)
    
    def decay_parameters(self):
        """Decay the temperature parameter."""
        self.temperature = max(self.temperature * self.temperature_decay, 
                               self.temperature_min)
        
    def reset_parameters(self):
        """Reset temperature to its initial value."""
        self.temperature = self.starter_temperature
        
    def copy(self):
        """
        Create a deep copy of this learning algorithm.
        
        Returns:
            BoltzmannQLearning: A new instance with the same parameters
        """
        return BoltzmannQLearning(
            temperature=self.starter_temperature,
            temperature_min=self.temperature_min,
            temperature_decay=self.temperature_decay,
            alpha=self.alpha
        )