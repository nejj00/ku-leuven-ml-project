"""
Experiment module for running multi-agent reinforcement learning experiments.

This module provides functions for running experiments with different
learning algorithms and matrix games.
"""
import numpy as np
from collections import deque
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

from matrix_game import MatrixGame
from q_learning import QLearning

def run_single_experiment(
    q_learning: QLearning,
    episodes: int,
    matrix_game: MatrixGame,
    alpha: float,
    gamma: float,
    start_q_table_1: List[float],
    start_q_table_2: List[float],
    use_leniency: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Run a single experiment trajectory from a given starting point.
    
    Args:
        q_learning: The Q-learning algorithm to use
        episodes: Number of episodes to run
        matrix_game: The matrix game environment
        alpha: Learning rate
        gamma: Discount factor
        start_q_table_1: Initial Q-values for player 1
        start_q_table_2: Initial Q-values for player 2
        use_leniency: Whether to use lenient learning (default: False)
        
    Returns:
        Tuple containing:
        - List of player 1's cooperation probabilities over time
        - List of player 2's cooperation probabilities over time
    """
    # Initialize Q-tables with the provided starting values
    q_learning.reset_parameters()  # Reset exploration/exploitation parameters
    q_table_1 = np.array(start_q_table_1, dtype=float)
    q_table_2 = np.array(start_q_table_2, dtype=float)
    
    player1_coop_probs = []  # Store cooperation probabilities for this run
    player2_coop_probs = []  # Store cooperation probabilities for this run
    
    # Setup for lenient learning if enabled
    kappa = 5  # Leniency buffer size
    reward_buffer_1 = [deque(maxlen=kappa) for _ in range(len(q_table_1))]
    reward_buffer_2 = [deque(maxlen=kappa) for _ in range(len(q_table_2))]
    
    for episode in range(episodes):
        # Get current action probabilities
        plotted_action = matrix_game.get_plotted_action()
        prob1 = q_learning.get_action_probabilities(q_table_1)[plotted_action]
        prob2 = q_learning.get_action_probabilities(q_table_2)[plotted_action]
        
        # Record probabilities for the action being tracked
        player1_coop_probs.append(prob1)
        player2_coop_probs.append(prob2)
        
        # Choose actions based on current policy
        action1 = q_learning.choose_action(q_table_1)
        action2 = q_learning.choose_action(q_table_2)

        # Get rewards from the matrix game
        reward1, reward2 = matrix_game.payoffs[(action1, action2)]
        
        # Store rewards in buffer for lenient learning
        reward_buffer_1[action1].append(reward1)
        reward_buffer_2[action2].append(reward2)

        if use_leniency:
            # Use maximum reward in buffer for lenient updates
            max_reward1 = max(reward_buffer_1[action1]) if reward_buffer_1[action1] else reward1
            max_reward2 = max(reward_buffer_2[action2]) if reward_buffer_2[action2] else reward2
            
            # Update Q-values using lenient learning
            q_table_1[action1] += alpha * (max_reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
            q_table_2[action2] += alpha * (max_reward2 + gamma * np.max(q_table_2) - q_table_2[action2])
        else:
            # Standard Q-learning update
            q_table_1[action1] += alpha * (reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
            q_table_2[action2] += alpha * (reward2 + gamma * np.max(q_table_2) - q_table_2[action2])

        # Decay exploration parameters
        q_learning.decay_parameters()
    
    return player1_coop_probs, player2_coop_probs


def run_experiments_for_starting_point(
    start_point: Tuple[List[float], List[float]],
    q_learning: QLearning,
    matrix_game: MatrixGame,
    episodes: int,
    alpha: float,
    gamma: float,
    runs_per_start_point: int,
    use_leniency: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to run multiple experiments for a specific starting point.
    This is designed to be used with multiprocessing.
    
    Args:
        start_point: Tuple of (player1_q_values, player2_q_values)
        q_learning: The Q-learning algorithm to use
        matrix_game: The matrix game environment
        episodes: Number of episodes per experiment
        alpha: Learning rate
        gamma: Discount factor
        runs_per_start_point: Number of runs to average for this starting point
        use_leniency: Whether to use lenient learning
        
    Returns:
        Tuple containing:
        - Average player 1 cooperation probabilities for this starting point
        - Average player 2 cooperation probabilities for this starting point
    """
    start_q1, start_q2 = start_point
    runs_player1_probs = []
    runs_player2_probs = []
    
    # Run multiple experiments for this starting point
    for _ in range(runs_per_start_point):
        # Create a fresh copy of the learning algorithm to ensure independent runs
        q_learning_copy = q_learning.copy()
        
        p1_probs, p2_probs = run_single_experiment(
            q_learning=q_learning_copy,
            episodes=episodes,
            matrix_game=matrix_game,
            alpha=alpha,
            gamma=gamma,
            start_q_table_1=start_q1,
            start_q_table_2=start_q2,
            use_leniency=use_leniency
        )
        runs_player1_probs.append(p1_probs)
        runs_player2_probs.append(p2_probs)
    
    # Calculate the average trajectory for this starting point
    avg_p1_probs = np.mean(runs_player1_probs, axis=0)
    avg_p2_probs = np.mean(runs_player2_probs, axis=0)
    
    return avg_p1_probs, avg_p2_probs


def run_multiple_experiments(
    q_learning: QLearning,
    matrix_game: MatrixGame,
    fixed_start_points: List[Tuple[List[float], List[float]]],
    episodes: int, 
    alpha: float,
    gamma: float,
    runs_per_start_point: int = 10,
    use_leniency: bool = False,
    n_processes: int = None
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Run multiple experiments from different starting points, potentially in parallel.
    
    Args:
        q_learning: The Q-learning algorithm to use
        matrix_game: The matrix game environment
        fixed_start_points: List of starting Q-values for both players
        episodes: Number of episodes per experiment
        alpha: Learning rate
        gamma: Discount factor
        runs_per_start_point: Number of runs to average for each starting point
        use_leniency: Whether to use lenient learning
        n_processes: Number of processes to use (None = use all available cores)
        
    Returns:
        Tuple containing:
        - List of average player 1 cooperation probabilities for each starting point
        - List of average player 2 cooperation probabilities for each starting point
    """
    all_avg_player1_probs = []
    all_avg_player2_probs = []

    # Create a partial function with all fixed parameters
    run_for_start_point = partial(
        run_experiments_for_starting_point,
        q_learning=q_learning,
        matrix_game=matrix_game,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        runs_per_start_point=runs_per_start_point,
        use_leniency=use_leniency
    )
    
    # Use multiprocessing to parallelize computations across starting points
    if n_processes != 1:  # Allow for disabling parallel processing
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(run_for_start_point, fixed_start_points)
        
        # Extract results
        for avg_p1_probs, avg_p2_probs in results:
            all_avg_player1_probs.append(avg_p1_probs)
            all_avg_player2_probs.append(avg_p2_probs)
    else:
        # Sequential processing (fallback)
        for start_point in fixed_start_points:
            avg_p1_probs, avg_p2_probs = run_for_start_point(start_point)
            all_avg_player1_probs.append(avg_p1_probs)
            all_avg_player2_probs.append(avg_p2_probs)
    
    return all_avg_player1_probs, all_avg_player2_probs


# Common starting points for experiments
def get_default_starting_points():
    """
    Return default starting points for experiments.
    
    Returns:
        List of (player1_q_values, player2_q_values) pairs
    """
    return [
        ([1, 0], [0, 1]),      # P1: Cooperate bias, P2: Defect bias
        ([0.5, 0], [0, 0.5]),  # P1: Mild cooperate bias, P2: Mild defect bias
        ([0, 1], [1, 0]),      # P1: Defect bias, P2: Cooperate bias
        ([0, 0.5], [0.5, 0]),  # P1: Mild defect bias, P2: Mild cooperate bias
        ([0, 0], [0, 0]),      # Neutral start (no bias)
        ([1, 0], [1, 0]),      # Both cooperate bias
    ]