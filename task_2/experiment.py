"""
Experiment module for running multi-agent reinforcement learning experiments.

This module provides functions for running experiments with different
learning algorithms and matrix games.
"""
import numpy as np
from collections import deque
from typing import List, Tuple, Callable
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
    kappa = getattr(q_learning, 'kappa', 0)  # Leniency buffer size; default 0 if not Boltzmann
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


def run_single_experiment_wrapper(run_idx: int, params: dict) -> Tuple[List[float], List[float]]:
    """
    Wrapper function for parallel execution of single experiments.
    
    Args:
        run_idx: Index of the current run (not used, but needed for mapping)
        params: Dictionary containing all parameters for the experiment
        
    Returns:
        Tuple containing:
        - List of player 1's cooperation probabilities over time
        - List of player 2's cooperation probabilities over time
    """
    # Create a fresh copy of the learning algorithm to ensure independent runs
    q_learning_copy = params['q_learning'].copy()
    
    # Check if we should print progress updates
    print_progress = params.get('print_progress', False)
    report_interval = params.get('report_interval', 10)
    
    # Print progress if appropriate (only every Nth run to avoid console spam)
    if print_progress and run_idx % report_interval == 0:
        total_runs = params.get('total_runs', 0)
        current_starting_point = params.get('current_starting_point', 0)
        total_starting_points = params.get('total_starting_points', 0)
        print(f"Starting point {current_starting_point}/{total_starting_points}: "
              f"Running simulation {run_idx+1}/{total_runs}")
    
    return run_single_experiment(
        q_learning=q_learning_copy,
        episodes=params['episodes'],
        matrix_game=params['matrix_game'],
        alpha=params['alpha'],
        gamma=params['gamma'],
        start_q_table_1=params['start_q1'],
        start_q_table_2=params['start_q2'],
        use_leniency=params['use_leniency']
    )


def run_experiments_for_starting_point(
    start_point: Tuple[List[float], List[float]],
    q_learning: QLearning,
    matrix_game: MatrixGame,
    episodes: int,
    alpha: float,
    gamma: float,
    runs_per_start_point: int,
    use_leniency: bool,
    n_processes: int = None,
    print_progress: bool = False,
    current_starting_point: int = 0,
    total_starting_points: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to run multiple experiments for a specific starting point.
    This function now parallelizes the individual runs for a single starting point.
    
    Args:
        start_point: Tuple of (player1_q_values, player2_q_values)
        q_learning: The Q-learning algorithm to use
        matrix_game: The matrix game environment
        episodes: Number of episodes per experiment
        alpha: Learning rate
        gamma: Discount factor
        runs_per_start_point: Number of runs to average for this starting point
        use_leniency: Whether to use lenient learning
        n_processes: Number of processes to use (None = use all available cores)
        print_progress: Whether to print progress updates
        current_starting_point: Index of current starting point
        total_starting_points: Total number of starting points
        
    Returns:
        Tuple containing:
        - Average player 1 cooperation probabilities for this starting point
        - Average player 2 cooperation probabilities for this starting point
    """
    start_q1, start_q2 = start_point
    
    if print_progress:
        print(f"Processing starting point {current_starting_point}/{total_starting_points}: {start_point}")
    
    # Package parameters for the wrapper function
    params = {
        'q_learning': q_learning,
        'episodes': episodes,
        'matrix_game': matrix_game,
        'alpha': alpha,
        'gamma': gamma,
        'start_q1': start_q1,
        'start_q2': start_q2,
        'use_leniency': use_leniency,
        'print_progress': print_progress,
        'report_interval': 100,  # Only report every 100th run to minimize console output
        'total_runs': runs_per_start_point,
        'current_starting_point': current_starting_point,
        'total_starting_points': total_starting_points
    }
    
    # Run multiple experiments for this starting point in parallel
    if n_processes != 1:  # Allow for disabling parallel processing
        with mp.Pool(processes=n_processes) as pool:
            # Map each run to a process
            results = pool.map(
                partial(run_single_experiment_wrapper, params=params),
                range(runs_per_start_point)
            )
        
        # Unpack results
        runs_player1_probs, runs_player2_probs = zip(*results)
    else:
        # Sequential processing (fallback)
        runs_player1_probs = []
        runs_player2_probs = []
        for run_idx in range(runs_per_start_point):
            p1_probs, p2_probs = run_single_experiment_wrapper(run_idx, params)
            runs_player1_probs.append(p1_probs)
            runs_player2_probs.append(p2_probs)
    
    # Calculate the average trajectory for this starting point
    avg_p1_probs = np.mean(runs_player1_probs, axis=0)
    avg_p2_probs = np.mean(runs_player2_probs, axis=0)
    
    if print_progress:
        print(f"Completed all {runs_per_start_point} runs for starting point {current_starting_point}/{total_starting_points}")
    
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
    n_processes: int = None,
    print_progress: bool = False
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Run multiple experiments from different starting points.
    For each starting point, the multiple runs are executed in parallel.
    
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
        print_progress: Whether to print progress updates
        
    Returns:
        Tuple containing:
        - List of average player 1 cooperation probabilities for each starting point
        - List of average player 2 cooperation probabilities for each starting point
    """
    all_avg_player1_probs = []
    all_avg_player2_probs = []
    
    total_starting_points = len(fixed_start_points)
    
    # Process each starting point (sequentially)
    for idx, start_point in enumerate(fixed_start_points):
        current_starting_point = idx + 1  # 1-indexed for user display
        
        avg_p1_probs, avg_p2_probs = run_experiments_for_starting_point(
            start_point=start_point,
            q_learning=q_learning,
            matrix_game=matrix_game,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            runs_per_start_point=runs_per_start_point,
            use_leniency=use_leniency,
            n_processes=n_processes,
            print_progress=print_progress,
            current_starting_point=current_starting_point,
            total_starting_points=total_starting_points
        )
        
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