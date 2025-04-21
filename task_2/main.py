"""
Main script for running multi-agent reinforcement learning experiments.

This script coordinates the execution of experiments with different
learning algorithms and matrix games, and visualizes the results.
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

from matrix_game import MatrixGame, PrisonnersDilemma, StagHunt, MatchingPennies
from q_learning import BoltzmannQLearning, EpsilonGreedyQLearning
from experiment import run_multiple_experiments, get_default_starting_points
from visualization import plot_combined_visualization


def run_temperature_comparison_experiment(temperatures, game_class=PrisonnersDilemma, n_processes=None):
    """
    Run experiments with Boltzmann Q-learning at different temperatures.
    
    Args:
        temperatures: List of temperature values to test
        game_class: The matrix game class to use
        n_processes: Number of processes to use (None = use all available cores)
        
    Returns:
        None (displays plots)
    """
    # Log start time for performance measurement
    start_time = time.time()
    
    # Experiment parameters
    alpha = 0.003  # Learning rate
    gamma = 0      # No discounting for matrix games
    episodes = 10000
    runs_per_start_point = 10
    
    # Get default starting points for the experiments
    fixed_start_points = get_default_starting_points()
    
    # Create a figure with subplots for each temperature
    num_temps = len(temperatures)
    max_cols = min(3, num_temps)
    cols = max_cols
    rows = (num_temps + cols - 1) // cols

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    # Normalize axes to be a flat array no matter what
    if num_temps == 1:
        axes = [axes]  # single Axes object -> wrap in list
    else:
        axes = axes.flatten()

    # Run experiments for each temperature
    for i, temp in enumerate(temperatures):
        # Initialize the game and learning algorithm
        game = game_class()
        boltzmann_q = BoltzmannQLearning(
            temperature=temp,
            temperature_min=0.01,  # Allow temperature to decay to a small value for better exploitation
            temperature_decay=0.9999,
            alpha=alpha
        )
        
        # Run the experiments with parallel processing
        all_avg_player1_probs, all_avg_player2_probs = run_multiple_experiments(
            q_learning=boltzmann_q,
            matrix_game=game,
            fixed_start_points=fixed_start_points,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            runs_per_start_point=runs_per_start_point,
            n_processes=n_processes
        )
        
        # Create the visualization
        title = f"Temperature = {temp}"
        plot_combined_visualization(
            all_avg_player1_probs, 
            all_avg_player2_probs, 
            game, 
            boltzmann_q, 
            axes[i],
            title=title
        )

    # Hide any unused subplots
    for j in range(num_temps, len(axes)):
        fig.delaxes(axes[j])

    # Add title and adjust layout
    fig.suptitle(f"Boltzmann Q-Learning: {game.name} - Average Cooperation Probability Trajectories", 
                fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Log and display execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    plt.show()


def run_game_comparison_experiment(game_classes, q_learning_algorithm, n_processes=None, use_leniency=False):
    """
    Run experiments with different games using the same learning algorithm.
    
    Args:
        game_classes: List of game classes to test
        q_learning_algorithm: The Q-learning algorithm to use
        n_processes: Number of processes to use (None = use all available cores)
        
    Returns:
        None (displays plots)
    """
    # Log start time for performance measurement
    start_time = time.time()
    
    # Experiment parameters
    alpha = 0.005    # Learning rate
    gamma = 0       # No discounting for matrix games
    episodes = 10000 
    runs_per_start_point = 30
    
    # Get default starting points for the experiments
    fixed_start_points = get_default_starting_points()
    
    # Create a figure with subplots for each game
    num_games = len(game_classes)
    max_cols = min(3, num_games)
    cols = max_cols
    rows = (num_games + cols - 1) // cols

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    # Normalize axes to be a flat array no matter what
    if num_games == 1:
        axes = [axes]  # single Axes object -> wrap in list
    else:
        axes = axes.flatten()

    # Run experiments for each game
    for i, game_class in enumerate(game_classes):
        # Initialize the game
        game = game_class()
        
        # Use lenient learning for social dilemmas (PD and Stag Hunt) but not for zero-sum games
        leniency_aux = game_class.__name__ in ["StagHunt", "SubsidyGame"] and use_leniency
        
        # Run the experiments with parallel processing
        all_avg_player1_probs, all_avg_player2_probs = run_multiple_experiments(
            q_learning=q_learning_algorithm,
            matrix_game=game,
            fixed_start_points=fixed_start_points,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            runs_per_start_point=runs_per_start_point,
            use_leniency=leniency_aux,
            n_processes=n_processes
        )
        
        # Create the visualization
        leniency_str = " (with Leniency)" if leniency_aux else ""
        plot_combined_visualization(
            all_avg_player1_probs, 
            all_avg_player2_probs, 
            game, 
            q_learning_algorithm, 
            axes[i],
            title=f"{game.name}{leniency_str}"
        )

    # Hide any unused subplots
    for j in range(num_games, len(axes)):
        fig.delaxes(axes[j])

    # Get algorithm name for title
    algo_name = q_learning_algorithm.__class__.__name__
    
    # Add title and adjust layout
    fig.suptitle(f"{algo_name}: Comparison Across Different Games", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Log and display execution time
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    plt.show()


if __name__ == "__main__":
    # Get number of available CPU cores
    num_cores = mp.cpu_count()
    print(f"Running with {num_cores} CPU cores available")
    
    # Set number of processes to use (you can adjust this parameter)
    use_cores = max(1, num_cores - 1)  # Use all cores except one by default
    print(f"Using {use_cores} cores for parallel processing")
    
    # Example 1: Compare different temperatures for Boltzmann Q-learning
    # temperatures = [1.0, 0.5, 0.1]
    # run_temperature_comparison_experiment(temperatures, n_processes=use_cores)
    
    # Example 2: Compare different games with a single learning algorithm
    games = [PrisonnersDilemma, StagHunt, MatchingPennies]
    boltzmann_q = BoltzmannQLearning(
        temperature=1.0,
        temperature_min=0.01,
        temperature_decay=0.999,
        alpha=0.01
    )
    run_game_comparison_experiment(games, boltzmann_q, n_processes=use_cores, use_leniency=False)
    
    # Example 3: Compare different learning algorithms on the same game
    # Uncomment to run this experiment
    # boltzmann_q = BoltzmannQLearning(
    #     temperature=1.0, 
    #     temperature_min=0.05, 
    #     temperature_decay=0.9998, 
    #     alpha=0.01
    # )
    # epsilon_q = EpsilonGreedyQLearning(
    #     epsilon=0.3,              # Higher initial exploration
    #     min_epsilon=0.05,         # Same floor
    #     epsilon_decay=0.9998,     # Slower decay
    #     alpha=0.01                # Higher learning rate
    # )
    # algorithms = [boltzmann_q, epsilon_q]
    # 
    # # Create a figure with subplots for each algorithm
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 
    # for i, algo in enumerate(algorithms):
    #     game = PrisonnersDilemma()
    #     all_avg_player1_probs, all_avg_player2_probs = run_multiple_experiments(
    #         q_learning=algo,
    #         matrix_game=game,
    #         fixed_start_points=get_default_starting_points(),
    #         episodes=10000,
    #         alpha=0.01,
    #         gamma=0,
    #         runs_per_start_point=10,
    #         use_leniency=True,    # Enable lenient learning
    #         n_processes=use_cores  # Use parallel processing
    #     )
    #     
    #     algo_name = algo.__class__.__name__
    #     plot_combined_visualization(
    #         all_avg_player1_probs, 
    #         all_avg_player2_probs, 
    #         game, 
    #         algo, 
    #         axes[i],
    #         title=f"{game.name} - {algo_name}"
    #     )
    # 
    # fig.suptitle("Comparison of Q-Learning Algorithms", fontsize=16)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()