"""
Main script for running multi-agent reinforcement learning experiments.

This script coordinates the execution of experiments with different
learning algorithms and matrix games, and visualizes the results.
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

from matrix_game import MatrixGame, PrisonnersDilemma, StagHunt, MatchingPennies, SubsidyGame
from q_learning import BoltzmannQLearning, EpsilonGreedyQLearning
from experiment import run_multiple_experiments, get_default_starting_points
from visualization import plot_combined_visualization


def get_cardinal_starting_points(distance_scaling=2):
    """
    Return starting points at the cardinal and intermediate directions.
    Creates 8 points evenly spaced around the probability space.
    
    Returns:
        List of (player1_q_values, player2_q_values) pairs
    """
    return [
        ([0, 0], [distance_scaling, 0]),                  # North
        ([distance_scaling, 0], [0, 0]),                  # East
        ([0, 0], [0, distance_scaling]),                  # South
        ([0, distance_scaling], [0, 0]),                  # West
        ([distance_scaling, 0], [distance_scaling, 0]),   # Northeast
        ([0, distance_scaling], [distance_scaling, 0]),   # Northwest
        ([distance_scaling, 0], [0, distance_scaling]),   # Southeast
        ([0, distance_scaling], [0, distance_scaling]),   # Southwest
    ]

def get_primary_cardinal():
    """
    Return starting points at the cardinal directions.
    Creates 4 points evenly spaced around the probability space.
    
    Returns:
        List of (player1_q_values, player2_q_values) pairs
    """
    distance_scaling = 2
    return [
        ([0, 0], [distance_scaling, 0]),                  # North
        ([distance_scaling, 0], [0, 0]),                  # East
        ([0, 0], [0, distance_scaling]),                  # South
        ([0, distance_scaling], [0, 0]),                  # West
    ]

def get_intermediate_cardinal():
    """
    Return starting points at the intermediate cardinal directions.
    Creates 4 points evenly spaced around the probability space.
    
    Returns:
        List of (player1_q_values, player2_q_values) pairs
    """
    distance_scaling = 2
    return [
        ([distance_scaling, 0], [distance_scaling, 0]),   # Northeast
        ([0, distance_scaling], [distance_scaling, 0]),   # Northwest
        ([distance_scaling, 0], [0, distance_scaling]),   # Southeast
        ([0, distance_scaling], [0, distance_scaling]),   # Southwest
    ]

def run_experiment_setup(
    game,
    q_learning,
    episodes=1000,
    runs_per_start_point=200,
    alpha=0.1,
    gamma=0,
    plot_rd=False,
    use_leniency=False,
    starting_points=None,
    n_processes=None,
    title=None
):
    """
    Run and visualize a multi-agent reinforcement learning experiment.
    
    Args:
        game: The matrix game to use (instance of MatrixGame)
        q_learning: The Q-learning algorithm to use
        episodes: Number of episodes per experiment
        runs_per_start_point: Number of runs to average for each starting point
        alpha: Learning rate
        gamma: Discount factor
        plot_rd: Whether to plot replicator dynamics
        use_leniency: Whether to use lenient learning
        starting_points: List of starting Q-values for both players, if None uses cardinal points
        n_processes: Number of processes to use (None = use all available cores)
        title: Optional title for the plot
        
    Returns:
        None (displays plots)
    """
    # Log start time for performance measurement
    start_time = time.time()
    
    # Set learning rate
    q_learning.alpha = alpha
    
    # Use provided starting points or default to cardinal points
    if starting_points is None:
        starting_points = get_cardinal_starting_points()
    
    total_starting_points = len(starting_points)
    print(f"Starting experiment with {total_starting_points} different starting points")
    print(f"Each starting point will run {runs_per_start_point} simulations in parallel")
    print(f"Total simulations to run: {total_starting_points * runs_per_start_point}")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Run the experiments with parallel processing
    all_avg_player1_probs, all_avg_player2_probs = run_multiple_experiments(
        q_learning=q_learning,
        matrix_game=game,
        fixed_start_points=starting_points,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        runs_per_start_point=runs_per_start_point,
        use_leniency=use_leniency,
        n_processes=n_processes,
        print_progress=True  # Enable progress printing
    )
    
    # Create the visualization
    plot_combined_visualization(
        all_avg_player1_probs, 
        all_avg_player2_probs, 
        game, 
        q_learning, 
        ax,
        title=title,
        use_leniency=use_leniency,
        plot_rd=plot_rd
    )
    
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
    use_cores = max(1, num_cores - 1)
    print(f"Using {use_cores} cores for parallel processing")
    
    # # ------ Epsilon-Greedy in Matching Pennies ------
    # print("\nRunning Experiment: Epsilon-Greedy Q-learning in Matching Pennies")
    # # Initialize the Matching Pennies game
    # game = MatchingPennies()
    # # Initialize epsilon greedy Q-learning
    # q_learning = EpsilonGreedyQLearning(
    #     epsilon=0.2,
    #     min_epsilon=0.2,
    #     epsilon_decay=1
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=1000,
    #     runs_per_start_point=5000,
    #     alpha=0.01,
    #     gamma=0,
    #     plot_rd=False,
    #     n_processes=use_cores,
    #     title="Epsilon-Greedy Q-learning in Matching Pennies"
    # )
    
    # # ------ Epsilon-Greedy in Prisonners Dilemma ------
    # print("\nRunning Experiment: Epsilon-Greedy Q-learning in Prisonners Dilemma")
    # # Initialize the Prisonners Dilemma game
    # game = PrisonnersDilemma()
    # # Initialize epsilon greedy Q-learning
    # q_learning = EpsilonGreedyQLearning(
    #     epsilon=0.2,
    #     min_epsilon=0.2,
    #     epsilon_decay=1,
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=2000,
    #     runs_per_start_point=50000,
    #     alpha=0.01,
    #     gamma=0,
    #     plot_rd=False,
    #     n_processes=use_cores,
    #     title="Epsilon-Greedy Q-learning in Prisonners Dilemma",
    #     starting_points=get_intermediate_cardinal()
    # )
    
    # # ------ Epsilon-Greedy in Stag Hunt ------
    # print("\nRunning Experiment: Epsilon-Greedy Q-learning in Stag Hunt")
    # # Initialize the Stag Hunt game
    # game = StagHunt()
    # # Initialize epsilon greedy Q-learning
    # q_learning = EpsilonGreedyQLearning(
    #     epsilon=0.2,
    #     min_epsilon=0.2,
    #     epsilon_decay=1,
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=2000,
    #     runs_per_start_point=2000,
    #     alpha=0.01,
    #     gamma=0,
    #     plot_rd=False,
    #     n_processes=use_cores,
    #     title="Epsilon-Greedy Q-learning in Stag Hunt",
    #     starting_points=get_cardinal_starting_points()
    # )
    
    # # ------ Boltzmann Q-learning in Matching Pennies (converging to center) ------
    # print("\nRunning Experiment: Boltzmann Q-learning in Matching Pennies (converging to center)")
    # # Initialize the Matching Pennies game
    # game = MatchingPennies()
    # # Initialize Boltzmann Q-learning
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.995
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=1750,
    #     runs_per_start_point=2000,
    #     alpha=0.006,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning in Matching Pennies (Converging to Center)"
    # )
    
    # # ------ Boltzmann Q-learning  in Matching Pennies (circles) ------
    # print("\nRunning Experiment: Boltzmann Q-learning  in Matching Pennies (circular orbits)")
    # # Initialize the Matching Pennies game
    # game = MatchingPennies()
    # # Initialize Boltzmann Q-learning with different decay rate
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.99
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=300,
    #     runs_per_start_point=2000,
    #     alpha=0.011,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning in Matching Pennies (Circular Orbits)"
    # )
    
    # # ------ Boltzmann in Stag Hunt ------
    # print("\nRunning Experiment: Boltzmann Q-learning in Stag Hunt")
    # # Initialize the Stag Hunt game
    # game = StagHunt()
    # # Initialize Boltzmann Q-learning
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.998
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=4000,
    #     runs_per_start_point=200,
    #     alpha=0.005,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning in Stag Hunt",
    #     starting_points=get_cardinal_starting_points() + get_cardinal_starting_points(distance_scaling=1)
    # )
    
    # # ------ Boltzmann in Prisonners Dilemma ------
    # print("\nRunning Experiment: Boltzmann Q-learning in Prisonners Dilemma")
    # # Initialize the Prisonners Dilemma game
    # game = PrisonnersDilemma()
    # # Initialize Boltzmann Q-learning
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.999
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=4000,
    #     runs_per_start_point=1000,
    #     alpha=0.005,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning in Prisonners Dilemma",
    #     starting_points=get_cardinal_starting_points(1)
    # )
    
    # # ------ Boltzmann Q-learning with leniency in Stag Hunt ------
    # print("\nRunning Experiment: Boltzmann Q-learning with leniency in Stag Hunt")
    # # Initialize the Stag Hunt game
    # game = StagHunt()
    # # Initialize Boltzmann Q-learning with leniency
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.998,
    #     kappa=5
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=4000,
    #     runs_per_start_point=200,
    #     alpha=0.005,
    #     gamma=0,
    #     use_leniency=True,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning with leniency in Stag Hunt",
    #     starting_points=get_cardinal_starting_points() + get_cardinal_starting_points(distance_scaling=1)
    # )
    
    # # ------ Boltzmann Q-learning with leniency in Prisoner's Dilemma ------
    # print("\nRunning Experiment: Boltzmann Q-learning with leniency in Prisoner's Dilemma")
    # # Initialize the Prisoner's Dilemma") game
    # game = PrisonnersDilemma()
    # # Initialize Boltzmann Q-learning with leniency
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.999,
    #     kappa=4
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=4000,
    #     runs_per_start_point=1000,
    #     alpha=0.005,
    #     gamma=0,
    #     use_leniency=True,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning with leniency in Prisoner's Dilemma",
    #     starting_points=get_cardinal_starting_points(1)
    # )
    
    # # ------ Boltzmann Q-learning with leniency in Matching Pennies (converging to center) ------
    # print("\nRunning Experiment: Boltzmann Q-learning with leniency in Matching Pennies (converging to center)")
    # # Initialize the Matching Pennies game
    # game = MatchingPennies()
    # # Initialize Boltzmann Q-learning with leniency
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.995
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=1750,
    #     runs_per_start_point=2000,
    #     alpha=0.006,
    #     use_leniency=True,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning with leniency in Matching Pennies (Converging to Center)"
    # )
    
    # # ------ Boltzmann in Prisonners Dilemma with too heavily biased starting points------
    # print("\nRunning Experiment: Boltzmann Q-learning in Prisonners Dilemma with too heavily biased starting points")
    # # Initialize the Prisonners Dilemma game
    # game = PrisonnersDilemma()
    # # Initialize Boltzmann Q-learning
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.99
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=2000,
    #     runs_per_start_point=200,
    #     alpha=0.00165,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning in Prisonners Dilemma with too heavily biased starting points",
    #     starting_points=get_cardinal_starting_points(3)
    # )
    
    # # ------ Boltzmann in Stag Hunt  with bad temperature decay------
    # print("\nRunning Experiment: Boltzmann Q-learning in Stag Hunt with bad temperature decay")
    # # Initialize the Stag Hunt game
    # game = StagHunt()
    # # Initialize Boltzmann Q-learning
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.9999
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=10000,
    #     runs_per_start_point=200,
    #     alpha=0.005,
    #     gamma=0,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning in Stag Hunt with bad temperature decay",
    #     starting_points=get_cardinal_starting_points(distance_scaling=1)
    # )

    # # ------ Boltzmann Q-learning with a lot of leniency in Stag Hunt ------
    # print("\nRunning Experiment: Boltzmann Q-learning with a lot of leniency in Stag Hunt")
    # # Initialize the Stag Hunt game
    # game = StagHunt()
    # # Initialize Boltzmann Q-learning with leniency
    # q_learning = BoltzmannQLearning(
    #     temperature=1,
    #     temperature_min=0,
    #     temperature_decay=0.9982,
    #     kappa=20
    # )
    # run_experiment_setup(
    #     game=game,
    #     q_learning=q_learning,
    #     episodes=4000,
    #     runs_per_start_point=200,
    #     alpha=0.006,
    #     gamma=0,
    #     use_leniency=True,
    #     plot_rd=True,
    #     n_processes=use_cores,
    #     title="Boltzmann Q-learning with a lot of leniency in Stag Hunt",
    #     starting_points=get_cardinal_starting_points(3)
    # )