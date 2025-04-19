from q_learning import QLearning, BoltzmannQLearning, EpsilonGreedyQLearning
from matrix_game import MatrixGame, PrisonnersDilemma, StagHunt, MatchingPennies
from plotting import plot_rep_dynamics_probability
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


def run_single_experiment(
    q_learning: QLearning,
    episodes: int,
    matrix_game: MatrixGame,
    alpha: float,
    gamma: float,
    start_q_table_1: list[float],
    start_q_table_2: list[float],
):
    """Runs a single experiment trajectory for a given starting point."""
    # Initialize Q-tables with the provided starting values
    q_learning.reset_parameters() # Reset exploration/exploitation parameters
    q_table_1 = np.array(start_q_table_1, dtype=float)
    q_table_2 = np.array(start_q_table_2, dtype=float)
    
    player1_coop_probs = []  # Store cooperation probabilities for this run
    player2_coop_probs = []  # Store cooperation probabilities for this run
    
    kappa = 5 # Keep leniency buffer logic as is
    reward_buffer_1 = [deque(maxlen=kappa) for _ in range(len(q_table_1))]
    reward_buffer_2 = [deque(maxlen=kappa) for _ in range(len(q_table_2))]
    
    for episode in range(episodes):
        prob1 = q_learning.get_action_probabilities(q_table_1)[matrix_game.get_plotted_action()]
        prob2 = q_learning.get_action_probabilities(q_table_2)[matrix_game.get_plotted_action()]
        
        player1_coop_probs.append(prob1)
        player2_coop_probs.append(prob2)
        
        action1 = q_learning.choose_action(q_table_1)
        action2 = q_learning.choose_action(q_table_2)

        reward1, reward2 = matrix_game.payoffs[(action1, action2)]
        
        reward_buffer_1[action1].append(reward1)
        reward_buffer_2[action2].append(reward2)

        max_reward1 = max(reward_buffer_1[action1]) if reward_buffer_1[action1] else reward1
        max_reward2 = max(reward_buffer_2[action2]) if reward_buffer_2[action2] else reward2

        # Standard Q-learning update (as in the original code before leniency was added)
        q_table_1[action1] += alpha * (reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
        q_table_2[action2] += alpha * (reward2 + gamma * np.max(q_table_2) - q_table_2[action2])

        q_learning.decay_parameters()
    
    return player1_coop_probs, player2_coop_probs


def multi_plot(temperature, ax):
    alpha = 0.003
    gamma = 0 # No discounting for matrix games usually
    episodes = 10000
    runs_per_start_point = 10 # Number of runs to average for each starting point
    
    boltzman_q = BoltzmannQLearning(temperature=temperature, temperature_min=temperature, temperature_decay=0.9999, alpha=alpha)
    game = PrisonnersDilemma()

    # Define fixed starting points (Q-values for [Cooperate, Defect])
    fixed_start_points = [
        ([1, 0], [0, 1]), # P1: Coop, P2: Defect
        ([0.5, 0], [0, 0.5]), # P1: Coop, P2: Defect
        ([0, 1], [1, 0]), # P1: Defect, P2: Coop
        ([0, 0.5], [0.5, 0]), # P1: Defect, P2: Coop
        ([0, 0], [0, 0]), # Neutral start
        ([1, 0], [1, 0]), # Both cooperate
    ]

    all_avg_player1_probs = []
    all_avg_player2_probs = []

    for start_q1, start_q2 in fixed_start_points:
        runs_player1_probs = []
        runs_player2_probs = []
        for _ in range(runs_per_start_point):
            p1_probs, p2_probs = run_single_experiment(
                q_learning=boltzman_q,
                episodes=episodes,
                matrix_game=game,
                alpha=alpha,
                gamma=gamma,
                start_q_table_1=start_q1,
                start_q_table_2=start_q2
            )
            runs_player1_probs.append(p1_probs)
            runs_player2_probs.append(p2_probs)
        
        # Calculate the average trajectory for this starting point
        avg_p1_probs = np.mean(runs_player1_probs, axis=0)
        avg_p2_probs = np.mean(runs_player2_probs, axis=0)
        
        all_avg_player1_probs.append(avg_p1_probs)
        all_avg_player2_probs.append(avg_p2_probs)

    # Pass the list of average trajectories to the plotting function
    plot_rep_dynamics_probability(all_avg_player1_probs, all_avg_player2_probs, game, q_learning=boltzman_q, ax=ax)
    ax.set_title(f"Avg Trajectories (Temp = {temperature})")


if __name__ == "__main__":
    temperatures = [1, 0.5, 0.1]  # works with any number of items
    num_temps = len(temperatures)

    max_cols = min(3, num_temps)
    cols = max_cols
    rows = (num_temps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    # Normalize axes to be a flat array no matter what
    if num_temps == 1:
        axes = [axes]  # single Axes object -> wrap in list
    else:
        axes = axes.flatten()

    for i, temp in enumerate(temperatures):
        multi_plot(temp, ax=axes[i])

    # Hide any unused subplots (only applies when num_temps < rows * cols)
    for j in range(num_temps, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boltzmann Q-Learning: Average Cooperation Probability Trajectories from Fixed Starts", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly
    plt.show()