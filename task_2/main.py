from q_learning import QLearning, BoltzmannQLearning, EpsilonGreedyQLearning
from matrix_game import MatrixGame, PrisonnersDilemma, StagHunt, MatchingPennies
import numpy as np
import random
from plotting import plot_probabilities, plot_rep_dynamics_probability
import matplotlib.pyplot as plt
from collections import deque


def run_experiments(
    q_learning: QLearning,
    episodes: int,
    runs: int,
    matrix_game: MatrixGame,
    alpha: float,
    gamma: float,
):
    all_player1_probs = []
    all_player2_probs = []
    
    for run in range(runs):
        # Initialize Q-tables with bias
        q_learning.reset_parameters()
        q_table_1 = np.zeros(2)
        q_table_2 = np.zeros(2)
        
        q_table_1 = [random.uniform(0, 1), random.uniform(0, 1)]  # Initial bias
        q_table_2 = [random.uniform(0, 1), random.uniform(0, 1)]  # Initial bias
        
        player1_coop_probs = []  # Store cooperation probabilities
        player2_coop_probs = []  # Store cooperation probabilities
        
        kappa = 5
        
        # Track recent rewards for leniency
        reward_buffer_1 = [deque(maxlen=kappa) for _ in range(2)]
        reward_buffer_2 = [deque(maxlen=kappa) for _ in range(2)]
        
        for episode in range(episodes):
            prob1 = q_learning.get_action_probabilities(q_table_1)[matrix_game.get_plotted_action()]  # Probability of cooperation
            prob2 = q_learning.get_action_probabilities(q_table_2)[matrix_game.get_plotted_action()]  # Probability of cooperation
            
            # Store probabilities
            player1_coop_probs.append(prob1)
            player2_coop_probs.append(prob2)
            
            # Choose actions based on 1D Q-values
            action1 = q_learning.choose_action(q_table_1)
            action2 = q_learning.choose_action(q_table_2)

            # Get rewards
            reward1, reward2 = matrix_game.payoffs[(action1, action2)]
            
            # Append rewards to buffer
            reward_buffer_1[action1].append(reward1)
            reward_buffer_2[action2].append(reward2)

            # Get max reward from buffer (lenient Q update)
            max_reward1 = max(reward_buffer_1[action1]) if reward_buffer_1[action1] else reward1
            max_reward2 = max(reward_buffer_2[action2]) if reward_buffer_2[action2] else reward2

            # Q-learning update rule - now updates directly on 1D Q-tables
            q_table_1[action1] += alpha * (reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
            q_table_2[action2] += alpha * (reward2 + gamma * np.max(q_table_2) - q_table_2[action2])
            
            # Decay temperature
            q_learning.decay_parameters()
        
        all_player1_probs.append(player1_coop_probs)
        all_player2_probs.append(player2_coop_probs)
    
    return all_player1_probs, all_player2_probs


def multi_plot(temperature, ax):
    alpha = 0.003
    boltzman_q = BoltzmannQLearning(temperature=temperature, temperature_min=temperature, temperature_decay=0.9999, alpha=alpha)
    epsilon_q = EpsilonGreedyQLearning(epsilon=0.2, min_epsilon=0.01, epsilon_decay=0.999)
    game = PrisonnersDilemma()
    
    player1_probs, player2_probs = run_experiments(
        q_learning=boltzman_q, 
        episodes=10000,
        runs=10,
        matrix_game=game,
        alpha=alpha,
        gamma=0
    )
    
    plot_rep_dynamics_probability(player1_probs, player2_probs, game, q_learning=boltzman_q, ax=ax)
    ax.set_title(f"Temp = {temperature}")


if __name__ == "__main__":
    temperatures = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # works with any number of items
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
        axes[i].set_title(f"Temp = {temp}")

    # Hide any unused subplots (only applies when num_temps < rows * cols)
    for j in range(num_temps, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Boltzmann Q-Learning: Cooperation Probabilities Across Temperatures", fontsize=11)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()