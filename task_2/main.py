from q_learning import QLearning, BoltzmannQLearning
from matrix_game import MatrixGame, PrisonnersDilemma, StagHunt, MatchingPennies
import numpy as np
import random
from plotting import plot_probabilities

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
        player2_coop_probs = []
        
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

            # Q-learning update rule - now updates directly on 1D Q-tables
            q_table_1[action1] += alpha * (reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
            q_table_2[action2] += alpha * (reward2 + gamma * np.max(q_table_2) - q_table_2[action2])
            
            # q_table_1[action1] += alpha * (reward1 - q_table_1[action1])
            # q_table_2[action2] += alpha * (reward2 - q_table_2[action2])

            # Decay temperature
            q_learning.decay_parameters()
        
        all_player1_probs.append(player1_coop_probs)
        all_player2_probs.append(player2_coop_probs)
    
    return all_player1_probs, all_player2_probs



if __name__ == "__main__":
    
    boltzman_q = BoltzmannQLearning(1.0, 0.5, 0.9999)
    
    player1_probs, player2_probs = run_experiments(
        q_learning=boltzman_q, 
        episodes=10000,
        runs=20,
        matrix_game=PrisonnersDilemma(),
        alpha=0.001,
        gamma=1
    )
    
    plot_probabilities(player1_probs, player2_probs)