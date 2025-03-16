import numpy as np
import matplotlib.pyplot as plt
import random

# Hyperparameters
alpha = 0.001
gamma = 1
tau = 1.0         # Initial temperature for exploration
tau_min = 0.5     # Minimum temperature
tau_decay = 0.9999 # Decay rate
episodes = 10000
num_runs = 20  # Number of independent runs for empirical trajectories

COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT]

# Payoff Matrix (Prisoner's Dilemma)
PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

def softmax(q_values, tau=1.0):
    q_max = np.max(q_values)
    e_x = np.exp((q_values - q_max) / tau)
    return e_x / np.sum(e_x)

def choose_action_boltzmann(q_values, tau):
    probabilities = softmax(q_values, tau)
    return np.random.choice(ACTIONS, p=probabilities)

# Store multiple learning trajectories
all_player1_probs = []
all_player2_probs = []

plt.figure(figsize=(10, 8))

for run in range(num_runs):
    tau = 1.0  # Reset temperature for each run
    player1_coop_probs = []
    player2_coop_probs = []

    # Initialize Q-tables
    q_table_1 = np.zeros(2)
    q_table_2 = np.zeros(2)

    q_table_1 = [random.uniform(0, 1), random.uniform(0, 1)]  # Initial bias
    q_table_2 = [random.uniform(0, 1), random.uniform(0, 1)]  # Initial bias

    for episode in range(episodes):
        prob1 = softmax(q_table_1, tau)[COOPERATE]
        prob2 = softmax(q_table_2, tau)[COOPERATE]

        player1_coop_probs.append(prob1)
        player2_coop_probs.append(prob2)

        action1 = choose_action_boltzmann(q_table_1, tau)
        action2 = choose_action_boltzmann(q_table_2, tau)

        reward1, reward2 = PAYOFFS[(action1, action2)]

        q_table_1[action1] += alpha * (reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
        q_table_2[action2] += alpha * (reward2 + gamma * np.max(q_table_2) - q_table_2[action2])

        tau = max(tau * tau_decay, tau_min)

    all_player1_probs.append(player1_coop_probs)
    all_player2_probs.append(player2_coop_probs)

    # Plot each trajectory with transparency
    plt.plot(player1_coop_probs, player2_coop_probs, alpha=0.3, color='gray')

# Convert lists to arrays
all_player1_probs = np.array(all_player1_probs)
all_player2_probs = np.array(all_player2_probs)

# Compute mean trajectory
mean_player1_probs = np.mean(all_player1_probs, axis=0)
mean_player2_probs = np.mean(all_player2_probs, axis=0)

# Overlay mean trajectory in bold
plt.plot(mean_player1_probs, mean_player2_probs, color='blue', linewidth=2, label="Mean Trajectory")

# Mark start and end points
plt.plot(mean_player1_probs[0], mean_player2_probs[0], 'go', markersize=10, label="Start")
plt.plot(mean_player1_probs[-1], mean_player2_probs[-1], 'ro', markersize=10, label="End")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Player 1's Probability of Cooperation")
plt.ylabel("Player 2's Probability of Cooperation")
plt.title("Multiple Learning Trajectories & Mean Strategy Evolution")
plt.grid(True)
plt.legend()
plt.show()
