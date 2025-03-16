import numpy as np
import matplotlib.pyplot as plt

# # Prisoner's Dilemma payoff matrix (row player)
# payoff_matrix = {
#     ("C", "C"): (3, 3),
#     ("C", "D"): (0, 5),
#     ("D", "C"): (5, 0),
#     ("D", "D"): (1, 1),
# }

# # Actions
# actions = ["C", "D"]

# # Initialize Q-tables for both players
# Q1 = {a: {b: 0 for b in actions} for a in actions}
# Q2 = {a: {b: 0 for b in actions} for a in actions}

# Hyperparameters
alpha = 0.001
gamma = 1
tau = 1.0         # Initial temperature for exploration
tau_min = 0.1     # Minimum temperature
tau_decay = 0.995 # Decay rate
episodes = 10000

# # Stag Hunt
# STAG = 0
# HARE = 1
# ACTIONS = [STAG, HARE]

# # Payoff Matrix (R, S, T, P)
# PAYOFFS = {
#     (STAG, STAG): (2/3, 2/3),
#     (STAG, HARE): (0, 2/3),
#     (HARE, STAG): (2/3, 0),
#     (HARE, HARE): (1, 1)
# }

# Prisoner's Dilemma
COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT]

# Payoff Matrix (R, S, T, P)
PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1)
}

# # Matching Pennies
# HEADS = 0
# TAILS = 1
# ACTIONS = [HEADS, TAILS]

# # Payoff Matrix (R, S, T, P)
# PAYOFFS = {
#     (HEADS, HEADS): (0, 1),
#     (HEADS, TAILS): (1, 0),
#     (TAILS, HEADS): (1, 0),
#     (TAILS, TAILS): (0, 1)
# }

# Arrays to store actions and probabilities
player1_actions = []
player2_actions = []
player1_coop_probs = []  # Store cooperation probabilities
player2_coop_probs = []

# Initial bias for COOPERATE action (positive values favor cooperation)
player1_initial_bias = 2
player2_initial_bias = 0

# Initialize Q-tables with bias
q_table_1 = np.zeros(2)
q_table_2 = np.zeros(2)

# Apply bias directly to specific actions
q_table_1[COOPERATE] = player1_initial_bias
q_table_2[COOPERATE] = player2_initial_bias

def softmax(q_values, tau=1.0):
    """
    Compute softmax values for x with temperature tau.
    
    This was derived from paper 2 p. 5 Equation 4
    """
    # Subtract max value for numerical stability
    q_max = np.max(q_values)
    e_x = np.exp((q_values - q_max) / tau)
    return e_x / np.sum(e_x)

def choose_action_boltzmann(q_values, tau):
    """Select an action using Boltzmann (softmax) distribution."""
    probabilities = softmax(q_values, tau)
    return np.random.choice(ACTIONS, p=probabilities)

# Training loop
for episode in range(episodes):
    # Calculate probabilities using softmax - now directly on 1D q-tables
    prob1 = softmax(q_table_1, tau)[COOPERATE]  # Probability of cooperation
    prob2 = softmax(q_table_2, tau)[COOPERATE]  # Probability of cooperation
    
    # Store probabilities
    player1_coop_probs.append(prob1)
    player2_coop_probs.append(prob2)
    
    # Choose actions based on 1D Q-values
    action1 = choose_action_boltzmann(q_table_1, tau)
    action2 = choose_action_boltzmann(q_table_2, tau)

    # Get rewards
    reward1, reward2 = PAYOFFS[(action1, action2)]

    # Q-learning update rule - now updates directly on 1D Q-tables
    q_table_1[action1] += alpha * (reward1 + gamma * np.max(q_table_1) - q_table_1[action1])
    q_table_2[action2] += alpha * (reward2 + gamma * np.max(q_table_2) - q_table_2[action2])

    # Decay temperature
    tau = max(tau * tau_decay, tau_min)
    
     # Store actions
    player1_actions.append(action1)
    player2_actions.append(action2)

# Print final Q-tables - now as 1D arrays
print("Q-table for Player 1:")
print(f"COOPERATE: {q_table_1[COOPERATE]}, DEFECT: {q_table_1[DEFECT]}")

print("\nQ-table for Player 2:")
print(f"COOPERATE: {q_table_2[COOPERATE]}, DEFECT: {q_table_2[DEFECT]}")

# Test final learned strategy
print("\nFinal Strategy (Greedy Policy):")
print("Player 1 chooses:", "Cooperate" if np.argmax(q_table_1) == COOPERATE else "Defect")
print("Player 2 chooses:", "Cooperate" if np.argmax(q_table_2) == COOPERATE else "Defect")

# Create trajectory plot (phase space)
plt.figure(figsize=(10, 8))

# Plot the entire trajectory
plt.plot(player1_coop_probs, player2_coop_probs, alpha=0.5, color='gray')

# Mark beginning and end points
plt.plot(player1_coop_probs[0], player2_coop_probs[0], 'go', markersize=10, label='Start')
plt.plot(player1_coop_probs[-1], player2_coop_probs[-1], 'ro', markersize=10, label='End')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Player 1's Probability of Cooperation")
plt.ylabel("Player 2's Probability of Cooperation")
plt.title("Learning Trajectory in Strategy Space")
plt.grid(True)
plt.legend()
plt.show()