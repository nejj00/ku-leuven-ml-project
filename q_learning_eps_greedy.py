import numpy as np

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
alpha = 0.01
gamma = 0.95
epsilon = 0.2
epsilon_decay = 0.999
min_epsilon = 0.01
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

# Matching Pennies
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

q_table_1 = np.zeros((2, 2))  # Player 1
q_table_2 = np.zeros((2, 2))  # Player 2

def choose_action_eps_greedy(q_table, last_opponent_action):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(ACTIONS)  # Explore
    return np.argmax(q_table[last_opponent_action])  # Exploit

last_action_1 = np.random.choice(ACTIONS)
last_action_2 = np.random.choice(ACTIONS)

# Training loop
for episode in range(episodes):
    action1 = choose_action_eps_greedy(q_table_1, last_action_2)
    action2 = choose_action_eps_greedy(q_table_2, last_action_1)

    # Get rewards
    reward1, reward2 = PAYOFFS[(action1, action2)]

    # Q-learning update rule
    q_table_1[last_action_2, action1] += alpha * (reward1 + gamma * np.max(q_table_1[action1]) - q_table_1[last_action_2, action1])
    q_table_2[last_action_1, action2] += alpha * (reward2 + gamma * np.max(q_table_2[action2]) - q_table_2[last_action_1, action2])
    
    last_action_1 = action1
    last_action_2 = action2

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Print final Q-tables
print("Q-table for Player 1:")
for a in ACTIONS:
    print(f"{a}: {q_table_1[a]}")

print("\nQ-table for Player 2:")
for a in ACTIONS:
    print(f"{a}: {q_table_2[a]}")


# Test final learned strategy
print("\nFinal Strategy (Greedy Policy):")
print("Player 1 chooses:", "Cooperate" if np.argmax(q_table_1[last_action_2]) == COOPERATE else "Defect")
print("Player 2 chooses:", "Cooperate" if np.argmax(q_table_2[last_action_1]) == COOPERATE else "Defect")