import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the payoff matrix for the Prisoner's Dilemma
payoff_matrix = np.array([
    [[3, 3], [0, 5]],  # Cooperate vs Cooperate, Cooperate vs Defect
    [[5, 0], [1, 1]]   # Defect vs Cooperate, Defect vs Defect
])

payoff_matrix = np.array([
    [[1, 1], [0, 2/3]],  # Cooperate vs Cooperate, Cooperate vs Defect
    [[2/3, 0], [2/3, 2/3]]   # Defect vs Cooperate, Defect vs Defect
])

# payoff_matrix = np.array([
#     [[0, 1], [1, 0]],  # Cooperate vs Cooperate, Cooperate vs Defect
#     [[1, 0], [0, 1]]   # Defect vs Cooperate, Defect vs Defect
# ])

# Set learning parameters
alpha = 1  # Learning rate
tau = 1  # Exploration temperature

def replicator_faq_rhs(t, x):
    p1 = x[0]  # Probability of Player 1 cooperating
    p2 = x[1]  # Probability of Player 2 cooperating
    
    # Expected payoffs
    u1_C = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    u1_D = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    u2_C = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    u2_D = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    
    # Average payoffs
    phi1 = p1 * u1_C + (1 - p1) * u1_D
    phi2 = p2 * u2_C + (1 - p2) * u2_D
    
    # Replicator dynamics with FAQ modifications
    # dp1_dt = alpha * p1 * (1 - p1) * ((u1_C - phi1) / tau - np.log(p1 / (1 - p1)))
    dp1_dt = (alpha * p1 / tau) * (u1_C - phi1) - alpha * p1 * (np.log(p1) - (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))
    # dp2_dt = alpha * p2 * (1 - p2) * ((u2_C - phi2) / tau - np.log(p2 / (1 - p2)))
    dp2_dt = (alpha * p2 / tau) * (u2_C - phi2) - alpha * p2 * (np.log(p2) - (p2 * np.log(p2) + (1 - p2) * np.log(1 - p2)))
    
    return [dp1_dt, dp2_dt]

# Create a grid of initial conditions
p1_vals = np.linspace(0.1, 0.9, 10)  # Avoid exactly 0 or 1 for log function
p2_vals = np.linspace(0.1, 0.9, 10)
X, Y = np.meshgrid(p1_vals, p2_vals)
U, V = np.zeros_like(X), np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        dxdt = replicator_faq_rhs(0, [X[i, j], Y[i, j]])
        U[i, j], V[i, j] = dxdt

# Plot the phase portrait
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, color='blue', alpha=0.6)
plt.xlabel("Player 1 Cooperation Probability")
plt.ylabel("Player 2 Cooperation Probability")
plt.title("Replicator Dynamics for FAQ Learning in Stag Hunt Game")
plt.show()
