import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the payoff matrix for the Prisoner's Dilemma
# payoff_matrix = np.array([
#     [[3, 3], [0, 5]],  # Cooperate vs Cooperate, Cooperate vs Defect
#     [[5, 0], [1, 1]]   # Defect vs Cooperate, Defect vs Defect
# ])

payoff_matrix = np.array([
    [[1, 1], [0, 2/3]],
    [[2/3, 0], [2/3, 2/3]]
])

# payoff_matrix = np.array([
#     [[0, 1], [1, 0]],
#     [[1, 0], [0, 1]]
# ])


p1_payoffs = payoff_matrix[:, :, 0]
p2_payoffs = payoff_matrix[:, :, 1]
print(p1_payoffs)
print(p2_payoffs)


# Set learning parameters
alpha = 1  # Learning rate
tau = 0.001  # Exploration temperature

def replicator_faq_rhs(t, x):
    p1_C = x[0]  # Probability of Player 1 cooperating
    p2_C = x[1]  # Probability of Player 2 cooperating
    
    # Expected payoffs - f_i(x) - (Ay)_i
    f1_C = p2_C * p1_payoffs[0, 0] + (1 - p2_C) * p1_payoffs[0, 1] # Expected payoff of cooperating for Player 1
    f1_D = p2_C * p1_payoffs[1, 0] + (1 - p2_C) * p1_payoffs[1, 1]
    f2_C = p1_C * p2_payoffs[0, 0] + (1 - p1_C) * p2_payoffs[1, 0]
    f2_D = p1_C * p2_payoffs[0, 1] + (1 - p1_C) * p2_payoffs[1, 1]
    
    # Average payoffs - f(x) - (x^T Ay)_i
    avg_f1 = p1_C * f1_C + (1 - p1_C) * f1_D
    avg_f2 = p2_C * f2_C + (1 - p2_C) * f2_D
    
    # Standard replicator with FAQ learning
    # dot_x_p1_C = p1_C * (1 - p1_C) * (f1_C - f1_D)
    # dot_x_p2_C = p2_C * (1 - p2_C) * (f2_C - f2_D)
    
    # Replicator dynamics with FAQ modifications
    # dp1_dt = alpha * p1 * (1 - p1) * ((u1_C - phi1) / tau - np.log(p1 / (1 - p1)))
    dot_x_p1_C = (alpha * p1_C / tau) * (f1_C - avg_f1) - alpha * p1_C * (np.log(p1_C) - (p1_C * np.log(p1_C) + (1 - p1_C) * np.log(1 - p1_C)))
    # dp2_dt = alpha * p2 * (1 - p2) * ((u2_C - phi2) / tau - np.log(p2 / (1 - p2)))
    dot_x_p2_C = (alpha * p2_C / tau) * (f2_C - avg_f2) - alpha * p2_C * (np.log(p2_C) - (p2_C * np.log(p2_C) + (1 - p2_C) * np.log(1 - p2_C)))
    
    return [dot_x_p1_C, dot_x_p2_C]

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
