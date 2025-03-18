import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Payoff matrices for both players (A for Player 1, B for Player 2)
A = np.array([[3, 0],  
              [5, 1]])
B = np.array([[3, 5],  
              [0, 1]])

# Function to compute replicator dynamics for both players
def replicator_dynamics(t, xy):
    x_C, y_C = xy  # x_C and y_C represent the probability of cooperating for both players
    x_D = 1 - x_C  # Probability of defecting for Player 1
    y_D = 1 - y_C  # Probability of defecting for Player 2
    
    # Compute fitness for each strategy for Player 1
    f_C_x = A[0, 0] * y_C + A[0, 1] * y_D  # Expected payoff of cooperating for Player 1
    f_D_x = A[1, 0] * y_C + A[1, 1] * y_D  # Expected payoff of defecting for Player 1
    
    # Compute fitness for each strategy for Player 2
    f_C_y = B[0, 0] * x_C + B[0, 1] * x_D  # Expected payoff of cooperating for Player 2
    f_D_y = B[1, 0] * x_C + B[1, 1] * x_D  # Expected payoff of defecting for Player 2
    
    # Compute average fitness for both players
    f_avg_x = x_C * f_C_x + x_D * f_D_x
    f_avg_y = y_C * f_C_y + y_D * f_D_y
    
    # Replicator equations
    dx_C = x_C * (f_C_x - f_avg_x)
    dy_C = y_C * (f_C_y - f_avg_y)
    
    return [dx_C, dy_C]

# Create a grid of initial conditions
x_vals = np.linspace(0, 1, 20)
y_vals = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x_vals, y_vals)
U, V = np.zeros_like(X), np.zeros_like(Y)

# Compute vector field
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xC, yC = X[i, j], Y[i, j]
        dxC, dyC = replicator_dynamics(0, [xC, yC])
        U[i, j] = dxC
        V[i, j] = dyC

# Plot the vector field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, color="blue", angles='xy', scale_units='xy', scale=5.0, alpha=0.4)
plt.xlabel("Fraction of Cooperators - Player 1 (x_C)")
plt.ylabel("Fraction of Cooperators - Player 2 (y_C)")
plt.title("Replicator Dynamics of Prisoner's Dilemma (Both Players)")

# # Solve and plot sample trajectories
# initial_conditions = [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]
# t_span = [0, 10]
# t_eval = np.linspace(0, 10, 100)

# for x0 in initial_conditions:
#     sol = solve_ivp(replicator_dynamics, t_span, x0, t_eval=t_eval)
#     plt.plot(sol.y[0], sol.y[1], label=f'Init: {x0}')

plt.legend()
plt.show()
