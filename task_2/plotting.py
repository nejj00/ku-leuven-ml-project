import matplotlib.pyplot as plt
import numpy as np


# # Define the payoff matrix for the Prisoner's Dilemma
# payoff_matrix = np.array([
#     [[3, 3], [0, 5]],  # Cooperate vs Cooperate, Cooperate vs Defect
#     [[5, 0], [1, 1]]   # Defect vs Cooperate, Defect vs Defect
# ])


# p1_payoffs = payoff_matrix[:, :, 0]
# p2_payoffs = payoff_matrix[:, :, 1]
# print(p1_payoffs)
# print(p2_payoffs)


# # Set learning parameters
# alpha = 1  # Learning rate
# tau = 0.001  # Exploration temperature

def replicator_faq_rhs(x, game, alpha = 1, tau = 0.001):
    p1_C = x[0]  # Probability of Player 1 cooperating
    p2_C = x[1]  # Probability of Player 2 cooperating
    
    p1_payoffs = game.get_payoff_matrix_player1()
    p2_payoffs = game.get_payoff_matrix_player2()
    
    # Expected payoffs - f_i(x) - (Ay)_i
    f1_C = p2_C * p1_payoffs[0, 0] + (1 - p2_C) * p1_payoffs[0, 1] # Expected payoff of cooperating for Player 1
    f1_D = p2_C * p1_payoffs[1, 0] + (1 - p2_C) * p1_payoffs[1, 1]
    f2_C = p1_C * p2_payoffs[0, 0] + (1 - p1_C) * p2_payoffs[1, 0]
    f2_D = p1_C * p2_payoffs[0, 1] + (1 - p1_C) * p2_payoffs[1, 1]
    
    # Average payoffs - f(x) - (x^T Ay)_i
    avg_f1 = p1_C * f1_C + (1 - p1_C) * f1_D
    avg_f2 = p2_C * f2_C + (1 - p2_C) * f2_D
    
    # Replicator dynamics with FAQ modifications
    dot_x_p1_C = (alpha * p1_C / tau) * (f1_C - avg_f1) - alpha * p1_C * (np.log(p1_C) - (p1_C * np.log(p1_C) + (1 - p1_C) * np.log(1 - p1_C)))
    dot_x_p2_C = (alpha * p2_C / tau) * (f2_C - avg_f2) - alpha * p2_C * (np.log(p2_C) - (p2_C * np.log(p2_C) + (1 - p2_C) * np.log(1 - p2_C)))
    
    return [dot_x_p1_C, dot_x_p2_C]


def plot_probabilities(all_player1_probs, all_player2_probs):
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(len(all_player1_probs)):
        # Plot each trajectory with transparency
        ax.plot(all_player1_probs[i], all_player2_probs[i], alpha=0.5, color='black')
    
    # Convert lists to arrays
    all_player1_probs = np.array(all_player1_probs)
    all_player2_probs = np.array(all_player2_probs)

    # Compute mean trajectory
    mean_player1_probs = np.mean(all_player1_probs, axis=0)
    mean_player2_probs = np.mean(all_player2_probs, axis=0)

    # Overlay mean trajectory in bold
    ax.plot(mean_player1_probs, mean_player2_probs, color='blue', linewidth=2, label="Mean Trajectory")

    # Mark start and end points
    ax.plot(mean_player1_probs[0], mean_player2_probs[0], 'go', markersize=10, label="Start")
    ax.plot(mean_player1_probs[-1], mean_player2_probs[-1], 'ro', markersize=10, label="End")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Player 1's Probability of Cooperation")
    ax.set_ylabel("Player 2's Probability of Cooperation")
    ax.set_title("Multiple Learning Trajectories & Mean Strategy Evolution")
    ax.grid(True)
    ax.legend()
    
    return fig


def plot_replicator_dynamics(fig, game):
    # Create a grid of initial conditions
    p1_vals = np.linspace(0.1, 0.9, 10)  # Avoid exactly 0 or 1 for log function
    p2_vals = np.linspace(0.1, 0.9, 10)
    X, Y = np.meshgrid(p1_vals, p2_vals)
    U, V = np.zeros_like(X), np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dxdt = replicator_faq_rhs(game=game, x=[X[i, j], Y[i, j]], alpha=1, tau=0.001)
            U[i, j], V[i, j] = dxdt

    # Plot the phase portrait
    ax = fig.gca()
    
    ax.quiver(X, Y, U, V, color='blue', alpha=0.3)
    
    return fig


def plot_rep_dynamics_prabability(all_player1_probs, all_player2_probs, game):
    
    
    fig = plot_probabilities(all_player1_probs, all_player2_probs)
    fig = plot_replicator_dynamics(fig, game)
    
    return fig