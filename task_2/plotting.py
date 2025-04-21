import matplotlib.pyplot as plt
import numpy as np


def lenient_expected_utility(A, y, kappa):
    """
    Computes the lenient utility vector u for a player given their payoff matrix A and opponent policy y.
    """
    u = np.zeros_like(y)
    for i in range(len(u)):  # For each action i
        numerator = 0.0
        for j in range(len(y)):  # For each opponent action j
            a_ij = A[i, j]
            y_j = y[j]

            # Probabilities for leniency adjustments
            sum_leq = sum(y[m] for m in range(len(y)) if A[i, m] <= a_ij)
            sum_lt  = sum(y[m] for m in range(len(y)) if A[i, m] < a_ij)
            sum_eq  = sum(y[m] for m in range(len(y)) if A[i, m] == a_ij)

            # Avoid division by zero
            if sum_eq > 0:
                delta = (sum_leq ** kappa - sum_lt ** kappa) / sum_eq
                numerator += a_ij * y_j * delta

        u[i] = numerator
    return u


def replicator_lfaq_rhs(x, game, alpha=1.0, tau=0.005, kappa=3):
    p1_C = x[0]
    p2_C = x[1]

    x1 = np.array([p1_C, 1 - p1_C])
    x2 = np.array([p2_C, 1 - p2_C])

    p1_payoffs = game.get_payoff_matrix_player1()
    p2_payoffs = game.get_payoff_matrix_player2()

    # Lenient expected utilities
    f1_C = lenient_expected_utility(p1_payoffs, x2, kappa)
    f2_C = lenient_expected_utility(p2_payoffs.T, x1, kappa)  # B.T because column player

    # Average payoffs
    avg_f1 = np.dot(x1, f1_C)
    avg_f2 = np.dot(x2, f2_C)

    # Replicator dynamics with entropy correction (Boltzmann-like)
    dot_x1_C = (alpha * x1[0] / tau) * (f1_C[0] - avg_f1) - alpha * x1[0] * (np.log(x1[0]) - np.dot(x1, np.log(x1)))
    dot_x2_C = (alpha * x2[0] / tau) * (f2_C[0] - avg_f2) - alpha * x2[0] * (np.log(x2[0]) - np.dot(x2, np.log(x2)))

    return [dot_x1_C, dot_x2_C]


def replicator_faq_rhs(x, game, alpha = 1, tau = 0.005):
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


def plot_probabilities(all_player1_probs, all_player2_probs, ax):
    
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


def plot_avg_trajectories(all_avg_player1_probs, all_avg_player2_probs, game, ax):
    """Plots multiple average learning trajectories."""
    
    num_trajectories = len(all_avg_player1_probs)
    # Use a colormap to distinguish trajectories if there are many
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories)) 

    plotted_action_name = game.actions[game.get_plotted_action()]

    for i in range(num_trajectories):
        avg_p1_probs = all_avg_player1_probs[i]
        avg_p2_probs = all_avg_player2_probs[i]
        
        # Plot each average trajectory
        ax.plot(avg_p1_probs, avg_p2_probs, color=colors[i], linewidth=1.5, label=f"Start {i+1}")
        
        # Mark start and end points for each average trajectory
        ax.plot(avg_p1_probs[0], avg_p2_probs[0], 'o', color=colors[i], markersize=8, markeredgecolor='black') # Start
        ax.plot(avg_p1_probs[-1], avg_p2_probs[-1], 'X', color=colors[i], markersize=10, markeredgecolor='black') # End

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Player 1's Probability of {plotted_action_name}")
    ax.set_ylabel(f"Player 2's Probability of {plotted_action_name}")
    # Title is set in main.py now
    # ax.set_title("Average Learning Trajectories from Fixed Starts") 
    ax.grid(True)
    # ax.legend() # Legend might get crowded, consider omitting or placing outside


def plot_replicator_dynamics(game, q_learning, ax):
    # Create a grid of initial conditions
    p1_vals = np.linspace(0, 1, 10)  # Avoid exactly 0 or 1 for log function
    p2_vals = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(p1_vals, p2_vals)
    U, V = np.zeros_like(X), np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dxdt = replicator_faq_rhs(game=game, x=[X[i, j], Y[i, j]], alpha=q_learning.alpha, tau=q_learning.temperature)
            U[i, j], V[i, j] = dxdt

    # Plot the phase portrait
    ax.quiver(X, Y, U, V, color='blue', alpha=0.3)


def plot_rep_dynamics_probability(all_avg_player1_probs, all_avg_player2_probs, game, q_learning, ax):
    """Plots average trajectories overlayed on replicator dynamics."""
    plot_avg_trajectories(all_avg_player1_probs, all_avg_player2_probs, game, ax)
    plot_replicator_dynamics(game, q_learning, ax)
    # Add a combined legend if desired, or rely on start/end markers
    handles, labels = ax.get_legend_handles_labels()
    # Example: Place legend outside plot if too crowded
    # ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))