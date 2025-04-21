"""
Visualization module for multi-agent reinforcement learning experiments.

This module provides functions for visualizing learning trajectories
and replicator dynamics in matrix games.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from matrix_game import MatrixGame
from q_learning import QLearning


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


def replicator_dynamics_equation_lenient(x, game, alpha=1.0, tau=0.005, kappa=5):
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


def replicator_dynamics_equation(x, game, alpha=1, tau=0.1):
    """
    Calculate the right-hand side of the standard replicator dynamics equation.
    
    Args:
        x: Current state [p1_coop_prob, p2_coop_prob]
        game: Matrix game instance
        alpha: Learning rate (scaling factor)
        tau: Temperature parameter (not used in standard replicator dynamics)
        
    Returns:
        List containing the derivatives [dp1/dt, dp2/dt]
    """
    p1_C = x[0]  # Probability of Player 1 cooperating
    p2_C = x[1]  # Probability of Player 2 cooperating
    
    # Avoid numerical issues with probabilities very close to 0 or 1
    epsilon = 1e-10
    p1_C = np.clip(p1_C, epsilon, 1 - epsilon)
    p2_C = np.clip(p2_C, epsilon, 1 - epsilon)
    
    # Get payoff matrices
    p1_payoffs = game.get_payoff_matrix_player1()
    p2_payoffs = game.get_payoff_matrix_player2()
    
    # Expected payoffs for each action
    f1_C = p2_C * p1_payoffs[0, 0] + (1 - p2_C) * p1_payoffs[0, 1]  # Expected payoff of cooperating for Player 1
    f1_D = p2_C * p1_payoffs[1, 0] + (1 - p2_C) * p1_payoffs[1, 1]  # Expected payoff of defecting for Player 1
    f2_C = p1_C * p2_payoffs[0, 0] + (1 - p1_C) * p2_payoffs[1, 0]  # Expected payoff of cooperating for Player 2
    f2_D = p1_C * p2_payoffs[0, 1] + (1 - p1_C) * p2_payoffs[1, 1]  # Expected payoff of defecting for Player 2
    
    # Average payoffs
    avg_f1 = p1_C * f1_C + (1 - p1_C) * f1_D
    avg_f2 = p2_C * f2_C + (1 - p2_C) * f2_D
    
    # # Entropy terms for the FAQ dynamics
    # entropy_p1 = np.log(p1_C) - (p1_C * np.log(p1_C) + (1 - p1_C) * np.log(1 - p1_C))
    # entropy_p2 = np.log(p2_C) - (p2_C * np.log(p2_C) + (1 - p2_C) * np.log(1 - p2_C))
    
    # # Replicator dynamics with FAQ modifications
    # dot_x_p1_C = (alpha * p1_C / tau) * (f1_C - avg_f1) - alpha * p1_C * entropy_p1
    # dot_x_p2_C = (alpha * p2_C / tau) * (f2_C - avg_f2) - alpha * p2_C * entropy_p2
    
    # Standard replicator dynamics equation: dx/dt = x * (fitness - average_fitness)
    # The alpha parameter scales the rate of change
    dot_x_p1_C = alpha * p1_C * (f1_C - avg_f1)
    dot_x_p2_C = alpha * p2_C * (f2_C - avg_f2)
    
    return [dot_x_p1_C, dot_x_p2_C]


def plot_trajectories(all_player1_probs, all_player2_probs, ax, game=None):
    """
    Plot multiple learning trajectories.
    
    Args:
        all_player1_probs: List of player 1's cooperation probability trajectories
        all_player2_probs: List of player 2's cooperation probability trajectories
        ax: Matplotlib axis to plot on
        game: Optional matrix game instance (for labels)
    """
    for i in range(len(all_player1_probs)):
        # Plot each trajectory with transparency
        ax.plot(all_player1_probs[i], all_player2_probs[i], alpha=0.5, color='black')
    
    # Convert lists to arrays
    all_player1_probs = np.array(all_player1_probs)
    all_player2_probs = np.array(all_player2_probs)

    # Compute mean trajectory if multiple trajectories provided
    if len(all_player1_probs) > 1:
        mean_player1_probs = np.mean(all_player1_probs, axis=0)
        mean_player2_probs = np.mean(all_player2_probs, axis=0)

        # Overlay mean trajectory in bold
        ax.plot(mean_player1_probs, mean_player2_probs, color='blue', linewidth=2, label="Mean Trajectory")

        # Mark start and end points
        ax.plot(mean_player1_probs[0], mean_player2_probs[0], 'go', markersize=10, label="Start")
        ax.plot(mean_player1_probs[-1], mean_player2_probs[-1], 'ro', markersize=10, label="End")

    # Configure plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')  # Ensure equal scaling of both axes
    
    # Set labels based on game if provided
    if game:
        plotted_action = game.actions[game.get_plotted_action()]
        ax.set_xlabel(f"Player 1's Probability of {plotted_action}")
        ax.set_ylabel(f"Player 2's Probability of {plotted_action}")
    else:
        ax.set_xlabel("Player 1's Probability")
        ax.set_ylabel("Player 2's Probability")
        
    ax.set_title("Learning Trajectories")
    ax.grid(True)
    ax.legend()


def plot_avg_trajectories(all_avg_player1_probs, all_avg_player2_probs, game, ax):
    """
    Plot average learning trajectories from multiple starting points.
    
    Args:
        all_avg_player1_probs: List of average player 1 trajectories
        all_avg_player2_probs: List of average player 2 trajectories
        game: Matrix game instance
        ax: Matplotlib axis to plot on
    """
    num_trajectories = len(all_avg_player1_probs)
    # Use a colormap to distinguish trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories)) 

    plotted_action_name = game.actions[game.get_plotted_action()]

    for i in range(num_trajectories):
        avg_p1_probs = all_avg_player1_probs[i]
        avg_p2_probs = all_avg_player2_probs[i]
        
        # Plot each average trajectory
        ax.plot(avg_p1_probs, avg_p2_probs, color=colors[i], linewidth=1, label=f"Start {i+1}")
        
        # Mark start and end points for each average trajectory
        ax.plot(avg_p1_probs[0], avg_p2_probs[0], 'o', color=colors[i], markersize=8, markeredgecolor='black')  # Start
        ax.plot(avg_p1_probs[-1], avg_p2_probs[-1], 'X', color=colors[i], markersize=10, markeredgecolor='black')  # End

    # Configure plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')  # Ensure equal scaling of both axes
    ax.set_xlabel(f"Player 1's Probability of {plotted_action_name}")
    ax.set_ylabel(f"Player 2's Probability of {plotted_action_name}")
    ax.grid(True)


def plot_replicator_dynamics(game, q_learning, ax, use_leniency=False):
    """
    Plot replicator dynamics vector field for a game.
    
    Args:
        game: Matrix game instance
        q_learning: Q-learning algorithm instance (for parameters)
        ax: Matplotlib axis to plot on
    """
    # Create a grid of initial conditions
    p1_vals = np.linspace(0.01, 0.99, 15)  # Avoid exactly 0 or 1 for log function
    p2_vals = np.linspace(0.01, 0.99, 15)
    X, Y = np.meshgrid(p1_vals, p2_vals)
    U, V = np.zeros_like(X), np.zeros_like(Y)

    # Calculate vector field
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if use_leniency:
                dxdt = replicator_dynamics_equation_lenient(
                    x=[X[i, j], Y[i, j]], 
                    game=game, 
                    # alpha=q_learning.alpha, 
                    # tau=getattr(q_learning, 'temperature', 0.1),  # Default if not Boltzmann
                    kappa=getattr(q_learning, 'kappa', 5)  # Example leniency parameter
                )
            else:
                dxdt = replicator_dynamics_equation(
                    x=[X[i, j], Y[i, j]], 
                    game=game, 
                    alpha=q_learning.alpha, 
                    tau=getattr(q_learning, 'temperature', 0.1)  # Default if not Boltzmann
                )

            U[i, j], V[i, j] = dxdt

    # Normalize vectors for better visualization
    magnitude = np.sqrt(U**2 + V**2)
    max_magnitude = np.max(magnitude)
    if max_magnitude > 0:  # Avoid division by zero
        U = U / max_magnitude
        V = V / max_magnitude

    # Plot the vector field
    ax.quiver(X, Y, U, V, color='blue', alpha=0.3)
    
    # Ensure equal scaling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')


def plot_combined_visualization(all_avg_player1_probs, all_avg_player2_probs, game, q_learning, ax, title=None, use_leniency=False):
    """
    Create a combined plot of trajectories and replicator dynamics.
    
    Args:
        all_avg_player1_probs: List of average player 1 trajectories
        all_avg_player2_probs: List of average player 2 trajectories
        game: Matrix game instance
        q_learning: Q-learning algorithm instance
        ax: Matplotlib axis to plot on
        title: Optional title for the plot
    """
    # Plot trajectories
    plot_avg_trajectories(all_avg_player1_probs, all_avg_player2_probs, game, ax)
    
    # Overlay replicator dynamics
    plot_replicator_dynamics(game, q_learning, ax, use_leniency=use_leniency)
    
    # Set title if provided
    if title:
        ax.set_title(title)