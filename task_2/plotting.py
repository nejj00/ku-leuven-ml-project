import matplotlib.pyplot as plt
import numpy as np


def plot_probabilities(all_player1_probs, all_player2_probs):
    
    plt.figure(figsize=(10, 8))
    
    for i in range(0, len(all_player1_probs)):
        # Plot each trajectory with transparency
        plt.plot(all_player1_probs[i], all_player2_probs[i], alpha=0.3, color='gray')
    
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