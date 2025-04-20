"""
Matrix Game implementations for multi-agent reinforcement learning.
Provides base class and implementations of common 2x2 games.
"""
from abc import ABC, abstractmethod
import numpy as np

class MatrixGame(ABC):
    """
    Abstract base class for 2x2 matrix games.
    
    A matrix game is defined by its payoff structure and actions.
    Each game has two players who simultaneously choose actions.
    """
    name: str
    payoffs: dict
    actions: list
    
    @abstractmethod
    def get_plotted_action(self):
        """
        Return the index of the action to track in plotting.
        
        Usually corresponds to the cooperate action in social dilemmas.
        """
        pass
    
    def get_payoff_matrix_player1(self):
        """Return the payoff matrix for player 1."""
        return np.array([[self.payoffs[(i, j)][0] for j in (0, 1)] 
                        for i in (0, 1)])
    
    def get_payoff_matrix_player2(self):
        """Return the payoff matrix for player 2."""
        return np.array([[self.payoffs[(i, j)][1] for j in (0, 1)] 
                        for i in (0, 1)])


class PrisonnersDilemma(MatrixGame):
    """
    Prisoner's Dilemma game implementation.
    
    A classic social dilemma where mutual cooperation is Pareto optimal,
    but mutual defection is the Nash equilibrium.
    """
    name = "Prisoner's Dilemma"
    COOPERATE = 0
    DEFECT = 1

    # Payoff Matrix in standard form (R, S, T, P)
    # R = Reward for mutual cooperation
    # S = Sucker's payoff (cooperate while other defects)
    # T = Temptation to defect (defect while other cooperates)
    # P = Punishment for mutual defection
    payoffs = {
        (COOPERATE, COOPERATE): (3, 3),  # (R, R)
        (COOPERATE, DEFECT): (0, 5),     # (S, T)
        (DEFECT, COOPERATE): (5, 0),     # (T, S)
        (DEFECT, DEFECT): (1, 1)         # (P, P)
    }
    
    actions = ["Cooperate", "Defect"]
    
    def get_plotted_action(self):
        """Return the cooperative action for plotting."""
        return self.COOPERATE


class StagHunt(MatrixGame):
    """
    Stag Hunt game implementation.
    
    A coordination game with two Nash equilibria:
    - Risk-dominant equilibrium (both hunt hare)
    - Payoff-dominant equilibrium (both hunt stag)
    """
    name = "Stag Hunt"
    STAG = 0  # Cooperative action
    HARE = 1  # Safe action
    
    # Payoff Matrix
    payoffs = {
        (STAG, STAG): (1, 1),       # High payoff coordination
        (STAG, HARE): (0, 2/3),     # Miscoordination (stag hunter gets nothing)
        (HARE, STAG): (2/3, 0),     # Miscoordination (stag hunter gets nothing)
        (HARE, HARE): (2/3, 2/3)    # Low payoff coordination
    }
    
    actions = ["Stag", "Hare"]
    
    def get_plotted_action(self):
        """Return the cooperative action (hunting stag) for plotting."""
        return self.STAG


class MatchingPennies(MatrixGame):
    """
    Matching Pennies game implementation.
    
    A zero-sum game with a unique mixed strategy Nash equilibrium.
    Player 1 wants to match, Player 2 wants to mismatch.
    """
    name = "Matching Pennies"
    HEADS = 0
    TAILS = 1

    # Payoff Matrix (zero-sum)
    payoffs = {
        (HEADS, HEADS): (1, -1),    # Player 1 wins
        (HEADS, TAILS): (-1, 1),    # Player 2 wins
        (TAILS, HEADS): (-1, 1),    # Player 2 wins
        (TAILS, TAILS): (1, -1)     # Player 1 wins
    }
    
    actions = ["Heads", "Tails"]
    
    def get_plotted_action(self):
        """Return the heads action for plotting."""
        return self.HEADS