from abc import abstractmethod
import numpy as np

class MatrixGame:
    name: str
    payoffs: dict
    actions: list
    
    @abstractmethod
    def get_plotted_action(self):
        """
        Return the action corresponding to the Nash equilibrium(NE) 
        """
        pass
    
    def get_payoff_matrix_player1(self):
        return np.array([[self.payoffs[(i, j)][0] for j in (0, 1)] 
                        for i in (0, 1)])
    
    def get_payoff_matrix_player2(self):
        return np.array([[self.payoffs[(i, j)][1] for j in (0, 1)] 
                        for i in (0, 1)])

class PrisonnersDilemma(MatrixGame):
    name = "Prisoner's Dilemma"
    COOPERATE = 0
    DEFECT = 1

    # Payoff Matrix (R, S, T, P)
    payoffs = {
        (COOPERATE, COOPERATE): (3, 3),
        (COOPERATE, DEFECT): (0, 5),
        (DEFECT, COOPERATE): (5, 0),
        (DEFECT, DEFECT): (1, 1)
    }
    
    actions = ["Cooperate", "Defect"]
    
    def get_plotted_action(self):
        return self.COOPERATE

class StagHunt(MatrixGame):
    name = "StagHunt"
    STAG = 0
    HARE = 1
    ACTIONS = [STAG, HARE]

    # Payoff Matrix (R, S, T, P)
    payoffs = {
        (STAG, STAG): (1, 1),
        (STAG, HARE): (0, 2/3),
        (HARE, STAG): (2/3, 0),
        (HARE, HARE): (2/3, 2/3)
    }
    
    actions = ["Stag", "Hare"]
    def get_plotted_action(self):
        return self.STAG

class MatchingPennies(MatrixGame):
    name = "Matching Pennies"
    HEADS = 0
    TAILS = 1

    # Payoff Matrix (R, S, T, P)
    payoffs = {
        (HEADS, HEADS): (0, 1),
        (HEADS, TAILS): (1, 0),
        (TAILS, HEADS): (1, 0),
        (TAILS, TAILS): (0, 1)
    }
    
    actions = ["Heads", "Tails"]
    
    def get_plotted_action(self):
        return self.HEADS