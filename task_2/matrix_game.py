class MatrixGame:
    name: str
    payoffs: dict
    actions: list

class PrisonnersDilemma(MatrixGame):
    name = "Prisoner's Dilemma"
    COOPERATE = 0
    DEFECT = 1

    # Payoff Matrix (R, S, T, P)
    PAYOFFS = {
        (COOPERATE, COOPERATE): (3, 3),
        (COOPERATE, DEFECT): (0, 5),
        (DEFECT, COOPERATE): (5, 0),
        (DEFECT, DEFECT): (1, 1)
    }
    
    actions = ["Cooperate", "Defect"]
    
class StagHunt(MatrixGame):
    name = "StagHunt"
    STAG = 0
    HARE = 1
    ACTIONS = [STAG, HARE]

    # Payoff Matrix (R, S, T, P)
    PAYOFFS = {
        (STAG, STAG): (2/3, 2/3),
        (STAG, HARE): (0, 2/3),
        (HARE, STAG): (2/3, 0),
        (HARE, HARE): (1, 1)
    }
    
    actions = ["Stag", "Hare"]
    
class MatchingPennies(MatrixGame):
    name = "Matching Pennies"
    HEADS = 0
    TAILS = 1

    # Payoff Matrix (R, S, T, P)
    PAYOFFS = {
        (HEADS, HEADS): (0, 1),
        (HEADS, TAILS): (1, 0),
        (TAILS, HEADS): (1, 0),
        (TAILS, TAILS): (0, 1)
    }
    
    actions = ["Heads", "Tails"]