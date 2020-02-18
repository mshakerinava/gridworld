import numpy as np

def size():
    '''
    Returns a `numpy.ndarray` representing the size of the gridworld.
    '''
    return np.array([4, 12])

def start():
    '''
    Returns a `numpy.ndarray` representing a starting position.
    '''
    return np.array([3, 0])

def terminal(pos):
    '''
    Checks if position `pos` is a terminal state.
    '''
    return (pos == [3, 11]).all()

def wind(pos):
    '''
    Returns the wind displacement affecting position `pos`.
    '''
    return np.zeros_like(size())

def wall(pos):
    '''
    Checks if position `pos` is blocked.
    '''
    return False

def reward(old_pos, action, new_pos):
    '''
    Returns reward for transition (`pos`, `action`, `new_pos`).
    '''
    if old_pos[0] == 2 and 1 <= old_pos[1] <= 10 and action == 0:
        return -100
    return -1

def teleport(pos):
    '''
    Returns teleportation target for position `pos`.
    '''
    if pos[0] == 3 and 1 <= pos[1] <= 10:
        return start()
    return pos.copy()

def stochastic(pos):
    '''
    Returns `True` only if actions are stochastic at position `pos`.
    Stochastic actions give equal probability to all actions, except
    for the opposite of the selected action. For example, in a 2D
    gridworld, the action LEFT has equal probability of going LEFT,
    UP, or DOWN (but not RIGHT, which is the opposite). 
    '''
    return False
