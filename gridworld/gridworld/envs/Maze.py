import numpy as np

_MAP = [
    'OOOOOOOXT',
    'OOXOOOOXO',
    'SOXOOOOXO',
    'OOXOOOOOO',
    'OOOOOXOOO',
    'OOOOOOOOO'
]

def size():
    '''
    Returns a `numpy.ndarray` representing the size of the gridworld.
    '''
    num_rows = len(_MAP)
    num_cols = len(_MAP[0])
    return np.array([num_rows, num_cols])

def start():
    '''
    Returns a `numpy.ndarray` representing a starting position.
    '''
    return np.array([2, 0])

def terminal(pos):
    '''
    Checks if position `pos` is a terminal state.
    '''
    tok = _MAP[pos[0]][pos[1]]
    return tok == 'T'

def wind(pos):
    '''
    Returns the wind displacement affecting position `pos`.
    '''
    return np.zeros_like(size())

def wall(pos):
    '''
    Checks if position `pos` is blocked.
    '''
    tok = _MAP[pos[0]][pos[1]]
    return tok == 'X'

def reward(old_pos, action, new_pos):
    '''
    Returns reward for transition (`pos`, `action`, `new_pos`).
    '''
    tok = _MAP[new_pos[0]][new_pos[1]]
    return 1 if tok == 'T' else 0

def teleport(pos):
    '''
    Returns teleportation target for position `pos`.
    '''
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

def irregular(pos, action):
    '''
    Returns `True` only if taking action `action` at position `pos`
    does not follow the standard transition dynamics of a gridworld.
    The position resulting from the action will be returned as the
    second element. In other cases, `False, None` is returned.
    '''
    return False, None
