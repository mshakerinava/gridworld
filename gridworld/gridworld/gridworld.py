import re
import sys
import importlib
import numpy as np


class Gridworld:
    @staticmethod
    def add_args(arg_parser, prefix=''):
        arg_parser.add_argument('--%sdiag-moves' % prefix, action='store_true', help='Adds diagonal moves')
        arg_parser.add_argument('--%sno-move' % prefix, action='store_true', help='Adds a no-move action')

    def __init__(self, env_id, diag_moves=False, no_move=False):
        '''
        Loads environment `env_id`. `env_id` is case-sensitive.
        `diag_moves = True` adds diagonal moves.
        `no_move = True` adds a no-move action.
        '''
        self.env_id = env_id
        self.diag_moves = diag_moves
        self.no_move = no_move
        self.spec = importlib.import_module('gridworld.gridworld.envs.%s' % self.env_id)
        self.D = self.spec.size().shape[0]
        self.num_actions = (3 ** self.D - 1 if diag_moves else 2 * self.D) + (1 if no_move else 0)
        self.reset()

    def get_state(self):
        state = 0
        for i in range(self.D):
            state *= self.spec.size()[i]
            state += self.pos[i]
        return state

    def _move(self, dim, delta):
        self.pos[dim] += delta
        self.pos[dim] = max(0, min(self.spec.size()[dim] - 1, self.pos[dim]))

    def apply_action(self, action, stochastic):
        '''
        Applies action on agent without consideration of external
        forces such as wind and teleportation.
        '''
        if self.no_move and action == 0:
            return # if `self.no_move == True` then action 0 is no-move

        old_pos = self.pos.copy()
        if self.diag_moves:
            if not self.no_move:
                action += 1
            # apply stochasticity (TODO?)
            assert not stochastic, '`diag_moves` is currently not supported in stochastic gridworlds'
            # move
            for dim in range(self.D):
                delta = (-1 if action % 3 == 2 else action % 3)
                action //= 3
                self._move(dim, delta)
        else:
            if self.no_move:
                action -= 1
            # apply stochasticity
            if stochastic:
                opp_action = action + (1 if action % 2 == 0 else -1)
                action = opp_action
                while action == opp_action:
                    action = np.random.randint(self.D * 2)
            # move
            dim = action // 2
            delta = (1 if action % 2 == 0 else -1)
            self._move(dim, delta)

        # check wall
        if self.spec.wall(self.pos):
            self.pos = old_pos

    def step(self, action):
        '''
        There are 2D actions in total where D is the dimensionality
        of the gridworld. Action #2d adds one to dimension #d and
        action #2d+1 decreses one from dimension #d. Wind will also
        have an additional effect on the outcome. If the resulting
        state is blocked, the action will have no effect.
        '''
        assert 0 <= action < self.num_actions
        self.last_action = action
        old_pos = self.pos.copy()
        irregular, pos_ = self.spec.irregular(self.pos, action)
        if not irregular:
            self.apply_action(action, self.spec.stochastic(self.pos))
        else:
            self.pos = pos_
        # apply wind
        # TODO: what if wind moves us across a terminal state?
        wind_delta = self.spec.wind(old_pos)
        for i in range(self.D):
            if wind_delta[i] != 0:
                sign = (1 if wind_delta[i] < 0 else 0)
                magn = np.abs(wind_delta[i])
                for k in range(magn):
                    self.apply_action(2 * i + sign, stochastic=False)
        # apply teleportation
        self.pos = self.spec.teleport(self.pos)
        # calc reward
        reward = self.spec.reward(old_pos, action, self.pos)
        # check termination
        done = self.spec.terminal(self.pos)
        return self.get_state(), reward, done

    def randomize_pos(self):
        self.pos = np.zeros(self.D)
        for i in range(self.D):
            self.pos[i] = np.random.randint(self.spec.size()[i])

    def reset(self, random=False):
        '''
        Resets environment. If `random == True`, then the standard
        starting position(s) will be ignored and the agent's initial
        state will be uniformly randomized.
        '''
        if random:
            self.randomize_pos()
            while self.spec.terminal(self.pos):
                self.randomize_pos()
        else:
            self.pos = self.spec.start()
        self.last_action = None
        return self.get_state()

    def render(self):
        assert self.D == 2, '`render` is only supported in 2D gridworlds'
        if self.last_action is not None:
            print('  (Action #%d)' % self.last_action)
        self.desc = [['o'] * self.spec.size()[1]] * self.spec.size()[0]
        self.desc = np.array(self.desc)
        for i in range(self.spec.size()[0]):
            for j in range(self.spec.size()[1]):
                _pos = np.array([i, j])
                if self.spec.wall(_pos):
                    self.desc[i, j] = 'X'
                if self.spec.terminal(_pos):
                    self.desc[i, j] = 'T'
                if (self.pos == _pos).all():
                    print('@', end='')
                else:
                    print(self.desc[i, j], end='')
            print()
