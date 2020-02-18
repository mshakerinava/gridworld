import sys
import pickle
import importlib
from gridworld.utils.prefix_arg import parse_known_args_with_prefix


class Agent:
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __init__(self, algo_id, num_actions, discount_factor, argv, prefix=''):
        self.num_actions = num_actions
        self.Algo = importlib.import_module('gridworld.agent.algos.%s' % algo_id).Algo

        self.parsed_kwargs = parse_known_args_with_prefix(argv, self.Algo.add_args, prefix)
        self.algo = self.Algo(num_actions, discount_factor, argv, prefix, **self.parsed_kwargs)
        self.parsed_kwargs = dict(list(self.parsed_kwargs.items()) + list(self.algo.parsed_kwargs.items()))

        # store counts inside `algo`
        self.algo.count_act = 0
        self.algo.count_learn = 0
        self.algo.count_sa = {}

    def _check_count(self, state):
        for action in range(self.num_actions):
            if (state, action) not in self.algo.count_sa:
                self.algo.count_sa[(state, action)] = 0

    def act(self, state, count=True):
        self._check_count(state)
        action = self.algo.act(state)
        if count:
            self.algo.count_act += 1
            self.algo.count_sa[(state, action)] += 1
        return action

    def best_action(self, state):
        self._check_count(state)
        return self.algo.best_action(state)

    def learn(self, state, action, reward, state_, terminal, count=True):
        self._check_count(state)
        self._check_count(state_)
        self.algo.learn(state, action, reward, state_, terminal)
        if count:
            self.algo.count_learn += 1
