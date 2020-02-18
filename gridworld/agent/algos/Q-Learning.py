from gridworld.sequence import Sequence
from gridworld.utils.prefix_arg import parse_known_args_with_prefix
import numpy as np


class Algo:
    @staticmethod
    def add_args(arg_parser, prefix=''):
        help_seq = 'a sequence, e.g., `Harmonic+Local` or `Const+Global`'
        arg_parser.add_argument('--%seps' % prefix, type=str, required=True,
            help='Probability of taking a random action (%s)' % help_seq)
        arg_parser.add_argument('--%sinit-q' % prefix, type=float, required=True,
            help='Initial value for q-value estimates')
        arg_parser.add_argument('--%slearning-rate' % prefix, type=str, required=True,
            help='The learning rate (%s)' % help_seq)

    def __init__(self, num_actions, discount_factor, argv, prefix, eps, init_q, learning_rate):
        self.q = {}
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.init_q = init_q

        self.eps_seq_id, self.eps_locality = [x.title() for x in eps.split('+')]
        assert self.eps_locality in ['Local', 'Global']
        self.eps = Sequence(self.eps_seq_id, argv, prefix + 'eps-')

        self.learning_rate_seq_id, self.learning_rate_locality = [x.title() for x in learning_rate.split('+')]
        assert self.learning_rate_locality in ['Local', 'Global']
        self.learning_rate = Sequence(self.learning_rate_seq_id, argv, prefix + 'learning-rate-')

        self.parsed_kwargs = {'eps_': self.eps.parsed_kwargs, 'learning_rate_': self.learning_rate.parsed_kwargs}

    def _check_q(self, state):
        if state not in self.q:
            self.q[state] = np.ones(self.num_actions) * self.init_q

    def _count_s(self, state):
        count = 0
        for action in range(self.num_actions):
            count += self.count_sa[(state, action)]
        return count

    def _get_eps(self, state):
        count_s = self._count_s(state)
        eps_global = self.eps(self.count_act)
        eps_local = self.eps(count_s)
        eps = (eps_global if self.eps_locality == 'Global' else eps_local)
        return eps

    def _get_learning_rate(self, state, action):
        lr_local = self.learning_rate(self.count_sa[(state, action)])
        lr_global = self.learning_rate(self.count_learn)
        learning_rate = (lr_global if self.learning_rate_locality == 'Global' else lr_local)
        return learning_rate

    def act(self, state):
        eps = self._get_eps(state)
        if np.random.binomial(1, eps) == 1:
            return np.random.randint(self.num_actions)
        return self.best_action(state)

    def best_action(self, state):
        self._check_q(state)
        return np.argmax(self.q[state])

    def learn(self, state, action, reward, state_, terminal):
        self._check_q(state)
        self._check_q(state_)
        err = reward + self.discount_factor * (0 if terminal else np.max(self.q[state_])) - self.q[state][action]
        learning_rate = self._get_learning_rate(state, action)
        self.q[state][action] += learning_rate * err
