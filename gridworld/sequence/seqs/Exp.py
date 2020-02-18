import numpy as np

class Seq:
    @staticmethod
    def add_args(arg_parser, prefix=''):
        arg_parser.add_argument('--%srate' % prefix, type=float, required=True, help='exp(-rate * t)')

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, t):
        return np.exp(-self.rate * t)
