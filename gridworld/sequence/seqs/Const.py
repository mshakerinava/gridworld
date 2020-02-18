import numpy as np

class Seq:
    @staticmethod
    def add_args(arg_parser, prefix=''):
        arg_parser.add_argument('--%svalue' % prefix, type=float, required=True)

    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return self.value
