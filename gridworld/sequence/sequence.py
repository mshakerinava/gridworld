import sys
import argparse
import importlib
from gridworld.utils.prefix_arg import parse_known_args_with_prefix


class Sequence:
    def __init__(self, seq_id, argv, prefix=''):
        self.Seq = importlib.import_module('gridworld.sequence.seqs.%s' % seq_id).Seq
        self.parsed_kwargs = parse_known_args_with_prefix(argv, self.Seq.add_args, prefix)
        self.seq = self.Seq(**self.parsed_kwargs)

    def __call__(self, t):
        return self.seq(t)
