import argparse


def remove_prefix(kwargs, prefix):
    assert len(prefix) > 0
    ret = {}
    for key, val in kwargs.items():
        if key.startswith(prefix):
            ret[key[len(prefix):]] = val
    return ret


def parse_known_args_with_prefix(argv, add_args_func, prefix=''):
    arg_parser = argparse.ArgumentParser(allow_abbrev=False)
    add_args_func(arg_parser, prefix)
    args, _ = arg_parser.parse_known_args(argv)
    args = vars(args)
    if prefix != '':
        prefix = prefix.replace('-', '_')
        args = remove_prefix(args, prefix)
    return args
