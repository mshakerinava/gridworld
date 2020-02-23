def add_dicts(*args):
    ret = {}
    for x in args:
        ret = dict(list(ret.items()) + list(x.items()))
    return ret
