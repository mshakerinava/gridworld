import json
import hashlib


def hash_args(args_dict):
    NO_HASH = ['seed', 'ckpt_period', 'env', 'algo']
    args_dict = {x: args_dict[x] for x in args_dict if x not in NO_HASH}
    args_str = json.dumps(args_dict, sort_keys=True, indent=4)
    args_hash = hashlib.md5(str.encode(args_str)).hexdigest()[-8:]
    return args_hash
