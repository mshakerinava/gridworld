import sys
import json
import time
import random
import argparse
import numpy as np

from gridworld.gridworld import Gridworld
from gridworld.utils.dict import add_dicts
from gridworld.utils.logging import log, log_tabular
from gridworld.utils.prefix_arg import parse_known_args_with_prefix


argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--discount-factor', type=float, default=1.0)
args, _ = parser.parse_known_args(argv)

# set RNG seeds
random.seed(args.seed, version=2)
np.random.seed(random.randint(0, 2**32 - 1))

# create gridworld
env_kwargs = parse_known_args_with_prefix(argv, Gridworld.add_args)
env = Gridworld(env_id=args.env, **env_kwargs)
num_actions = env.num_actions

all_args = add_dicts(vars(args), env_kwargs)
print('parsed args = %s' % json.dumps(all_args, sort_keys=True, indent=4), file=sys.stderr)

# logging
keys = ['Total Reward', 'Total Discounted Reward', 'Time Steps']
formats = ['%9g', '%9g', '%9d']
log_tabular(vals=keys, file_csv=None, file_txt=None, write=False)

reward_list = []
total_rew = 0
disc_rew = 0
episode_step = 0
s = env.reset()
env.render()
while True:
    a = -1
    while not (0 <= a < num_actions):
        a = input('>> Action (integer in [0, %d]): ' % (num_actions - 1))
        try:
            a = int(a)
        except:
            a = -1
            continue
    episode_step += 1
    s_, rew, done = env.step(a)
    env.render()
    reward_list.append(rew)
    s = s_
    if done:
        break
total_rew = sum(reward_list)
for rew in reward_list[::-1]:
    disc_rew = rew + args.discount_factor * disc_rew
print()
log_tabular(vals=[total_rew, disc_rew, episode_step], file_csv=None, file_txt=None,
    keys=keys, formats=formats, write=False)
