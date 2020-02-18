import json
import time
import random
import argparse
import numpy as np

from hash import hash_args
from agent import Agent
from gridworld import Gridworld
from log_utils import log, log_tabular

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--agent', type=str, required=True)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
log('args = %s' % json.dumps(vars(args), sort_keys=True, indent=4), write=False)

# set RNG seeds
random.seed(args.seed, version=2)
np.random.seed(random.randint(0, 2**32 - 1))

# create gridworld
env = Gridworld(env_id=args.env)
num_actions = env.D * 2

# load agent
agent = Agent.load(path=args.agent)
print(agent.algo.eps)

# logging
keys = ['Total Reward', 'Total Discounted Reward', 'Time Steps']
formats = ['%9g', '%9g', '%9d']
log_tabular(vals=keys, file_csv=None, file_txt=None, write=False)

reward_list = []
total_rew = 0
disc_rew = 0
episode_step = 0
s = env.reset()
while True:
    a = agent.act(s)
    episode_step += 1
    s_, rew, done = env.step(a)
    env.render()
    time.sleep(0.1)
    reward_list.append(rew)
    s = s_
    if done:
        break
total_rew = sum(reward_list)
for rew in reward_list[::-1]:
    disc_rew = rew + agent.algo.discount_factor * disc_rew
print()
log_tabular(vals=[total_rew, disc_rew, episode_step], file_csv=None, file_txt=None,
    keys=keys, formats=formats, write=False)
