import os
import sys
import json
import time
import shutil
import random
import argparse
import numpy as np
from os.path import join as pjoin

from gridworld.agent import Agent
from gridworld.gridworld import Gridworld
from gridworld.utils.hash import hash_args
from gridworld.utils.logging import log, log_tabular

argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CliffWalking')
parser.add_argument('--algo', type=str, default='Q-Learning')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--ckpt-period', type=int, default=1000)
parser.add_argument('--num-episodes', type=int, default=1000)
parser.add_argument('--discount-factor', type=float, default=0.99)
args, _ = parser.parse_known_args(argv)

# set RNG seeds
random.seed(args.seed, version=2)
np.random.seed(random.randint(0, 2**32 - 1))

# create gridworld
env = Gridworld(env_id=args.env)
num_actions = env.D * 2

# create agent
agent = Agent(args.algo, num_actions, args.discount_factor, argv)

# calculate hash of all arguments
all_args = dict(list(vars(args).items()) + list(agent.parsed_kwargs.items()))
print('parsed args = %s' % json.dumps(all_args, sort_keys=True, indent=4), file=sys.stderr)
args_hash = hash_args(all_args)

# create output folder for experiment
save_path = 'experiments/%s__%s__#%s/seed-%02d/' % (args.env, args.algo, args_hash, args.seed)
os.makedirs(save_path, exist_ok=False)

# open log files
file_log_txt = open(pjoin(save_path, 'log.txt'), 'w')
file_log_csv = open(pjoin(save_path, 'log.csv'), 'w')

# save all arguments to `args.json`
with open(pjoin(save_path, 'args.json'), 'w') as f:
    json.dump(all_args, f, sort_keys=True, indent=4)

# logging
keys = ['Episode', 'Total Reward', 'Total Discounted Reward', 'Time Steps']
formats = ['%9d', '%9g', '%9g', '%9d']
log_tabular(vals=keys, file_csv=file_log_csv, file_txt=file_log_txt)

global_step = 0
for episode_t in range(args.num_episodes):
    reward_list = []
    total_rew = 0
    disc_rew = 0
    episode_step = 0
    s = env.reset()
    while True:
        a = agent.act(s)
        episode_step += 1
        global_step += 1
        s_, rew, done = env.step(a)
        reward_list.append(rew)
        agent.learn(s, a, rew, s_, done)
        if global_step % args.ckpt_period == 0:
            agent.save(path=save_path + 'agent-%dk.pkl' % (global_step // args.ckpt_period))
        s = s_
        if done:
            break
    total_rew = sum(reward_list)
    for rew in reward_list[::-1]:
        disc_rew = rew + args.discount_factor * disc_rew
    log_tabular(vals=[episode_t, total_rew, disc_rew, episode_step], file_csv=file_log_csv, file_txt=file_log_txt,
        keys=keys, formats=formats)

agent.save(path=save_path + 'agent-final.pkl')

#-- log learned policy --#
file_policy_txt = open(pjoin(save_path, 'policy.txt'), 'w')
actions = ['D', 'U', 'R', 'L']
for row in range(env.spec.size()[0]):
    for col in range(env.spec.size()[1]):
        env.pos = np.array([row, col])
        s = env.get_state()
        a_star = agent.best_action(s)
        a_char = actions[a_star]
        log(a_char, file=file_policy_txt, end='')
    log('', file=file_policy_txt)
#------------------------#

# close log files
file_log_txt.close()
file_log_csv.close()
