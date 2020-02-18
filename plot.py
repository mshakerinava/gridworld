import os
import re
import json
import glob
import hashlib
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.ndimage.filters import gaussian_filter1d
from gridworld.utils.hash import hash_args


parser = argparse.ArgumentParser()
parser.add_argument('--exp-paths', type=str, nargs='*')
parser.add_argument('--metric', type=str, default='Total Reward')
parser.add_argument('--y-min', type=float)
parser.add_argument('--y-max', type=float)
parser.add_argument('--smoothing', type=float, default=0.001)
parser.add_argument('--only-mean', action='store_true')
args = parser.parse_args()
args_json = json.dumps(vars(args), sort_keys=True, indent=4)
print('args = %s' % args_json)
args_hash = hashlib.md5(str.encode(args_json)).hexdigest()[-8:]


def parse_single_seed(exp_path, seed):
    exp_path_seed = pjoin(exp_path, 'seed-%02d' % seed)
    log_csv_path = pjoin(exp_path_seed, 'log.csv')
    args_json_path = pjoin(exp_path_seed, 'args.json')
    log_arr = np.genfromtxt(log_csv_path, delimiter=',', dtype=None)
    header = np.array(log_arr[0], dtype=str)
    body = np.array(log_arr[1:], dtype=float)
    n = log_arr.shape[1]
    d = {'logs': {header[i]: body[:, i] for i in range(n)}}
    with open(args_json_path) as f:
        d['args'] = json.load(f)
    return d


def parse_all_seed(exp_path):
    d = {'seeds': {}}
    pattern = pjoin(exp_path, 'seed-*')
    path_list = glob.glob(pattern)
    for path in path_list:
        regex = re.fullmatch('.*seed-(\\d*)', path)
        seed = int(regex.group(1))
        d['seeds'][seed] = parse_single_seed(exp_path, seed)
    d['logs mean'] = {}
    d['logs std'] = {}
    for name in d['seeds'][seed]['logs'].keys():
        x = []
        for d_ in d['seeds'].values():
            x.append(d_['logs'][name])
        d['logs mean'][name] = np.mean(x, axis=0)
        d['logs std'][name] = np.std(x, axis=0)
    d['args'] = d['seeds'][seed]['args'] # just to ease access
    return d


fonts = {'fontname': 'CMU serif', 'fontsize': 10}
plt.style.use('seaborn')
fig = plt.figure(figsize=(5, 5))
args.exp_paths[0] = os.path.normpath(args.exp_paths[0])
env_id = args.exp_paths[0].split(os.sep)[1].split('__')[0]

save_path = pjoin('plots', '%s__#%s' % (env_id, args_hash))
os.makedirs(save_path, exist_ok=False)

plt.title(env_id, **fonts)
for exp_path in args.exp_paths:
    exp_path = os.path.normpath(exp_path)
    # it doesn't make much sense to compare in different envs
    assert env_id == exp_path.split(os.sep)[1].split('__')[0]
    d = parse_all_seed(exp_path)
    y_mid = gaussian_filter1d(d['logs mean'][args.metric], sigma=args.smoothing)
    x = np.arange(y_mid.shape[0])
    label = exp_path.split('__', 1)[1].replace('__', '')
    plt.plot(x, y_mid, label=label)
    if not args.only_mean:
        y_std = d['logs std'][args.metric]
        y_top = gaussian_filter1d(y_mid + y_std, sigma=args.smoothing)
        y_bot = gaussian_filter1d(y_mid - y_std, sigma=args.smoothing)
        plt.fill_between(x, y_bot, y_top, alpha=0.25)

plt.xlabel('Episode', **fonts)
plt.ylabel(args.metric, **fonts)
plt.legend(loc='best')
plt.xlim([x[0], x[-1] + 1])

if args.y_min is not None:
    plt.ylim(bottom=args.y_min)

if args.y_max is not None:
    plt.ylim(top=args.y_max)

with open(pjoin(save_path, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)
fig.savefig(pjoin(save_path, 'plot.svg'), bbox_inches='tight')
fig.savefig(pjoin(save_path, 'plot.pdf'), bbox_inches='tight')
fig.savefig(pjoin(save_path, 'plot.png'), bbox_inches='tight', dpi=300)
plt.close()
