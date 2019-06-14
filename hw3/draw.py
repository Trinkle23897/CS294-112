#!/usr/bin/env python3

import os, pickle
from pylab import *

def get_data(filename):
	d = pickle.loads(open(filename,'rb').read())
	return d['time'], d['reward']

v_t, v_r = get_data(os.path.join('data', 'vanilla_PongNoFrameskip-v4.pkl'))
d_t, d_r = get_data(os.path.join('data', 'double_PongNoFrameskip-v4.pkl'))
figure()
p = subplot(111)
p.plot(v_t, v_r, '-', label='Vanilla DQN')
p.plot(d_t, d_r, '-', label='Double DQN')
p.set_xlabel('# of Epochs')
p.set_ylabel('Total Reward')
p.legend()
ticklabel_format(style='sci', axis='x', scilimits=(0,0))
savefig(os.path.join('result', 'PongNoFrameskip-v4.png'))