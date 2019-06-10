#!/usr/bin/env python3

import os
from pylab import *

def get_data(filename):
	data = open(filename).read().split('\n')[:-1]
	i = []
	mean = []
	std = []
	for s in data: 
		i.append(int(s.split('#')[1].split(': mean')[0]))
		mean.append(float(s.split('mean: ')[1].split(', std')[0]))
		std.append(float(s.split('std: ')[1]))
	return np.array(i), np.array(mean), np.array(std)

for env in 'Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2'.split(' '):
	bc_i, bc_mean, bc_std = get_data(os.path.join('logs', f'bc-{env}.log'))
	da_i, da_mean, da_std = get_data(os.path.join('logs', f'da-{env}.log'))
	figure()
	p = subplot(111)
	p.plot(bc_i, bc_mean, '-o', color='red', label='Behavioral Cloning')
	p.fill_between(bc_i, bc_mean - bc_std, bc_mean + bc_std, facecolor="lightcoral")
	p.plot(da_i, da_mean, '-o', color='blue', label='DAgger')
	p.fill_between(da_i, da_mean - da_std, da_mean + da_std, facecolor="lightblue")
	p.set_xlabel('# of Epochs')
	p.set_ylabel('Total Reward')
	p.legend()
	savefig(os.path.join('result', f'{env}.png'))