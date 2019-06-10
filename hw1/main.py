#!/usr/bin/env python3

import os, gym, pickle, tf_util, argparse, load_policy
import numpy as np
import tensorflow as tf
from tensorflow import keras

class ReplayMemory(object):
	def __init__(self, state_shape, action_shape, init_path):
		data = pickle.loads(open(init_path, 'rb').read())
		self.s = data['observations'].reshape([-1] + list(state_shape))
		self.a = data['actions'].reshape([-1] + list(action_shape))
		self.capacity = self.s.shape[0]
		self.pos = 0

	def add(self, s, a):
		self.s[self.pos] = s
		self.a[self.pos] = a
		self.pos = (self.pos + 1) % self.capacity

	def sample(self, sample_size):
		i = sample(range(0, self.capacity), sample_size)
		return self.s[i], self.a[i]

class Network(object):
	def __init__(self, args):
		self.model = keras.Sequential([
			keras.layers.Flatten(input_shape=args.state_shape),
			keras.layers.Dense(128, activation=tf.nn.relu),
			keras.layers.Dense(256, activation=tf.nn.relu),
			keras.layers.Dense(64, activation=tf.nn.relu),
			keras.layers.Dense(args.action_shape[0], activation=None)
		])
		self.model.compile(optimizer=keras.optimizers.Adam(args.lr), loss='mean_squared_error')
		self.batch_size = args.batch_size

	def learn(self, s, a):
		return self.model.fit(s, a, epochs=1, batch_size=self.batch_size)

	def act(self, s):
		return self.model.predict(s)

def run_episode(env, policy, timeout):
	state = []
	obs = env.reset()
	done = False
	steps = 0
	totalr = 0
	while not done and steps < timeout:
		state.append(obs)
		action = policy(obs[None])
		obs, r, done, _ = env.step(action)
		steps += 1
		totalr += r
	return state, totalr

def bc(args):
	for i in range(args.epoch):
		args.nn.learn(args.mem.s, args.mem.a)
		returns = []
		for j in range(args.num_rollouts):
			state, r = run_episode(args.env, args.nn.act, args.max_timesteps)
			returns.append(r)
		with open(args.log_file, 'a+') as f:
			f.write(f'#{i}: mean: {np.mean(returns)}, std: {np.std(returns)}\n')
		print(f'#{i}: mean: {np.mean(returns)}, std: {np.std(returns)}')

def da(args):
	for i in range(args.epoch):
		args.nn.learn(args.mem.s, args.mem.a)
		returns = []
		obs = []
		for j in range(args.num_rollouts):
			state, r = run_episode(args.env, args.nn.act, args.max_timesteps)
			returns.append(r)
			obs.extend(state)
		with open(args.log_file, 'a+') as f:
			f.write(f'#{i}: mean: {np.mean(returns)}, std: {np.std(returns)}\n')
		print(f'#{i}: mean: {np.mean(returns)}, std: {np.std(returns)}')
		obs = np.array(obs)
		act = args.policy_fn(obs)
		for i in range(obs.shape[0]):
			args.mem.add(obs[i], act[i])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('type', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument("--epoch", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument('--num_rollouts', type=int, default=20,
						help='Number of expert roll outs')
	args = parser.parse_args()
	args.expert_policy = os.path.join('experts', f'{args.envname}.pkl')
	args.expert_data = os.path.join('expert_data', f'{args.envname}.pkl')
	args.policy_fn = load_policy.load_policy(args.expert_policy)
	args.log_file = os.path.join('logs', args.type + '-' + args.envname + '.log')
	if os.path.exists(args.log_file):
		os.remove(args.log_file)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config):
		tf_util.initialize()
		args.env = gym.make(args.envname)
		args.state_shape = args.env.observation_space.shape
		args.action_shape = args.env.action_space.shape
		if args.max_timesteps == None:
			args.max_timesteps = args.env.spec.timestep_limit
		args.mem = ReplayMemory(args.state_shape, args.action_shape, args.expert_data)
		args.nn = Network(args)
		if args.type == 'bc':
			bc(args)
		elif args.type == 'da':
			da(args)
