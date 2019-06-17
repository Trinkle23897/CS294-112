#!/bin/bash

mkdir output

# problem 1
python3 train_policy.py 'pm-obs' --exp_name q1 --history 1 -lr 5e-5 -n 200 --num_tasks 4

# problem 2
python3 train_policy.py 'pm' --exp_name q2_mlp_1 --history 1 --discount 0.90 -lr 5e-4 -n 60
python3 train_policy.py 'pm' --exp_name q2_rnn_1 --history 1 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python3 train_policy.py 'pm' --exp_name q2_mlp_20 --history 20 --discount 0.90 -lr 5e-4 -n 60
python3 train_policy.py 'pm' --exp_name q2_rnn_20 --history 20 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python3 train_policy.py 'pm' --exp_name q2_mlp_50 --history 50 --discount 0.90 -lr 5e-4 -n 60
python3 train_policy.py 'pm' --exp_name q2_rnn_50 --history 50 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python3 train_policy.py 'pm' --exp_name q2_mlp_100 --history 100 --discount 0.90 -lr 5e-4 -n 60
python3 train_policy.py 'pm' --exp_name q2_rnn_100 --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent

# problem 3
python3 train_policy.py 'pm' --exp_name q3_rnn_50_1 --history 50 --discount 0.90 -lr 5e-4 -n 60 --generalized --granularity 1 --recurrent
python3 train_policy.py 'pm' --exp_name q3_rnn_50_2 --history 50 --discount 0.90 -lr 5e-4 -n 60 --generalized --granularity 2 --recurrent
python3 train_policy.py 'pm' --exp_name q3_rnn_50_4 --history 50 --discount 0.90 -lr 5e-4 -n 60 --generalized --granularity 4 --recurrent
python3 train_policy.py 'pm' --exp_name q3_rnn_50_5 --history 50 --discount 0.90 -lr 5e-4 -n 60 --generalized --granularity 5 --recurrent
python3 train_policy.py 'pm' --exp_name q3_rnn_50_10 --history 50 --discount 0.90 -lr 5e-4 -n 60 --generalized --granularity 10 --recurrent