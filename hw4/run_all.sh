#!/usr/bin/env bash

##########
### Q1 ###
##########

python3 main.py q1 --exp_name exp

##########
### Q2 ###
##########

python3 main.py q2 --exp_name exp

###########
### Q3a ###
###########

python3 main.py q3 --exp_name default
python3 plot.py --exps HalfCheetah_q3_default --save HalfCheetah_q3_default

###########
### Q3b ###
###########

python3 main.py q3 --exp_name action128 --num_random_action_selection 128
python3 main.py q3 --exp_name action4096 --num_random_action_selection 4096
python3 main.py q3 --exp_name action16384 --num_random_action_selection 16384
python3 plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions

python3 main.py q3 --exp_name horizon10 --mpc_horizon 10
python3 main.py q3 --exp_name horizon15 --mpc_horizon 15
python3 main.py q3 --exp_name horizon20 --mpc_horizon 20
python3 plot.py --exps HalfCheetah_q3_horizon10 HalfCheetah_q3_horizon15 HalfCheetah_q3_horizon20 --save HalfCheetah_q3_mpc_horizon

python3 main.py q3 --exp_name layers1 --nn_layers 1
python3 main.py q3 --exp_name layers2 --nn_layers 2
python3 main.py q3 --exp_name layers3 --nn_layers 3
python3 plot.py --exps HalfCheetah_q3_layers1 HalfCheetah_q3_layers2 HalfCheetah_q3_layers3 --save HalfCheetah_q3_nn_layers
