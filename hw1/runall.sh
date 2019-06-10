#!/bin/bash
python3 -u main.py bc Hopper-v2 &
python3 -u main.py da Hopper-v2
python3 -u main.py bc Ant-v2 &
python3 -u main.py da Ant-v2
python3 -u main.py bc HalfCheetah-v2 &
python3 -u main.py da HalfCheetah-v2
python3 -u main.py bc Humanoid-v2 &
python3 -u main.py da Humanoid-v2
python3 -u main.py bc Reacher-v2 --num_rollouts 200 &
python3 -u main.py da Reacher-v2 --num_rollouts 200
python3 -u main.py bc Walker2d-v2 &
python3 -u main.py da Walker2d-v2
