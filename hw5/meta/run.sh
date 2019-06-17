#!/bin/bash

mkdir output

# problem 1
python3 train_policy.py 'pm-obs' --exp_name q1 --history 1 -lr 5e-5 -n 200 --num_tasks 4

# problem 2
python3 train_policy.py 'pm' --exp_name <experiment_name> --history <history> --discount 0.90 -lr 5e-4 -n 60

# problem 3
python3 train_policy.py 'pm' --exp_name <experiment_name> --history <history> --discount 0.90 -lr 5e-4 -n 60