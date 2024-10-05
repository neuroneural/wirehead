#!/bin/bash

srun -p qTRDGPUH -A psy53c17 -v -t600 -N1-1 -c16 --gres=gpu:A100:1 --mem=200g --nodelist=arctrddgxa003 --pty python train.py
