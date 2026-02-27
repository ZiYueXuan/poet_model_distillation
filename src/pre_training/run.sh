#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# 正确的 deepspeed 启动方式
pdm run deepspeed --num_gpus 2 pretrain.py
