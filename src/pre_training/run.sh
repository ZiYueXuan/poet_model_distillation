#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

# 将标准输出和标准错误都追加写入 log.txt（避免覆盖历史日志）
pdm run deepspeed --num_gpus 2 pretrain.py >> training_log.txt 2>&1

