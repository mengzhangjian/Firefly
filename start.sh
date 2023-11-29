#!/bin/bash

cd /nfs/volume-76-1/zhangjian/Documents/Firefly/

source python/bin/activate

torchrun --nproc_per_node=4 train_qlora.py --train_args_file train_args/qlora/baichuan-7b-sft-qlora.json
