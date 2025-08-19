#!/usr/bin/env bash
GLOG_vmodule=MemcachedClient=-1
srun --gpu -n4 --cpus-per-task=4  "python main_dvs.py --spike --step 10"