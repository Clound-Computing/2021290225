#!/bin/bash
# ===================== Configuration =====================
# dataset_name='Chinese'
dataset_name='English'
macro_env_days=3

# ===================== Get the macro env and rank its internal items by similarites =====================
python News-Environment-Perception-main\\preprocess\\NewsEnv\\get_env.py --dataset ${dataset_name} --macro_env_days ${macro_env_days}