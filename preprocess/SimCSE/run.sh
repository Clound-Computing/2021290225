#!/bin/bash
# ===================== Configuration =====================
# 'Chinese' or 'English'
dataset_name='English'

# ===================== Obtain the representations of posts and news =====================
CUDA_VISIBLE_DEVICES=0 python News-Environment-Perception-main\\preprocess\\SimCSE\\get_repr.py --dataset ${dataset_name}
