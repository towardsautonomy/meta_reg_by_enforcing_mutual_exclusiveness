#!/bin/sh

## Omniglot dataset
# 20-way, 1-shot
python train.py --n_way=20 --k_shot=1 --meta_batch_size=16
python train.py --n_way=20 --k_shot=1 --meta_batch_size=16 --mutual_exclusive

# 10-way, 1-shot
python train.py --n_way=10 --k_shot=1 --meta_batch_size=16
python train.py --n_way=10 --k_shot=1 --meta_batch_size=16 --mutual_exclusive

# 5-way, 1-shot
python train.py --n_way=5 --k_shot=1 --meta_batch_size=16
python train.py --n_way=5 --k_shot=1 --meta_batch_size=16 --mutual_exclusive

## Pose dataset
# 10-way, 1-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=4 --n_way=5 --k_shot=1

# 5-way, 1-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=4 --n_way=5 --k_shot=2

# 5-way, 5-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=2 --n_way=5 --k_shot=5

