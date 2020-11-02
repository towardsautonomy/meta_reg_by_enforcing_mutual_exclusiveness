#!/bin/sh

## Omniglot dataset
20-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1 --mutual_exclusive

# 10-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=10 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=10 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1 --mutual_exclusive

# 5-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=5 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=5 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1 --mutual_exclusive

## Pose dataset
# 5-way, 1-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=4 --n_way=5 --k_shot=1 --meta_train_iterations=5000 --num_inner_updates=1

# 5-way, 2-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=4 --n_way=5 --k_shot=2 --meta_train_iterations=5000 --num_inner_updates=1

# 5-way, 5-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=2 --n_way=5 --k_shot=5 --meta_train_iterations=5000 --num_inner_updates=1