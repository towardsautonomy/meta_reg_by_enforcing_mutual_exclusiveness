#!/bin/sh

## Omniglot dataset
# 20-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=25 --meta_train_iterations=10000 --num_inner_updates=1
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=25 --meta_train_iterations=10000 --num_inner_updates=1 --mutual_exclusive

# 10-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=10 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=10 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1 --mutual_exclusive

# 5-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=5 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=5 --k_shot=1 --meta_batch_size=16 --meta_train_iterations=10000 --num_inner_updates=1 --mutual_exclusive

## Pose dataset
# 10-way, 1-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=15 --n_way=10 --k_shot=1 --meta_train_iterations=14000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.01
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=15 --n_way=10 --k_shot=1 --meta_train_iterations=14000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.04
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=15 --n_way=10 --k_shot=1 --meta_train_iterations=14000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.1
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=15 --n_way=10 --k_shot=1 --meta_train_iterations=14000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.4
