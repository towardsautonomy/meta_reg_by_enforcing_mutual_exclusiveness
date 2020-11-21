#!/bin/sh

## Omniglot dataset
# 20-way, 1-shot
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=2000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --mutual_exclusive
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.1 --metareg_tau=1.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.1 --metareg_tau=2.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.1 --metareg_tau=3.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.2 --metareg_tau=3.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.3 --metareg_tau=3.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.4 --metareg_tau=3.0 
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.5 --metareg_tau=3.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.3 --metareg_tau=4.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.4 --metareg_tau=4.0
python train.py --dataset=omniglot --data_path=datasets/omniglot_resized --n_way=20 --k_shot=1 --meta_batch_size=10 --meta_train_iterations=5000 --num_inner_updates=1 --meta_lr=0.0025 --inner_update_lr=0.04 --metareg --metareg_lambda=0.5 --metareg_tau=4.0

# ## Pose dataset
# 10-way, 1-shot
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=10 --n_way=10 --k_shot=5 --meta_train_iterations=15000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.01 --learn_inner_update_lr
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=10 --n_way=10 --k_shot=5 --meta_train_iterations=15000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.01 --learn_inner_update_lr --metareg --metareg_lambda=0.1 --metareg_tau=1.0
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=10 --n_way=10 --k_shot=5 --meta_train_iterations=15000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.01 --learn_inner_update_lr --metareg --metareg_lambda=0.1 --metareg_tau=2.0
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=10 --n_way=10 --k_shot=5 --meta_train_iterations=15000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.01 --learn_inner_update_lr --metareg --metareg_lambda=0.1 --metareg_tau=3.0
python train.py --dataset=pose --data_path=datasets/pascal3d_pose --meta_batch_size=10 --n_way=10 --k_shot=5 --meta_train_iterations=15000 --num_inner_updates=1 --meta_lr=0.001 --inner_update_lr=0.01 --learn_inner_update_lr --metareg --metareg_lambda=0.2 --metareg_tau=1.0
