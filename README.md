# Meta-Regularization by enforcing Mutual-Exclusiveness

## Setup Instructions  
 - To setup conda environment `conda env create -f conda_env.yml`  
 
### Omniglot Dataset  
 - To prepare non-mutual-exclusive dataset, use the script: `src/prepOmniglotDataset.py`.  

### Pose Dataset  
 - Copy the license text file `mjkey.txt` at `src/pose_data/`.  
 - Download CAD models from [Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild](http://cvgl.stanford.edu/projects/pascal3d.html) PASCAL3D+_release1.1.  
 - Configure data path in the script `src/pose_data/convert_and_render.sh` and run this script. This will render the dataset using CAD models and save png files along with labels at configured directory. It will also generate `pickle` files of the dataset.      
 - Or you can download the prepared pickle file from here: `https://drive.google.com/drive/folders/1V_9NqqelQyuyYtPv6ndoqXQp-mp_zmeG?usp=sharing`    

## Experiments    

<!-- *Table I - Experiment Results*
| N-way | meta-train K-shot | meta-test K-shot | MAML Test Accuracy | MAML Accuracy StdDev |
|-------|-------------------|------------------|--------------------|----------------------|
| 10    | 1                 | 1                | 0.9130001          | 0.087736346          |
| 10    | 1                 | 10               | 0.9779833          | 0.024727175          |    -->

![](plots/cls_20.mbs_16.k_shot_1.inner_numstep_1.inner_updatelr_0.4.learn_inner_update_lr_False.mutual_exclusive_True.png)  
*Figure 1 - Training and Validation plot for 20-way, 1-shot, batch_size=16, mutual-exclusive*  

![](plots/cls_20.mbs_16.k_shot_1.inner_numstep_1.inner_updatelr_0.4.learn_inner_update_lr_False.mutual_exclusive_False.png)  
*Figure 2 - Training and Validation plot for 20-way, 1-shot, batch_size=16, non-mutual-exclusive*   

![](plots/cls_5.mbs_4.k_shot_2.inner_numstep_1.inner_updatelr_0.4.learn_inner_update_lr_False.dataset_pose.mutual_exclusive_False.png)  
*Figure 3 - Training and Validation mean-squared-error for 5-tasks, 2-shot, batch_size=4*  

### With meta-regularization

 - meta-regularization model maximize distance between task tail params only   

![](media/cls_10.mbs_15.k_shot_1.inner_numstep_1.meta_lr_0.001.inner_updatelr_0.4.learn_inner_update_lr_False.dataset_pose.mutual_exclusive_False.png)
 (3.0752056, 1.5007443, 0.12008455722511353)

 - meta-regularization model maximize distance between complete model params   
![](media/cls_10.mbs_15.k_shot_1.inner_numstep_1.meta_lr_0.001.inner_updatelr_0.4.learn_inner_update_lr_False.dataset_pose.mutual_exclusive_False_metareg_True_complete_model_dist_maximize.png)  
 (26.181309, 144.12361, 11.532290726043973)

 - meta-regularization model minimize distance between encoder params and maximize distance between task tail params  
 ![](media/cls_10.mbs_15.k_shot_1.inner_numstep_1.meta_lr_0.001.inner_updatelr_0.4.learn_inner_update_lr_False.dataset_pose.mutual_exclusive_False_metareg_True_tail_dist_maximize_en_dist_minimize.png)   
 (3.7077172, 1.1246978, 0.08999456736729294)
