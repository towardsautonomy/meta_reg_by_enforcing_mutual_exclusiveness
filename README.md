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

### Running the experiments  

 - To run experiments:  

 ```
 sh run_experiments.sh
 ```
