#!/bin/sh

DATA_DIR='/media/shubham/GoldMine/datasets/pascal3d/PASCAL3D+_release1.1'
CAD_DIR=${DATA_DIR}/CAD
RENDERED_DATA_DIR=${DATA_DIR}/rendered

# Convert the CAD models from *.off to *.stl.
chmod 755 meshconv
find ${CAD_DIR} -maxdepth 2 -mindepth 2 -name "*.off" -exec ./meshconv -c stl {} \;

# Render the dataset. Using the utilities in pose_data
mkdir -p ${RENDERED_DATA_DIR}/rotate
mkdir -p ${RENDERED_DATA_DIR}/rotate_resize
python mujoco_render.py --CAD_dir=${CAD_DIR} --data_dir=${RENDERED_DATA_DIR}
cp -r ${RENDERED_DATA_DIR}/rotate/* ${RENDERED_DATA_DIR}/rotate_resize/*
python resize_images.py --data_dir=${RENDERED_DATA_DIR}/rotate_resize
python data_gen.py --data_dir=${RENDERED_DATA_DIR}/rotate_resize 