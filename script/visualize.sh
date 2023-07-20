#!/bin/bash
conda activate torch
cd ..
echo $PWD
DATA_PATH="../Dataset"
TARGET_PATH="/Exp/demo"
DATE="/0328"
in_out_len="2010"
P=$DATA_PATH$TARGET_PATH$DATE
label="ground_truth missing_point predicted"
output="result/nthu_report/0422/S09_03_01_"$in_out_len"_.gif"
frames_limit=900

echo $P
python visualize_3D.py -f $P/S09_03_01_ori.pkl $P/S09_03_01_0926_V3_Human36M_train_angle_01_${in_out_len}_ori.pkl\
    $P/S09_03_01_0926_V3_Human36M_train_angle_01_2010.pkl -l $label -o $output -n $frames_limit
cd script

# visualize_3D.py -f /home/156785978/refactor/BiLSTM-VAE/result/for_thesis/reconstruction/gt/0%.pkl -o /home/156785978/refactor/BiLSTM-VAE/result/for_thesis/reconstruction/gt/0%.gif