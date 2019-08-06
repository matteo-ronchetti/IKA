#!/usr/bin/env bash

mkdir data
mkdir models

# Download Rome patches dataset
wget -nc http://pascal.inrialpes.fr/data2/paulin/RomePatches/patch_retrieval_Rome_train.mat -P data/
wget -nc http://pascal.inrialpes.fr/data2/paulin/RomePatches/patch_retrieval_Rome_train_labels.mat -P data/
wget -nc http://pascal.inrialpes.fr/data2/paulin/RomePatches/patch_retrieval_Rome_test.mat -P data/
wget -nc http://pascal.inrialpes.fr/data2/paulin/RomePatches/patch_retrieval_Rome_test_labels.mat -P data/

# Download HardNet++ model
wget -nc https://github.com/DagnyT/hardnet/raw/master/pretrained/pretrained_all_datasets/HardNet%2B%2B.pth -P models/
wget -nc https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty/checkpoint_liberty_no_aug.pth -P models/

python convert_mat_to_npz.py