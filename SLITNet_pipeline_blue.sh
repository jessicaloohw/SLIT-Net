#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# SLIT-Net
# DOI: 10.1109/JBHI.2020.2983549

# Main directory:
MODEL_DIR='Models/Blue_Light'

# Training and validation:
for K_FOLD in {'K1','K2','K3','K4','K5','K6','K7'}
do
    python SLITNet_train_blue.py $MODEL_DIR $K_FOLD 1 'coco' 0 100 $K_FOLD 0 1
    python SLITNet_validate_blue.py $MODEL_DIR $K_FOLD 79 99 $K_FOLD
done

# Testing (replace xx with model number):
python SLITNet_test_blue.py $MODEL_DIR 'K1' xx 'K1'
python SLITNet_test_blue.py $MODEL_DIR 'K2' xx 'K2'
python SLITNet_test_blue.py $MODEL_DIR 'K3' xx 'K3'
python SLITNet_test_blue.py $MODEL_DIR 'K4' xx 'K4'
python SLITNet_test_blue.py $MODEL_DIR 'K5' xx 'K5'
python SLITNet_test_blue.py $MODEL_DIR 'K6' xx 'K6'
python SLITNet_test_blue.py $MODEL_DIR 'K7' xx 'K7'

# Testing with trained models (uncomment to use):
MODEL_DIR='Trained_Models/Blue_Light'

#python SLITNet_test_blue.py $MODEL_DIR 'K1' 88 'K1'
#python SLITNet_test_blue.py $MODEL_DIR 'K2' 99 'K2'
#python SLITNet_test_blue.py $MODEL_DIR 'K3' 87 'K3'
#python SLITNet_test_blue.py $MODEL_DIR 'K4' 86 'K4'
#python SLITNet_test_blue.py $MODEL_DIR 'K5' 91 'K5'
#python SLITNet_test_blue.py $MODEL_DIR 'K6' 83 'K6'
#python SLITNet_test_blue.py $MODEL_DIR 'K7' 92 'K7'
