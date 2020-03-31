#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


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
