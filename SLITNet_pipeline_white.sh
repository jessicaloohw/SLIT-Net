#!/bin/bash
export CUDA_VISIBLE_DEVICES=1


# Training and validation:
MODEL_DIR='Models/White_Light'
for K_FOLD in {'K1','K2','K3','K4','K5','K6','K7'}
do
    python SLITNet_train_white.py $MODEL_DIR $K_FOLD 1 'coco' 0 100 $K_FOLD 0 1
    python SLITNet_validate_white.py $MODEL_DIR $K_FOLD 79 99 $K_FOLD
done

# Testing:
MODEL_DIR='Trained_Models/White_Light'
python SLITNet_test_white.py $MODEL_DIR 'K1' 92 'K1'
python SLITNet_test_white.py $MODEL_DIR 'K2' 96 'K2'
python SLITNet_test_white.py $MODEL_DIR 'K3' 98 'K3'
python SLITNet_test_white.py $MODEL_DIR 'K4' 93 'K4'
python SLITNet_test_white.py $MODEL_DIR 'K5' 80 'K5'
python SLITNet_test_white.py $MODEL_DIR 'K6' 80 'K6'
python SLITNet_test_white.py $MODEL_DIR 'K7' 97 'K7'