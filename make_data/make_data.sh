#!/bin/bash

python make_data/preprocessing.py \
    --file_path ./data/main/English_train.json \
    --folder_path ./data/preprocessed/ \
    --language en 

python make_data/preprocessing.py \
    --file_path ./data/main/English_dev.json \
    --folder_path ./data/preprocessed/ \
    --language en 

python make_data/preprocessing.py \
    --file_path ./data/main/Arabic_train.json \
    --folder_path ./data/preprocessed/ \
    --language ar 

python make_data/preprocessing.py \
    --file_path ./data/main/Arabic_dev.json \
    --folder_path ./data/preprocessed/ \
    --language ar