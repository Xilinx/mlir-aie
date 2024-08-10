#!/bin/bash

cd data/
rm -rf *.txt

cd ../../reference/

python uni_layer.py ../maxpool/include/test_params.h -maxpool -gen -genPath ../maxpool/data/


