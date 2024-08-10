#!/bin/bash

cd data/
rm -rf *.txt

cd ../../reference/

python uni_layer.py ../conv2d/include/test_params.h -conv2d -genPath ../conv2d/data/

cd ../conv2d/data/

cat AIn.txt WIn.txt > AInWIn.txt



