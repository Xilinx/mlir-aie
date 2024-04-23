#!/usr/bin/env bash
set -eux

eval "$(conda shell.bash hook)"
conda activate mlir-aie

OLD_PYTHON_PATH=$(python -c "import sys; print(';'.join(sys.path))")
export PYTHONPATH="$PWD/../../../cmake-build-debug/python:$OLD_PYTHON_PATH"

my_array=( `pytest --collect-only -q` )
my_array_length=${#my_array[@]}

for element in "${my_array[@]}"
do
   pytest $element
done