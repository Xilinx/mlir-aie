#!/usr/bin/env bash

my_array=`pytest --collect-only -q`
my_array_length=${my_array[@]}

for element in "${my_array[@]}"
do
   pytest $element
done
echo "DONE"