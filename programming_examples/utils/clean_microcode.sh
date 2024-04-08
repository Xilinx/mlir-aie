#!/bin/bash
#echo $1
sed -i ':a;N;$!ba;s/\.delay_slot\n//g' $1
sed -i ':a;N;$!ba;s/\.swstall delay_slot\n//g' $1
sed -i ':a;N;$!ba;s/\.no_stack_arguments\n//g' $1
sed -i ':a;N;$!ba;s/\.swstall chess_separator_scheduler\n//g' $1
sed -i ':a;N;$!ba;s/\.noswbrkpt\n//g' $1
sed -i ':a;N;$!ba;s/\.nohwbrkpt\n//g' $1
