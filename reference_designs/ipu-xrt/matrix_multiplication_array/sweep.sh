#!/usr/bin/bash

csv_out=sweep.csv
log_out=sweep.log
runargs="--verify 0 --iters 10 --warmup 10"
iterations=1

M_lo=256
M_step=256
M_hi=4096
Ms=$(seq $M_lo $M_step $M_hi)
K_lo=256
K_step=256
K_hi=4096
Ks=$(seq $K_lo $K_step $K_hi)
N_lo=256
N_step=256
N_hi=4096
Ns=$(seq $N_lo $N_step $N_hi)

here=$(realpath $(dirname $BASH_SOURCE[0]))
cd $here

printf "M,\tK,\tN" >>$csv_out
for i in $(seq 1 $iterations); do
    printf ",\tIt"$i >>$csv_out
done
printf "\n" >>$csv_out

for M in $Ms; do
    for K in $Ks; do
        for N in $Ns; do
            echo ${M}x${K}x${N} 1>&2
            rm -r /lib/firmware/amdnpu/1502/*.xclbin  # Signing step may hang otherwise
            M=${M} K=${K} N=${N} make all 1>>$log_out 2>&1
            printf "${M},\t${K},\t${N}" >>$csv_out
            for i in $(seq 1 $iterations); do
                M=${M} K=${K} N=${N} runargs=${runargs} make run >.tmp_run.log
                cat .tmp_run.log $run_output >>$log_out
                t=$(cat .tmp_run.log | sed -rn 's/^Avg NPU matmul time: ([0-9]+)us.$/\1/p')
                printf ",\t${t}" >>$csv_out
            done
            printf "\n" >>$csv_out
        done
    done
done
