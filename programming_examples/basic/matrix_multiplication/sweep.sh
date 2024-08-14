#!/usr/bin/bash

# run this script from one of the subdirectories to perform a sweep,
# e.g. from within whole_array, run ../sweep.sh.


csv_out=my_sweep.csv
log_out=my_sweep.log
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

export m=64
export k=64
export n=64
export dtype_in=i16
export dtype_out=i32
export n_aie_cols=4
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
export runargs="--iters 1 --warmup 1"

# Print configuration used to run for reproducibility
env >>$log_out
cat Makefile >>$log_out

printf "M,K,N" >>$csv_out
for i in $(seq 1 $iterations); do
    printf ",It"$i >>$csv_out
done
printf "\n" >>$csv_out

start_M=2304
start_K=3328
start_N=2560

for M in $Ms; do
    for K in $Ks; do
        for N in $Ns; do
            if [ $M -lt $start_M ] || ([ $M -eq $start_M ] && [ $K -lt $start_K ]) || ([ $M -eq $start_M ] && [ $K -eq $start_K ] && [ $N -lt $start_N ]); then
                continue
            fi
            export M=$M
            export K=$K
            export N=$N
            echo ${M}x${K}x${N} 1>&2
            make clean 1>>$log_out 2>&1
            make all 1>>$log_out 2>&1
            printf "${M},${K},${N}" >>$csv_out
            for i in $(seq 1 $iterations); do
                make run >.tmp_run.log
                cat .tmp_run.log $run_output >>$log_out
                t=$(cat .tmp_run.log | sed -rn 's/^Avg NPU matmul time: ([0-9.]+)us.$/\1/p')
                printf ",${t}" >>$csv_out
            done
            printf "\n" >>$csv_out
        done
    done
done
