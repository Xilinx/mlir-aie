#!/usr/bin/bash

# run this script from one of the subdirectories to perform a sweep,
# e.g. from within whole_array, run ../my_sweep.sh.

runargs="--iters 20 --warmup 10"
iterations=1

Ms=(512 1024 2560 4096)
Ks=(512 1024 2560 4096)
Ns=(512 1024 2560 4096)

perform_sweep() {
    echo "Performing function sweep"
    # Print configuration used to run for reproducibility
    env >>$log_out
    cat Makefile >>$log_out

    printf "M,K,N" >>$csv_out

    for i in $(seq 1 $iterations); do
        printf ",It"$i >>$csv_out
    done

    printf ",Status" >>$csv_out

    printf ",shuffle_time" >>$csv_out

    printf "\n" >>$csv_out

    for M in "${Ms[@]}"; do
        for K in "${Ks[@]}"; do
            for N in "${Ns[@]}"; do
                export M=$M
                export K=$K
                export N=$N
                echo ${M}x${K}x${N} 1>&2
                make clean 1>>$log_out 2>&1
                printf "${M},${K},${N}" >>$csv_out
                for i in $(seq 1 $iterations); do
                    make run &>.tmp_run.log
                    cat .tmp_run.log $run_output >>$log_out
                    t=$(cat .tmp_run.log | sed -rn 's/^Avg NPU matmul time: ([0-9.]+)us.$/\1/p')
                    printf ",${t}" >>$csv_out
                    if cat .tmp_run.log | grep -q -F "PASS!"; then
                        printf ",PASS" >>$csv_out
                    else
                        printf ",FAIL" >>$csv_out
                    fi
                    shuffle_time=$(cat .tmp_run.log | sed -rn 's/^Shuffle time: ([0-9.]+)us.$/\1/p')
                    printf ",${shuffle_time}" >>$csv_out
                done
                printf "\n" >>$csv_out
            done
        done
    done
}

run_selected_hyperparameters() {
    EXPECTED_ARGS=10
    if [[ "$#" -ne "$EXPECTED_ARGS" ]]; then
        echo "Usage: $0 <m> <k> <n> <r> <s> <t> <dtype_in> <dtype_out> <n_aie_cols> <use_chess>"
        echo "Error: Incorrect number of arguments. Expected $EXPECTED_ARGS, got $#."
        exit 1
    fi

    if [ "${10}" == 1 ]; then
        compiler="chess"
    else
        compiler="peano"
    fi

    csv_out="results/${1}x${2}x${3}_${4}x${5}x${6}_${8}_out_${9}_col_${compiler}.csv"
    log_out="logs/${1}x${2}x${3}_${4}x${5}x${6}_${8}_out_${9}_col_${compiler}.log"

    export m=$1
    export k=$2
    export n=$3
    export dtype_in=$7
    export dtype_out=$8
    export n_aie_cols=$9
    export use_chess=${10}

    perform_sweep
}

# This is constant for all runs
export runargs="${runargs}"

# run_selected_hyperparameters: m k n r s t dtype_in dtype_out n_aie_cols emulate_bfloat16_mmul_with_bfp16 use_chess

# cd ./whole_array_mixed
# mkdir results
# mkdir logs
# run_selected_hyperparameters 64 64 64 8 8 8 bfp16-bf16 bfp16-bf16 4 1

# cd ../whole_array
# mkdir results
# mkdir logs
# run_selected_hyperparameters 64 64 64 8 8 8 bfp16 bfp16 4 1

cd ./whole_array_mixed
mkdir results
mkdir logs
run_selected_hyperparameters 64 64 64 8 8 8 bfp16-bf16 bfp16-bf16 8 1

cd ../whole_array_shuffle
mkdir results
mkdir logs
run_selected_hyperparameters 64 64 64 8 8 8 bfp16 bfp16-shuffle 8 1

# I want to run this again to see if the transposition and pipelining pragmas change anything
cd ../whole_array
mkdir results
mkdir logs
run_selected_hyperparameters 64 64 64 8 8 8 bfp16 bfp16 8 1
