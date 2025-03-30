#!/bin/bash

# how to run:
# nohup bash mult_sweeps.sh &

python3 DDR_GEMM_imitation_sweep_A_B_row_maj.py --npu_columns=4


# python3 DDR_GEMM_imitation_sweep.py --npu_columns=4 --b_col_maj=1