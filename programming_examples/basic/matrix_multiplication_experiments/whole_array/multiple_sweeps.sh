#!/bin/bash

###########################################
# Use this script to run multiple sweeps
###########################################

# how to run:
# nohup bash multiple_sweeps.sh &

############### i8/i16 #################
# i8/i16 optimized mtk = 4k, B row-maj
python3 GEMM_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=0 --m=80 --k=88 --n=96 --cont_coeff=4

# i8/i16 optimized mtk = 4k, ktn = 4k, B col-maj
python3 GEMM_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=1 --m=80 --k=88 --n=96 --cont_coeff=4

# # i8/i16 baseline mtk = k, B row-maj
# python3 GEMM_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=0 --m=64 --k=184 --n=64 --cont_coeff=1


# ############## i8/i8 #################
# # i8/i8 optimized mtk = 4k, B row-maj
# python3 GEMM_sweep.py --in_ab_type=i8 --out_c_type=i8 --b_col_maj=0 --m=96 --k=96 --n=112 --cont_coeff=4

# # i8/i8 optimized mtk = 4k, ktn = 4k, B col-maj
# python3 GEMM_sweep.py --in_ab_type=i8 --out_c_type=i8 --b_col_maj=1 --m=96 --k=96 --n=112 --cont_coeff=4



# # ############### bf16/bf16 #################
# # bf16/bf16 optimized mtk = 4k, B row-maj
# python3 GEMM_sweep.py --in_ab_type=bf16 --out_c_type=bf16 --b_col_maj=0 --m=80 --k=40 --n=96 --cont_coeff=4

# # bf16/bf16 optimized mtk = 4k, ktn = 4k, B col-maj
# python3 GEMM_sweep.py --in_ab_type=bf16 --out_c_type=bf16 --b_col_maj=1 --m=80 --k=40 --n=96 --cont_coeff=4

# # bf16/bf16 baseline mtk = k, B row-maj
# python3 GEMM_sweep.py --in_ab_type=bf16 --out_c_type=bf16 --b_col_maj=0 --m=64 --k=88 --n=64 --cont_coeff=1




############## i16/i32 #################
# i16/i32 optimized mtk = 4k, B row-maj
# python3 GEMM_sweep.py --in_ab_type=i16 --out_c_type=i32 --b_col_maj=0 --m=48 --k=100 --n=56 --cont_coeff=4

# # i8/i8 optimized mtk = 4k, ktn = 4k, B col-maj
# python3 GEMM_sweep.py --in_ab_type=i16 --out_c_type=i32 --b_col_maj=1 --m=48 --k=100 --n=56 --cont_coeff=4


# ############### contiguous mtk, ktn #################
# python3 contiguous_mtk_ktn_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=0 --m=80 --k=88 --n=96 --M=4160 --K=4224 --N=4224

# python3 contiguous_mtk_ktn_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=1 --m=80 --k=88 --n=96 --M=4160 --K=4224 --N=4224

# python3 contiguous_mtk_ktn_sweep.py --in_ab_type=bf16 --out_c_type=bf16 --b_col_maj=0 --m=80 --k=40 --n=96 --M=4160 --K=4160 --N=4224

# python3 contiguous_mtk_ktn_sweep.py --in_ab_type=bf16 --out_c_type=bf16 --b_col_maj=1 --m=80 --k=40 --n=96 --M=4160 --K=4160 --N=4224