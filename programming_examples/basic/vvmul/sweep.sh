M_lo=4096
M_step=4096
M_hi=12288

Ms=$(seq $M_lo $M_step $M_hi)

for M in $Ms; do
    make clean && make run data_size=$M
done