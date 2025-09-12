set(EXAMPLE "swiglu")
set(INPUT_DATA_TYPE_PYTHON bf16)
set(OUTPUT_DATA_TYPE_PYTHON bf16)
set(INPUT_DATA_TYPE_CPP bfloat16_t)
set(OUTPUT_DATA_TYPE_CPP bfloat16_t)

add_aie_design(swiglu_mv
    PYTHON ../matrix_vector_mul/matrix_vector_mul.py
    PYTHON_FLAGS --dev ${DEVICE} -M 2048 -K 2048 -m 4 --cols 1
    AIE_CORE_KERNELS mv
    EXTRA_AIECC_FLAGS --xclbin-instance-name=swiglu_mv_inst --xclbin-kernel-id=0x901
    OUTPUT_XCLBIN SWIGLU_MV_XCLBIN
    OUTPUT_INSTS SWIGLU_MV_INSTS)

add_aie_design(swiglu_silu
    PYTHON ../silu/silu.py
    PYTHON_FLAGS --dev ${DEVICE} -l 2048 --co 8 --chan 1 --tile-size 256 --trace-size 0
    AIE_CORE_KERNELS silu
    EXTRA_AIECC_FLAGS --xclbin-input=${SWIGLU_MV_XCLBIN} --xclbin-instance-name=swiglu_silu_inst --xclbin-kernel-id=0x902
    OUTPUT_XCLBIN SWIGLU_SILU_XCLBIN
    OUTPUT_INSTS SWIGLU_SILU_INSTS)
add_dependencies(swiglu_silu_xclbin swiglu_mv_xclbin)
add_dependencies(swiglu_silu_insts swiglu_mv_xclbin)

add_aie_design(swiglu_eltwise_mul
    PYTHON ../elementwise_mul/eltwise_mul.py
    PYTHON_FLAGS -d ${DEVICE} -l 2048 -co 8 -ch 2 -ts 256 -t 0
    AIE_CORE_KERNELS mul_aie2
    EXTRA_AIECC_FLAGS --xclbin-input=${SWIGLU_SILU_XCLBIN} --xclbin-instance-name=swiglu_eltwise_mul_inst --xclbin-kernel-id=0x903
    OUTPUT_XCLBIN SWIGLU_ELTWISE_MUL_XCLBIN
    OUTPUT_INSTS SWIGLU_ELTWISE_MUL_INSTS)
add_dependencies(swiglu_eltwise_mul_xclbin swiglu_silu_xclbin)
add_dependencies(swiglu_eltwise_mul_insts swiglu_silu_xclbin)

# the last xclbin contains all the kernels since we chained them together using --xclbin-input
# make sure to use different --xclbin-instance-name, --xclbin-kernel-id and --xclbin-kernel-name arguments to aiecc.py when chaining like this
set(SWIGLU_COMBINED_XCLBIN ${SWIGLU_ELTWISE_MUL_XCLBIN})

set(GOLDEN_VALUES_PATH "${CMAKE_BINARY_DIR}/example/swiglu/golden_${EXAMPLE}")

add_aie_host(${EXAMPLE} HOST swiglu.cpp HOST_FLAGS GOLDEN_REFERENCE_PATH="${GOLDEN_VALUES_PATH}/golden_reference.bin" OUTPUT_HOST SWIGLU_HOST)
add_dependencies(
    ${EXAMPLE} 
    swiglu_eltwise_mul_xclbin # ${SWIGLU_ELTWISE_MUL_XCLBIN}
    swiglu_eltwise_mul_insts # ${SWIGLU_ELTWISE_MUL_INSTS}
)

add_aie_ci_test(${EXAMPLE}
                RUN 
                    "${SWIGLU_HOST} \
                        --dim 2048 \
                        -x ${SWIGLU_COMBINED_XCLBIN} \
                        --insts-mv ${SWIGLU_MV_INSTS} \
                        --insts-silu ${SWIGLU_SILU_INSTS} \
                        --insts-eltwise-mul ${SWIGLU_ELTWISE_MUL_INSTS}
                        --epsilon 0.07 \
                        --abs_th 0.7"
                CHECK
                    "PASS!"
                METRICS 
                    "Latency" [=[Elapsed time \(μs\): (?P<metric>\d+)]=])

add_golden_reference_binary_generator(${EXAMPLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/swiglu_golden.py
    ${GOLDEN_VALUES_PATH}
    --dim 2048
)