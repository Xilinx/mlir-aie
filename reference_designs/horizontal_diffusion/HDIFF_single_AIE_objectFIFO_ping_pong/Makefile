
# (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET

# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..
SYSROOT = $XILINX_SYSROOT

i32: i32_chess i32.elf
f32: f32_chess f32.elf

build: 
	xchessmk test.prx
sim: 
	xca_udm_dbg -P $AIETOOLS_ROOT/data/versal_prod/lib/ -t sim.tcl


i32_chess: 
	xchesscc -p me -P $AIETOOLS_ROOT/data/versal_prod/lib/ -L $AIETOOLS_ROOT/data/versal_prod/lib/ -c hdiff.cc

f32_chess: 
	xchesscc -p me -P $AIETOOLS_ROOT/data/versal_prod/lib/ -L $AIETOOLS_ROOT/data/versal_prod/lib/ -c hdiff_fp32.cc


i32.elf: 
	aiecc.py --sysroot=$SYSROOT --host-target=aarch64-linux-gnu aie.mlir \
	-I$RUNTIME_LIB $RUNTIME_LIB/test_library.cpp ./test.cpp -o test.elf

f32.elf: 
	aiecc.py --sysroot=$SYSROOT --host-target=aarch64-linux-gnu aie_fp32.mlir \
	-I$RUNTIME_LIB $RUNTIME_LIB/test_library.cpp ./test.cpp -o test.elf


clean:
	rm -rf work *mem acdc_project core_* test.elf *.o


