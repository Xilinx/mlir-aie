
# (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET

# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT

include ../../../../mlir_tutorials/makefile-common

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..
SYSROOT = $(XILINX_SYSROOT)

i32: i32_chess i32.elf
f32: f32_chess f32.elf

build: 
	xchessmk test.prx
sim: 
	xca_udm_dbg -P $(VITIS_AIETOOLS_DIR)/data/versal_prod/lib/ -t sim.tcl


i32_chess: 
	xchesscc -p me -P $(VITIS_AIETOOLS_DIR)/data/versal_prod/lib/ -L $(VITIS_AIETOOLS_DIR)/data/versal_prod/lib/ -c ./hdiff_lap.cc ./hdiff_flux1.cc ./hdiff_flux2.cc

f32_chess: 
	xchesscc -p me -P $(VITIS_AIETOOLS_DIR)/data/versal_prod/lib/ -L $(VITIS_AIETOOLS_DIR)/data/versal_prod/lib/ -c ./hdiff_lap_fp32.cc ./hdiff_flux1_fp32.cc ./hdiff_flux2_fp32.cc

i32.elf: 
	aiecc.py --sysroot=$(SYSROOT) --host-target=aarch64-linux-gnu aie.mlir \
	-I$(AIE_RUNTIME_LIB) $(AIE_RUNTIME_LIB)/test_library.cpp ./test.cpp -o test.elf


f32.elf: 
	aiecc.py --sysroot=$(SYSROOT) --host-target=aarch64-linux-gnu aie_fp32.mlir \
	-I$(AIE_RUNTIME_LIB) $(AIE_RUNTIME_LIB)/test_library.cpp ./test.cpp -o test.elf

clean:
	rm -rf work *mem acdc_project core_* test.elf *.o


