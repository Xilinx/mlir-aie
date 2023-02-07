# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# # SPDX-License-Identifier: MIT

#
# From one diretory up, you can invoke the simulator by typing
# > make -C sim
#
# If you want to change the host source file, you can redefine host:
# > make -C sim host=../yourhost.cpp
#
# Note: The host file location is relative to the <sim> folder or can 
#       be an absolute path
#

MLIR_AIE_INSTALL = $(dir $(shell which aie-opt))/..

ifeq ($(host),) 
host:=../test.cpp
endif
ifeq ($(MLIR_AIE_SRC_DIR),) 
MLIR_AIE_SRC_DIR:=. 
endif
ifeq ($(XILINX_VITIS_AIETOOLS),)
XILINX_VITIS_AIETOOLS = $(dir $(shell which aiesimulator))/..
endif
.PHONY: all link sim clean
.NOTPARALLEL:
all: sim

CC_ENV := (export LD_LIBRARY_PATH=${XILINX_VITIS_AIETOOLS}/lib/lnx64.o:$(LD_LIBRARY_PATH))
CC := "${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/bin/g++" 
CC_ARGS := -fPIC -fpermissive -c -std=c++17 -D__AIEARCH__=10 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -Wno-deprecated-declarations -DSC_INCLUDE_DYNAMIC_PROCESSES -D__AIESIM__ -D__PS_INIT_AIE__ -DXAIE_DEBUG -Og -flto -D main\(...\)=ps_main\(...\) -I${XILINX_VITIS_AIETOOLS}/include -I${XILINX_VITIS_AIETOOLS}/include/drivers/aiengine -I${XILINX_HLS}/include -I${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/include/c++/8.3.0 -I${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/include/c++/8.3.0/backward -I${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/include/c++/8.3.0/x86_64-pc-linux-gnu -I${XILINX_VITIS_AIETOOLS}/data/osci_systemc/include -I${XILINX_VITIS_AIETOOLS}/tps/boost_1_72_0 -I. -I$(MLIR_AIE_SRC_DIR) -I${XILINX_VITIS_AIETOOLS}/include/xtlm/include -I${XILINX_VITIS_AIETOOLS}/include/common_cpp/common_cpp_v1_0/include -I${MLIR_AIE_INSTALL}/runtime_lib  -I ../. -I ../acdc_project 

ps/test_library.o: ${MLIR_AIE_INSTALL}/runtime_lib/test_library.cpp
	$(CC_ENV);$(CC) $(CC_ARGS) -o $@ $<
	
ps/test.o: $(host)
	$(CC_ENV);$(CC) $(CC_ARGS) -o $@ $<

ps/genwrapper_for_ps.o: ps/genwrapper_for_ps.cpp $(host) ../acdc_project/aie_inc.cpp ${MLIR_AIE_INSTALL}/runtime_lib/test_library.cpp
	$(CC_ENV);$(CC) $(CC_ARGS) -o $@ $<

ps/ps.so: ps/genwrapper_for_ps.o ps/test.o ps/test_library.o $(eval PATH:=$(XILINX_VITIS_AIETOOLS)/tps/lnx64/gcc/bin/:$(PATH))
	(${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/bin/g++ -o "ps/ps.so" ps/genwrapper_for_ps.o ps/test.o ps/test_library.o -Wl,--as-needed -shared -lxaiengine -lxioutils -ladf_api -lsystemc -lxtlm -flto -L ${XILINX_VITIS_AIETOOLS}/lib/lnx64.o -L${XILINX_VITIS_AIETOOLS}/data/osci_systemc/lib/lnx64)

link: ps/ps.so

sim: ps/ps.so
	cd ..; aiesimulator --pkg-dir=./sim --dump-vcd foo

clean:
	(rm -rf ps/*.o ps/*.so *.log aiesimulator_output *vcd)
