#  (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET     

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..
SYSROOT = /mnt/scratch/gagsingh/sysroot/sysroot

all: test.elf

build: 
	xchessmk test.prx
sim: 
	xca_udm_dbg -P /mnt/scratch/tools_tmp/Xilinx/Vitis/2021.2/aietools/data/cervino/lib/ -t sim.tcl


chess: 
	/mnt/scratch/tools_tmp/Xilinx/Vitis/2021.2/aietools/bin/xchesscc -p me -P /mnt/scratch/tools_tmp/Xilinx/Vitis/2021.2/aietools/data/cervino/lib/ -L /mnt/scratch/tools_tmp/Xilinx/Vitis/2021.2/aietools/data/cervino/lib/ -c laplacian_2d.cc
# test.elf: 
# 	/mnt/scratch/gagsingh/mlir-air/utils/mlir-aie/install/bin/aiecc.py -v --aie-generate-xaiev2 --host-target=x86_64-amd-linux-gnu --sysroot=/mnt/scratch/gagsingh/xilinx-versal-setup/xilinx-versal-common-sysroot-2020.2/sources/x86_64-linux aie.mlir \
# 	-I/mnt/scratch/gagsingh/mlir-air/utils/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2/gaganlib/include -L/mnt/scratch/gagsingh/mlir-air/utils/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2/gaganlib/lib \
# 	 -Wl,-R/mnt/scratch/gagsingh/mlir-air/utils/embeddedsw/XilinxProcessorIPLib/drivers/aienginev2/gaganlib/lib \
# 	-I/mnt/scratch/gagsingh/mlir-air/install-pcie/runtime_lib/airhost/include -I/mnt/scratch/gagsingh/mlir-air/utils/mlir-aie/runtime_lib  -I/mnt/scratch/gagsingh/mlir-air/utils/mlir-aie/runtime_lib/test_library.cpp \
# 	-L/mnt/scratch/gagsingh/mlir-air/install-pcie/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o test.elf


test.elf: 
	/mnt/scratch/gagsingh/mlir-aie/install/bin/aiecc.py --sysroot=/mnt/scratch/gagsingh/sysroot/sysroot --host-target=aarch64-linux-gnu aie.mlir \
	-I/mnt/scratch/gagsingh/mlir-aie/install/runtime_lib /mnt/scratch/gagsingh/mlir-aie/install/runtime_lib/test_library.cpp ./test.cpp -o test.elf
# test.elf: 
# 	/mnt/scratch/gagsingh/mlir-aie/install/bin/aiecc.py --aie-generate-xaiev2 --host-target=aarch64-linux-gnu --sysroot=/mnt/scratch/gagsingh/sysroot/sysroot\
# 	 aie.mlir -I/mnt/scratch/gagsingh/sysroot/sysroot/opt/xaienginev2/include -L/mnt/scratch/gagsingh/sysroot/sysroot/opt/xaienginev2/lib\
# 	  -Wl,-R/mnt/scratch/gagsingh/sysroot/sysroot/opt/xaienginev2/lib -I/mnt/scratch/gagsingh/mlir-aie/install/runtime_lib -I/mnt/scratch/gagsingh/mlir-aie/install/runtime_lib/test_library.cpp\
# 	   -Wl,--no-whole-archive -lstdc++ test.cpp -o test.elf
# test.elf: aie.mlir test.cpp
# 	aiecc.py -v --aie-generate-xaiev2 --host-target=x86_64-amd-linux-gnu --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
# 						-I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib $(ACDC_AIE)/runtime_lib/test_library.cpp \
# 						-L$(ACDC_AIR)/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@ 

clean:
	rm -rf work *mem acdc_project core_* test.elf *.o


