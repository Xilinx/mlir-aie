all: build sim
image: clean chess elf scp
.PHONY : all build sim clean

build: 
	xchessmk test.prx

sim: 
	xca_udm_dbg -P /proj/xbuilds/2021.1_daily_latest/installs/lin64/Vitis/2021.1/aietools//data/cervino/lib -t sim.tcl

chess: 
	xchesscc -p me -P /proj/xbuilds/2021.1_released/installs/lin64/Vitis/2021.1/aietools/data/cervino/lib -L/proj/xbuilds/2021.1_released/installs/lin64/Vitis/2021.1/cardano/lib -c ./hdiff_lap.cc ./hdiff_flux.cc

elf:
	aiecc.py --sysroot=/group/xrlabs/platforms/pynq_on_versal_vck190_hacked/vck190-sysroot aie.mlir -v -I../../../../runtime_lib ../../../../runtime_lib/test_library.cpp ./test.cpp -v -o test.elf

scp:
	scp -r ../HDIFF_DMA_multi_AIE xilinx@172.20.9.236:/home/xilinx/gagan
clean:
	rm -rf work *mem *.o acdc_project core_* test.elf