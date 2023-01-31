ifeq ($(AIE_SRC_DIR),) 
AIE_SRC_DIR:=. 
endif
ifeq ($(XILINX_VITIS_AIETOOLS),)
XILINX_VITIS_AIETOOLS:=${XILINX_VITIS}/aietools
endif
.PHONY: all clean
.NOTPARALLEL:
all: clean link

genwrapper_for_ps_i6:
	(export LD_LIBRARY_PATH=${XILINX_VITIS_AIETOOLS}/lib/lnx64.o:$(LD_LIBRARY_PATH);"${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/bin/g++" -fPIC -fpermissive -c -std=c++17 -D__AIEARCH__=10 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -Wno-deprecated-declarations -DSC_INCLUDE_DYNAMIC_PROCESSES -D__AIESIM__ -D__PS_INIT_AIE__ -DXAIE_DEBUG -Og -flto -D main\(...\)=ps_main\(...\) -I${XILINX_VITIS_AIETOOLS}/include -I${XILINX_VITIS_AIETOOLS}/include/drivers/aiengine -I${XILINX_HLS}/include -I${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/include/c++/8.3.0 -I${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/include/c++/8.3.0/backward -I${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/include/c++/8.3.0/x86_64-pc-linux-gnu -I${XILINX_VITIS_AIETOOLS}/data/osci_systemc/include -I${XILINX_VITIS_AIETOOLS}/tps/boost_1_72_0 -I. -I$(AIE_SRC_DIR) -I${XILINX_VITIS_AIETOOLS}/include/xtlm/include -I${XILINX_VITIS_AIETOOLS}/include/common_cpp/common_cpp_v1_0/include -I${MLIR_AIE_DIR}/runtime_lib  -I ../../../../. -I ../../../../acdc_project -o "generated-objects/genwrapper_for_ps_i6.o" generated-source/genwrapper_for_ps_i6.cpp) 

link: genwrapper_for_ps_i6 $(eval PATH:=$(XILINX_VITIS_AIETOOLS)/tps/lnx64/gcc/bin/:$(PATH))
	(${XILINX_VITIS_AIETOOLS}/tps/lnx64/gcc/bin/g++ -o "generated-objects/ps_i6.so" generated-objects/genwrapper_for_ps_i6.o -Wl,--as-needed -shared -lxaiengine -lxioutils -ladf_api -lsystemc -lxtlm -flto -L ${XILINX_VITIS_AIETOOLS}/lib/lnx64.o -L${XILINX_VITIS_AIETOOLS}/data/osci_systemc/lib/lnx64)

clean:
	(rm -rf generated-objects/*)
