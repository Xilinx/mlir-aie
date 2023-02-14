#
#Copyright 2020 Xilinx, Inc.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#
CXX = aarch64-linux-gnu-g++
LIBCXX_VERSION ?= 10.2.0
SDKTARGETSYSROOT ?= ../../petalinux/build/petalinux/images/linux/sdk/sysroots/cortexa72-cortexa53-xilinx-linux

HOST_EXE     = ../host.exe
HOST_INC   = -I../ -I../aie
HOST_OBJ   = aie_control_xrt.o graph.o
#HOST_OBJ   = aie_control.o graph.o
AIE_CTRL_CPP = ../Work/ps/c_rts/aie_control_xrt.cpp
#AIE_CTRL_CPP = ../Work/ps/c_rts/aie_control.cpp
GRAPH_CPP = ../aie/graph.cpp

CXXFLAGS += -std=c++14 -I$(XILINX_VIVADO)/include/ -I${SDKTARGETSYSROOT}/usr/include/xrt/ -O0 -g -Wall -c -fmessage-length=0 --sysroot=${SDKTARGETSYSROOT} -I${XILINX_VITIS}/aietools/include ${HOST_INC} -I${SDKTARGETSYSROOT}/usr/include/c++/${LIBCXX_VERSION}/ -I${SDKTARGETSYSROOT}/usr/include/c++/${LIBCXX_VERSION}/aarch64-xilinx-linux/

LDFLAGS += -ladf_api_xrt -lgcc -lc -lxrt_coreutil -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ -L${SDKTARGETSYSROOT}/usr/lib/ --sysroot=${SDKTARGETSYSROOT} -L$(XILINX_VITIS)/aietools/lib/aarch64.o
#LDFLAGS += -lcardano_api_xrt -lgcc -lc -lxrt_coreutil -lxilinxopencl -lpthread -lrt -ldl -lcrypt -lstdc++ -L${SDKTARGETSYSROOT}/usr/lib/ --sysroot=${SDKTARGETSYSROOT} -L$(XILINX_VITIS)/cardano/lib/aarch64.o

${HOST_EXE}: ${HOST_OBJ}
	${CXX} -o $@ $^ ${LDFLAGS}

#%.o: %.cpp
#	${CXX} ${CXXFLAGS} -o $@ $<

aie_control_xrt.o : $(AIE_CTRL_CPP)
	${CXX} ${CXXFLAGS} -o $@ $<

graph.o: $(GRAPH_CPP)
	${CXX} ${CXXFLAGS} -o $@ $<


#aie_control_xrt.cpp: ${AIE_CTRL_CPP}
#	cp -f ${AIE_CTRL_CPP} .
#aie_control.cpp: ${AIE_CTRL_CPP}

#graph.cpp: ${GRAPH_CPP}
#	cp -f ${GRAPH_CPP} .

clean:
#	rm -rf ${HOST_OBJ} .Xil ${HOST_EXE} aie_control.cpp
	rm -rf ${HOST_OBJ} .Xil ${HOST_EXE} aie_control_xrt.cpp
