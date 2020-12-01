
#export LD_LIBRARY_PATH=/opt/xaiengine/lib:$LD_LIBRARY_PATH

CC := gcc
#CFLAGS := -Wall -std=c++11 -I/opt/xaiengine/include
CFLAGS := -std=c++11 -I/opt/xaiengine/include
#CFLAGS := -I/opt/xaienginev2/include

AIE_SRC_DIR  = kernel
AIE_BASE_OBJ = ../../../runtime_lib/me_basic.o

AIE_OPT     = aie-opt
AIE_XLATE   = aie-translate

all: test.exe

# compile sources
%.o: %.cpp aie_inc.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

# compile and link on board
%.exe: %.o
	$(CC) $^ \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-ldl \
		-L/opt/xaiengine/lib \
		-o $@

# mlir translate into include c source (e.g. aie_inc.cpp)
%_inc.cpp: %.mlir
	$(AIE_OPT) --aie-create-flows --aie-find-flows $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

%.elf : %.mlir
	$(AIE_OPT) --aie-llvm-lowering="tilecol=1 tilerow=2" $^ | $(AIE_XLATE) --aie-generate-llvmir | opt -strip -S | llc -O0 --march=aie --filetype=obj -o=$*.o && \
	ld.lld $*.o $(AIE_BASE_OBJ) -T ./ld.script -o $@

# generate kernel source
%.elf: $(AIE_SRC_DIR)/%.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

clean:
	rm -rf aie_inc.cpp *.o *.exe *.elf ./$(AIE_SRC_DIR)/work
