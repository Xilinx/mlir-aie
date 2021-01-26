
#export LD_LIBRARY_PATH=/opt/xaiengine/lib:$LD_LIBRARY_PATH

CC := gcc
#CFLAGS := -Wall -std=c++11 -I/opt/xaiengine/include
CFLAGS := -std=c++11 -I/opt/xaiengine/include
#CFLAGS := -I/opt/xaienginev2/include

AIE_SRC_DIR  = .
AIE_BASE_OBJ = ../../../runtime_lib/me_basic.o

AIE_OPT     = aie-opt
AIE_XLATE   = aie-translate
XBRIDGE     = xbridge
#AIE_OPT     = ../../../../build/aie/bin/aie-opt
#AIE_XLATE   = ../../../../build/aie/bin/aie-translate
#XBRIDGE     = ../../../../build/aie-tools/bin/xbridge
#AIE_OPT     = ../../../../build-container/aie/bin/aie-opt
#AIE_XLATE   = ../../../../build-container/aie/bin/aie-translate
#XBRIDGE     = ../../../../build-container/aie-tools/bin/xbridge


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
#%_inc.cpp: %.mlir
aie_inc.cpp: aie.mlir
	$(AIE_OPT) --aie-create-flows --aie-find-flows $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

#%.mlir_addresses : %.mlir
#	$(AIE_OPT) --aie-assign-buffer-addresses $*.mlir > $@
%.mlir_addresses : aie.mlir
	$(AIE_OPT) --aie-assign-buffer-addresses aie.mlir > $@

%13.ll : %13.mlir_addresses
	$(AIE_OPT) --aie-llvm-lowering="tilecol=1 tilerow=3" $^ | $(AIE_XLATE) --aie-generate-llvmir | opt -strip -S > $*13.ll

%14.ll : %14.mlir_addresses
	$(AIE_OPT) --aie-llvm-lowering="tilecol=1 tilerow=4" $^ | $(AIE_XLATE) --aie-generate-llvmir | opt -strip -S > $*14.ll

%.o : %.ll
	llc $*.ll -O2 --march=aie -align-all-blocks=4 --filetype=obj -o=$*.o

%13.bcf : %13.mlir_addresses
	$(AIE_XLATE) --aie-generate-bcf --tilecol=1 --tilerow=3 $< > $@

%14.bcf : %14.mlir_addresses
	$(AIE_XLATE) --aie-generate-bcf --tilecol=1 --tilerow=4 $< > $@

%.elf : %.o %.bcf
	$(XBRIDGE) $< -c $*.bcf -o $@

#ld.lld $*.o $(AIE_BASE_OBJ) -T ./ld.script -o $@

#%.bin : %.mlir
#	$(AIE_OPT) --aie-llvm-lowering="tilecol=1 tilerow=3" $^ | $(AIE_XLATE) --aie-generate-llvmir | opt -strip -S | llc -O0 --march=aie --filetype=obj -o=$*.o && \
#	ld.lld $*.o $(AIE_BASE_OBJ) -T ./ld.script --oformat binary -o $@

.PRECIOUS: aie.ll aie.o

# generate kernel source
#%.elf: $(AIE_SRC_DIR)/%.cc
#	cd ./$(AIE_SRC_DIR) && \
#	xchessmk $*.prx && \
#	cp work/Release_LLVM/$*.prx/$* ../$*.elf

clean:
	rm -rf aie_inc.cpp *.o *.ll *.mlir_addresses *.exe *.elf* ./$(AIE_SRC_DIR)/work kernel test.mem *bcf
