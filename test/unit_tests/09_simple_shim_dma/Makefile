
#export LD_LIBRARY_PATH=/opt/xaiengine/lib:$LD_LIBRARY_PATH

CC := gcc
#CFLAGS := -Wall -std=c++11 -I/opt/xaiengine/include
CFLAGS := -std=c++11 -I/opt/xaiengine/include
#CFLAGS := -I/opt/xaienginev2/include

AIE_SRC_DIR  = .
AIE_BASE_OBJ = ../../../runtime_lib/me_basic.o

AIE_OPT     = aie-opt
AIE_XLATE   = aie-translate
LLC         = llc
XBRIDGE     = xbridge
AIECC       = ../../../tools/aiecc/aiecc.py

all: test.exe

#acdc_project/aie_inc.cpp: aie.mlir
#	$(AIECC) aie.mlir
#	$(AIECC) --no-xbridge aie.mlir

# compile sources
%.o: %.cpp acdc_project/aie_inc.cpp
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
acdc_project/aie_inc.cpp: aie.mlir
	$(AIE_OPT) --aie-create-flows --aie-find-flows --aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@
#	$(AIE_OPT) --aie-create-flows --aie-find-flows $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

.PRECIOUS: aie.ll aie.o

clean:
	rm -rf aie_inc.cpp *.o *.ll *.mlir_addresses *.exe *.elf* ./$(AIE_SRC_DIR)/work kernel test.mem *bcf acdc_project
