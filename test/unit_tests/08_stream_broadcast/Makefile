
#export LD_LIBRARY_PATH=/opt/xaiengine/lib:$LD_LIBRARY_PATH
CC := gcc
CFLAGS := -std=c++11 -I/opt/xaiengine/include -I../../../runtime_lib

all: test.exe

#acdc_project/aie_inc.cpp: aie.mlir
#	$(AIECC) --no-xbridge aie.mlir

# compile sources
test.o: test.cpp acdc_project/aie_inc.cpp 
	$(CC) $(CFLAGS) -c -o $@ $<

test_library.o: ../../../runtime_lib/test_library.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

# compile and link on board
test.exe: test.o test_library.o
	$(CC) $^ \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-ldl \
		-L/opt/xaiengine/lib \
		-o $@

clean:
	rm -rf *.exe *.elf* acdc_project
