include ../../test/ipu-xrt/makefile-common

all: hello_world_elfstrings.csv

kernel.o: kernel.cc
	xchesscc ${CHESSCC2_FLAGS} -c kernel.cc -o kernel.o

hello_world.mlir: hello_world.py
	python3 $< > $@

hello_world.xclbin: hello_world.mlir kernel.o
	aiecc.py --aie-generate-cdo --aie-generate-ipu --no-compile-host \
		--xclbin-name=$@ --ipu-insts-name=insts.txt $<

hello_world_elfstrings.csv: hello_world.xclbin
	python3 elfStringParser.py --input ./ --output $@

helloWorld.exe: test.cpp
	rm -rf _build
	mkdir _build
	cd _build && powershell.exe cmake ../
	cd _build && powershell.exe cmake --build . --config Release
	cp _build/Release/helloWorld.exe ./

run: helloWorld.exe hello_world_elfstrings.csv
	powershell.exe ./helloWorld.exe -x hello_world.xclbin -i insts.txt -k PP_FD_PRE -e hello_world_elfstrings.csv

clean:
	rm -rf *.elf* *.bif aie.mlir.prj *.xclbin insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x hello_world.mlir \
		*.csv *.mlir.prj *.o _build CMakeCache.txt CMakeFiles *.exe
