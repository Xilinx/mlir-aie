testdir?=.
xrtflags?=-I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil

%.pdi: ${testdir}/aie.mlir 
	aiecc.py --aie-generate-pdi --pdi-name=${@F} --device-name ${*F} --no-xchesscc --no-xbridge $<

%_sequence.bin: ${testdir}/aie.mlir
	aiecc.py --aie-generate-npu-insts --npu-insts-name=${*F}_sequence.bin --device-name ${*F} --no-xchesscc --no-xbridge  $<

aie.elf: ${testdir}/config.json add_two.pdi add_two_sequence.bin
	aiebu-asm -t aie2_config -j $< -o $@

test: ${testdir}/test_elf.cpp aie.elf
	clang -o $@ $< -std=c++17 -lstdc++ ${xrtflags}

.PHONY: clean
clean:
	rm -rf aie.mlir.prj test *.elf *.pdi *.bin
