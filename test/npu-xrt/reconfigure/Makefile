
aie.elf: aie.mlir
	aiecc.py -v --generate-full-elf --no-xchesscc --no-xbridge aie.mlir

test: test.cpp
	clang -o test test.cpp -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_core -lxrt_coreutil

.PHONY: baboo
baboo:
	aie-opt \
		--pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))" \
		broken-aie.mlir.prj/pipeline_outputs/04_aie-assign-runtime-sequence-bd-ids.mlir \
		-o npu_insts.mlir

.PHONY: babee
babee:
	aie-translate --aie-npu-to-binary npu_insts.mlir --aie-device-name main --aie-sequence-name sequence -o main_sequence.bin 

.PHONY: babony
babony:
	aiebu-asm -t aie2_config -j /scratch/roesti/mlir-aie/test/npu-xrt/reconfigure/config.json -o aie.elf

.PHONY: run
run: test aie.elf
	./test 

.PHONY: clean
clean:
	rm -rf test aie.mlir.prj aie.elf