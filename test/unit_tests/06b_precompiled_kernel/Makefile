
#aie-opt aie.mlir --canonicalize --convert-std-to-llvm=use-bare-ptr-memref-call-conv -canonicalize | aie-translate --aie-generate-llvmir | opt --strip-debug | llvm-dis | llc -O2 -mtriple=aie --filetype=obj -o aie.o

all: core_1_3.elf

core_1_3.elf:
	aie-opt --aie-assign-buffer-addresses --convert-scf-to-std aie.mlir -o input_with_addresses.mlir
	aie-opt input_with_addresses.mlir --canonicalize --convert-std-to-llvm=use-bare-ptr-memref-call-conv -canonicalize -o aie_out.mlir
	aie-opt aie_out.mlir --aie-llvm-lowering="tilecol=1 tilerow=3" -o core_1_3.mlir
	aie-translate --aie-generate-llvmir core_1_3.mlir -o core_1_3.ll
	aie-translate aie_out.mlir --aie-generate-bcf --tilecol=1 --tilerow=3 -o core_1_3.bcf
	opt --strip-debug core_1_3.ll -o core_1_3.stripped_debug.ll
	#opt --strip -S core_1_3.ll -o core_1_3.stripped_S.ll
	llvm-dis core_1_3.stripped_debug.ll -o core_1_3.stripped_debug.ll.ll
	#llc -O2 -mtriple=aie --filetype=obj core_1_3.stripped_debug.ll.ll -o core_1_3.o
	##llc -O2 -march=aie --filetype=obj core_1_3.stripped_debug.ll.ll -o core_1_3.o
	#xbridge core_1_3.o -c core_1_3.bcf -o core_1_3.elf
	aie-translate aie_out.mlir --aie-generate-ldscript --tilecol=1 --tilerow=3 -o core_1_3.ld.script
	clang -O2 --target=aie core_1_3.stripped_debug.ll ./chess_example/kernel.o ../../../runtime_lib/me_basic.o -Wl,-T,core_1_3.ld.script -o core_1_3.elf


clean:
	rm -rf *bcf *ll* *elf input_with_addresses.mlir aie_out.mlir core* *.o *ld.script
