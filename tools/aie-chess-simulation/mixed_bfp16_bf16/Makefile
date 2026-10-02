all: build sim
.PHONY : all build sim clean

build: 
	rm -rf output && mkdir output && xchesscc --aiearch aie2p -p me -C Release_LLVM -D__AIENGINE__ - -I ${AIETOOLS_ROOT}/include -I ${AIETOOLS_ROOT}/include/aie_api -P ${AIETOOLS_ROOT}/data/aie2p/lib -d -f -g +s +w work +o work -I. -I $../../../.. test.cc
	
sim: 
	xca_udm_dbg --aiearch aie2p -qf -T -P ${AIETOOLS_ROOT}/data/aie2p/lib -t "sim.tcl work/a.out"

clean:
	rm -rf work *txt *mem *.output output 




