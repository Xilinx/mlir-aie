
include ../../makefile-common

.PHONY: build gui sim

mlir: kernel.o

%.o: %.cc
	xchesscc ${CHESSCC_FLAGS} -c $<

build:
	xchessmk -P ${CHESS_FLAGS} test.prx

gui:
	xchessde -P ${CHESS_FLAGS} test.prx &

sim:
	xca_udm_dbg -P ${CHESS_FLAGS} -t sim.tcl

clean:
	rm -rf work kernel.o test.prf
