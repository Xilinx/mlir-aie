include ../../../utils/vitisVariables.config

.PHONY: build gui sim

MY_VITIS_BIN_DIR := `dirname \`which vitis\``

mlir: kernel.o

%.o: %.cc
	xchesscc -p me -P ${VITIS_AIE_INCLUDE_DIR} -c $<

build:
	xchessmk -P ${VITIS_AIE_INCLUDE_DIR} test.prx

gui:
	xchessde -P ${VITIS_AIE_INCLUDE_DIR} test.prx &

sim:
	xca_udm_dbg -P ${VITIS_AIE_INCLUDE_DIR} -t sim.tcl

clean:
	rm -rf work kernel.o test.prf
