
.PHONY: all gui sim

MY_VITIS_BIN_DIR := `dirname \`which vitis\``

all:
	#xchessmk -P ${XILINX_VITIS}/aietools/data/aie_ml/lib/ test.prx
	#xchessmk -P ${XILINX_LOC}/aietools/data/cervino/lib/ test.prx
	xchessmk -P ${MY_VITIS_BIN_DIR}/../aietools/data/cervino/lib/ test.prx

gui:
	xchessde -P ${MY_VITIS_BIN_DIR}/../aietools/data/cervino/lib/ test.prx &

sim:
	xca_udm_dbg -P ${MY_VITIS_BIN_DIR}/../aietools/data/cervino/lib -t sim.tcl
