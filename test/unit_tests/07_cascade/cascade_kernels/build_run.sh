#!/bin/bash

CARDANO=/proj/xbuilds/2020.1_daily_latest/installs/lin64/Vitis/2020.1/cardano/
${CARDANO}/bin/xchessmk kernel13.prx
${CARDANO}/bin/xchessmk kernel23.prx
cp ./work/Release_LLVM/kernel13.prx/kernel13 ../aie13.elf
cp ./work/Release_LLVM/kernel23.prx/kernel23 ../aie23.elf
#${CARDANO}/bin/xca_udm_dbg -t sim.tcl

