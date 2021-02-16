#!/bin/bash

CARDANO=/proj/xbuilds/2020.1_daily_latest/installs/lin64/Vitis/2020.1/cardano/
${CARDANO}/bin/xchessmk kernel13.prx
${CARDANO}/bin/xchessmk kernel23.prx
cp ./work/Release_LLVM/kernel13.prx/kernel13 ../core_1_3.elf
cp ./work/Release_LLVM/kernel23.prx/kernel23 ../core_2_3.elf
#${CARDANO}/bin/xca_udm_dbg -t sim.tcl
