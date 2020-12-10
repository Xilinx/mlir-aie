#!/bin/bash
#xchessmk test
VER=2020.1
# FIXME: get rid of this hardcoded path once bridge is included in aie-tools
CARDANO=/proj/xbuilds/${VER}_daily_latest/installs/lin64/Vitis/${VER}/cardano
PATH="${CARDANO}/bin:${CARDANO}/tps/lnx64/target/bin/LNa64bin:${PATH}"

xca_udm_dbg -t sim.tcl

