#!/bin/bash

CARDANO=/proj/xbuilds/2020.1_daily_latest/installs/lin64/Vitis/2020.1/cardano/
#${CARDANO}/bin/xchessmk kernel.prx
${CARDANO}/bin/xchesscc -p me -P ${CARDANO}/data/cervino/lib -c kernel.cc |& tee xchesscc.log
#${CARDANO}/bin/xca_udm_dbg -t sim.tcl

