#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# (c) Copyright 2021 Xilinx Inc.
# 

opt_design -srl_remap_modes { {max_depth_srl_to_ffs 4} } -sweep
##source [pwd]/myproj/project_1.srcs/utils_1/imports/aie_vnc/prohibitCascBramAcrossRbrk.tcl
##source [pwd]/myproj/project_1.srcs/utils_1/imports/aie_vnc/prohibitCascUramAcrossRbrk.tcl
##source [pwd]/myproj/project_1.srcs/utils_1/imports/aie_vnc/prohibitCascDspAcrossRbrk.tcl
##source [pwd]/myproj/project_1.srcs/utils_1/imports/aie_vnc/waive_BLI_AIE_timing_violations_preplace.tcl

##prohibitCascBramAcrossRbrk.tcl
catch {unset ys}
foreach cr [get_clock_regions {X*Y1 X*Y2 X*Y3}] {
  set bram [get_sites -quiet -filter NAME=~RAMB36_X*Y* -of $cr]
  if {$bram == {}} { continue }
  lappend ys [lindex [lsort -integer -increasing [regsub -all {RAMB36_X\d+Y(\d+)} $bram {\1}]] end]
}
foreach y $ys {
  set_property PROHIBIT TRUE [get_sites RAMB36_X*Y$y]
  set_property PROHIBIT TRUE [get_sites RAMB18_X*Y[expr 2 * $y]]
  set_property PROHIBIT TRUE [get_sites RAMB18_X*Y[expr 2 * $y + 1]]
}

##prohibitCascUramAcrossRbrk.tcl
catch {unset ys}
foreach cr [get_clock_regions {X*Y1 X*Y2 X*Y3}] {
  set uram [get_sites -quiet -filter NAME=~URAM288_X*Y* -of $cr]
  if {$uram == {}} { continue }
  lappend ys [lindex [lsort -integer -increasing [regsub -all {URAM288_X\d+Y(\d+)} $uram {\1}]] end]
}
foreach y $ys {
  set_property PROHIBIT TRUE [get_sites URAM288_X*Y$y]
}

##prohibitCascDspAcrossRbrk.tcl
catch {unset ys}
foreach cr [get_clock_regions {X*Y1 X*Y2 X*Y3}] {
  set dsp [get_sites -quiet -filter NAME=~DSP_X*Y* -of $cr]
  if {$dsp == {}} { continue }
  lappend ys [lindex [lsort -integer -increasing [regsub -all {DSP_X\d+Y(\d+)} $dsp {\1}]] end]
}
foreach y $ys {
  set_property PROHIBIT TRUE [get_sites DSP*_X*Y$y]
}

##waive_BLI_AIE_timing_violations_preplace.tcl
# Last update; 2019/09/06
# Last change: checking on AIE_PL BEL location to make sure there is a BLI site available
#
# Last update; 2019/11/20
# Last change: adding support for 128b interfaces
#
set debug 0
foreach aiePL [get_cells -quiet -hier -filter "REF_NAME=~AIE_PL_* && PRIMITIVE_LEVEL!=MACRO"] {
  set loc [get_property LOC $aiePL]
  if {$loc == ""} { if {$debug} { puts "D - Missing LOC - $aiePL" }; continue } ;# Unplace AIE_PL cell => unsafe to waive timing
  set bel [get_property BEL $aiePL]
  if {$bel == "AIE_PL.AIE_PL_S_AXIS_3"} { if {$debug} { puts "D - BEL 3 - $aiePL" }; continue } ;# No BLI register site available
  if {$bel == "AIE_PL.AIE_PL_S_AXIS_7"} { if {$debug} { puts "D - BEL 7 - $aiePL" }; continue } ;# No BLI register site available
  foreach dir {IN OUT} {
    set bliFD [get_cells -quiet -filter "REF_NAME=~FD* && BLI==TRUE" -of [get_pins -leaf -of [get_nets -filter "FLAT_PIN_COUNT==2" -of [get_pins -filter "!IS_CLOCK && DIRECTION==$dir" -of $aiePL]]]]
    if {$bliFD == {}} { if {$debug} { puts "D - no BLI FD - $dir - $aiePL" }; continue }
    set refName [get_property REF_NAME $aiePL]
    set locBel "$loc/[regsub {.*\.} $bel {}]"
    if {$dir == "IN"} {
      puts [format "INFO - Adding False Path waiver from %2s BLI registers to   $refName ($locBel) - $aiePL" [llength $bliFD]]
      set_false_path -from $bliFD -to $aiePL
    } else {
      puts [format "INFO - Adding False Path waiver to   %2s BLI registers from $refName ($locBel) - $aiePL" [llength $bliFD]]
      set_false_path -from $aiePL -to $bliFD
    }
  }
}

