#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# (c) Copyright 2021 Xilinx Inc.
# 

##source [pwd]/myproj/project_1.srcs/utils_1/imports/aie_vnc/waive_hold_violations_sprite.tcl
##source [pwd]/myproj/project_1.srcs/utils_1/imports/aie_vnc/waive_extra_margin_hold_violations.tcl

##waive_hold_violations_sprite.tcl

# David Pefourque
# This script can be used to report or waive Hold violations that are not on intrasites or cascaded paths
# Version: 08/14/2019
########################################################################################
## 2019.08.20 - Removed report_hold_violations, calling waive_hold_violations - SPRITE
## 2019.08.14 - Added waive_hold_violations to add Min Delay on safe intra-site/cascaded
##              Hold violations
##            - Added support for cascaded DSPs (report_hold_violations)
##            - Change command line arguments (report_hold_violations)
##            - Added new categories for safe/unsafe intra-site/cascaded Hold violations
##              (report_hold_violations)
##            - Changed the default number of paths to 10000 (report_hold_violations)
##            - Code refactorization
## 2019.05.21 - Initial release
########################################################################################

# report_hold_violations [-prefix <prefix>] [-max <max_paths>]
#     Default: <prefix> = hold
#              <max_paths> = 10000
# If <max_paths>=-1 then report_timing_summary is run to extract the number
# of failing endpoints for Hold and report_hold_violations is run on all
# the failing endpoints.
#
# E.g: report_hold_violations -prefix hold -max 300

# waive_hold_violations [-max <max_paths>]
#     Default: <max_paths> = 10000
# If <max_paths>=-1 then report_timing_summary is run to extract the number
# of failing endpoints for Hold and waive_hold_violations is run on all
# the failing endpoints.
#
# E.g: waive_hold_violations -max 300

# Force analysis on DSPs:
#   set dsps [get_cells -hier -filter {IS_PRIMITIVE && PRIMITIVE_SUBGROUP==DSP}] ; llength $dsps
#   set paths [get_timing_paths -hold -from $dsps -to $dsps -slack_less_than 0.0 -max_paths 1000 -nworst 1]
#   # Timing paths passed through reference (not $paths)
#   report_hold_violations hold - paths

proc find_hold_violations { &paths &db args } {

  upvar 1 ${&paths} paths
  upvar 1 ${&db} db

  array set defaults [list \
      -verbose 0 \
      -debug 0 \
      -uncertainty 0.050 \
    ]
  array set options [array get defaults]
  array set options $args

  # Only consider paths with user uncertainty
  if {$options(-uncertainty)} {
    set FAILING_HOLD_PATHS [filter $paths {USER_UNCERTAINTY != {}}]
    puts " -I- find_hold_violations: [llength $FAILING_HOLD_PATHS] failing hold paths considered (with user uncertainty)"
#     set FAILING_HOLD_PATHS [filter $paths {(USER_UNCERTAINTY != {}) && (USER_UNCERTAINTY != 0)}]
  } else {
    set FAILING_HOLD_PATHS $paths
  }

  if {$options(-debug)} {
    puts " -I- find_hold_violations: [llength [filter $paths {USER_UNCERTAINTY == {}}]] failing hold paths without user uncertainty"
    puts " -I- find_hold_violations: [llength [filter $paths {USER_UNCERTAINTY != {}}]] failing hold paths with user uncertainty"
  }

  set sps [get_property -quiet STARTPOINT_PIN $FAILING_HOLD_PATHS]
  set eps [get_property -quiet ENDPOINT_PIN $FAILING_HOLD_PATHS]
  set slacks [get_property -quiet SLACK $FAILING_HOLD_PATHS]
  set uncertainties [get_property -quiet USER_UNCERTAINTY $FAILING_HOLD_PATHS]
  # Intra-site paths that have a negative slack caused by the user clock uncertainty: ok to waive
  catch {unset arrIntOK}
  # Intra-site paths that have a negative slack that is not entirely caused by the user clock uncertainty: shouldn't be waived
  catch {unset arrInt}
  # Cascaded paths that have a negative slack caused by the user clock uncertainty: ok to waive
  catch {unset arrCasOK}
  # Cascaded paths that have a negative slack that is not entirely caused by the user clock uncertainty: shouldn't be waived
  catch {unset arrCas}
  catch {unset arrVio}
  set arrIntOK(-) [list]
  set arrInt(-) [list]
  set arrCasOK(-) [list]
  set arrCas(-) [list]
  set arrVio(-) [list]
  set idx -1
  foreach PATH $FAILING_HOLD_PATHS {
    incr idx
    set sp [lindex $sps $idx]
    set ep [lindex $eps $idx]
    # Get the absolute value of the path slack to make it easier to compare to the user clock uncertainty
    set slack [expr abs([lindex $slacks $idx])]
    set uncertainty [lindex $uncertainties $idx]
    set NETS [get_nets -of $PATH]
    if {[lsort -unique [get_property -quiet ROUTE_STATUS $NETS]] == "INTRASITE"} {
      # Intra-site path
      if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is INTRASITE" }
      set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
      if {$slack <= $uncertainty} {
        # Safe
        lappend arrIntOK($key) $PATH
        lappend arrIntOK(-) $PATH
      } else {
        # Unsafe
        lappend arrInt($key) $PATH
        lappend arrInt(-) $PATH
      }
    } elseif {[llength $NETS] == 1} {
      # Single net: verify the endpoint pin name
      switch -regexp -- [get_property -quiet REF_PIN_NAME $ep] {
        {^ACIN.+$} -
        {^BCIN.+$} -
        {^PCIN.+$} -
        {^CARRYCAS.+$} -
        {^CAS.+$} {
          # Cascaded path
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          if {$slack <= $uncertainty} {
            # Safe
            lappend arrCasOK($key) $PATH
            lappend arrCasOK(-) $PATH
          } else {
            # Unsafe
            lappend arrCas($key) $PATH
            lappend arrCas(-) $PATH
          }
        }
        default {
          # Valid hold violation
          if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is not INTRASITE" }
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          lappend arrVio($key) $PATH
          lappend arrVio(-) $PATH
        }
      }
    } else {
      # Multiple nets
      set CELLS [get_cells -quiet -of $PATH]
      if {([llength [lsort -unique [get_property -quiet PARENT $CELLS]]] == 2) && ([lsort -unique [get_property -quiet PRIMITIVE_SUBGROUP $CELLS]] == {DSP})} {
        # The path is made of 2 different DSP macros
        # If one of the ACIN/BCIN/CARRYCASCIN/PCIN is found on the path, it means that the 2 DSPs macros are cascaded
        set PINS [get_property -quiet BUS_NAME [get_pins -quiet -of $PATH]]
        if {[regexp {(ACIN|BCIN|CARRYCASCIN|PCIN)} $PINS]} {
          # The 2 DSPs macros are cascaded
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          if {$slack <= $uncertainty} {
            # Safe
            lappend arrCasOK($key) $PATH
            lappend arrCasOK(-) $PATH
          } else {
            # Unsafe
            lappend arrCas($key) $PATH
            lappend arrCas(-) $PATH
          }
        } else {
          # The 2 DSPs macros are not cascaded -> valid hold violation
          if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is not INTRASITE" }
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          lappend arrVio($key) $PATH
          lappend arrVio(-) $PATH
        }
      } else {
        # Valid hold violation
        if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is not INTRASITE" }
        set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
        lappend arrVio($key) $PATH
        lappend arrVio(-) $PATH
      }
    }
  }

  set db(arrVio)   [array get arrVio]
  set db(arrCasOK) [array get arrCasOK]
  set db(arrCas)   [array get arrCas]
  set db(arrIntOK) [array get arrIntOK]
  set db(arrInt)   [array get arrInt]

  return -code ok
}

proc waive_hold_violations { args } {

  if {$args != {}} {
    puts " -I- waive_hold_violations: $args"
  }

  array set defaults [list \
      -verbose 0 \
      -debug 0 \
      -uncertainty 0.050 \
      -max 10000 \
      -paths {} \
    ]
  array set options [array get defaults]
  array set options $args

  # Paths passed by reference: &paths -> variable name in caller space
  set &paths $options(-paths)
  upvar 1 ${&paths} paths

  set max $options(-max)
  set uncertainty $options(-uncertainty)
  set verbose $options(-verbose)
  set debug $options(-debug)
  set start [clock seconds]

  if {![info exists paths]} {
    set paths {}
  }

  if {$paths != {}} {
    puts " -I- waive_hold_violations: using user timing paths"
    set FAILING_HOLD_PATHS $paths
  } elseif {$max == -1} {
    puts " -I- waive_hold_violations: running report_timing_summary"
    set report [report_timing_summary -quiet -no_check_timing -no_detailed_paths -return_string]
    set report [split $report \n]
    if {[set i [lsearch -regexp $report {Design Timing Summary}]] != -1} {
       foreach {wns tns tnsFallingEp tnsTotalEp whs ths thsFallingEp thsTotalEp wpws tpws tpwsFailingEp tpwsTotalEp} [regexp -inline -all -- {\S+} [lindex $report [expr $i + 6]]] { break }
    }
    if {[info exist thsFallingEp] && ($thsFallingEp != 0)} {
      puts " -I- waive_hold_violations: $thsFallingEp failing Hold endpoints extracted from report_timing_summary"
      set FAILING_HOLD_PATHS [get_timing_paths -quiet -hold -slack_less_than 0.0 -max_paths $thsFallingEp -nworst 1]
    } else {
      puts " -I- waive_hold_violations: no failing Hold path found"
      return -code ok
    }
  } else {
    set FAILING_HOLD_PATHS [get_timing_paths -quiet -hold -slack_less_than 0.0 -max_paths $max -nworst 1]
  }
#   set FAILING_HOLD_PATHS [get_timing_paths -quiet -hold -slack_less_than 0.0 -max_paths $max -nworst 1]

  puts " -I- waive_hold_violations: [llength $FAILING_HOLD_PATHS] failing hold paths found"

  catch {unset db}
  catch {unset arrIntOK}
  catch {unset arrInt}
  catch {unset arrCasOK}
  catch {unset arrCas}
  catch {unset arrVio}
  find_hold_violations FAILING_HOLD_PATHS db -uncertainty $uncertainty -verbose $verbose -debug $debug
  array set arrIntOK $db(arrIntOK)
  array set arrInt $db(arrInt)
  array set arrCasOK $db(arrCasOK)
  array set arrCas $db(arrCas)
  array set arrVio $db(arrVio)

  puts " -I- waive_hold_violations: [llength $arrVio(-)] failing hold paths are real violations"
  puts " -I- waive_hold_violations: [llength $arrCasOK(-)] failing hold paths that are safe to waive (CASCADED)"
  puts " -I- waive_hold_violations: [llength $arrCas(-)] failing hold paths that are UNSAFE to waive (CASCADED)"
  puts " -I- waive_hold_violations: [llength $arrIntOK(-)] failing hold paths that are safe to waive (INTRASITE)"
  puts " -I- waive_hold_violations: [llength $arrInt(-)] failing hold paths that are UNSAFE to waive (INTRASITE)"

  # Adding Min Delay constraint on safe path endpoint
  if {[llength $arrCasOK(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrCasOK(-)] u [get_property -quiet USER_UNCERTAINTY $arrCasOK(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] safe endpoints (CASCADED)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  # Adding Min Delay constraint on unsafe path endpoint
  if {[llength $arrCas(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrCas(-)] u [get_property -quiet USER_UNCERTAINTY $arrCas(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] unsafe endpoints (CASCADED)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  # Adding Min Delay constraint on safe path endpoint
  if {[llength $arrIntOK(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrIntOK(-)] u [get_property -quiet USER_UNCERTAINTY $arrIntOK(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] safe endpoints (INTRASITE)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  # Adding Min Delay constraint on unsafe path endpoint
  if {[llength $arrInt(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrInt(-)] u [get_property -quiet USER_UNCERTAINTY $arrInt(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] unsafe endpoints (INTRASITE)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  set end [clock seconds]
  puts " -I- waive_hold_violations completed in [expr $end - $start] seconds"
  return -code ok
}

waive_hold_violations

#waive_extra_margin_hold_violations.tcl

########################################################################################
## Author: Xilinx
## This script can be used to report or waive Hold violations that are not on intrasites
##             or cascaded paths
## Version: 08/20/2019
########################################################################################
## 2019.12.10 - Initial release to Versal Lounge
## 2019.08.20 - Internal version complete
########################################################################################

# waive_hold_violations [-max <max_paths>]
#     Default: <max_paths> = 10000
# If <max_paths>=-1 then report_timing_summary is run to extract the number
# of failing endpoints for Hold and waive_hold_violations is run on all
# the failing endpoints.
#
# E.g: waive_hold_violations -max 300

# Force analysis on DSPs:
#   set dsps [get_cells -hier -filter {IS_PRIMITIVE && PRIMITIVE_SUBGROUP==DSP}] ; llength $dsps
#   set paths [get_timing_paths -hold -from $dsps -to $dsps -slack_less_than 0.0 -max_paths 1000 -nworst 1]
#   # Timing paths passed through reference (not $paths)
#   report_hold_violations hold - paths

proc find_hold_violations { &paths &db args } {

  upvar 1 ${&paths} paths
  upvar 1 ${&db} db

  array set defaults [list \
      -verbose 0 \
      -debug 0 \
      -uncertainty 0.050 \
    ]
  array set options [array get defaults]
  array set options $args

  # Only consider paths with user uncertainty
  if {$options(-uncertainty)} {
    set FAILING_HOLD_PATHS [filter $paths {USER_UNCERTAINTY != {}}]
    puts " -I- find_hold_violations: [llength $FAILING_HOLD_PATHS] failing hold paths considered (with user uncertainty)"
#     set FAILING_HOLD_PATHS [filter $paths {(USER_UNCERTAINTY != {}) && (USER_UNCERTAINTY != 0)}]
  } else {
    set FAILING_HOLD_PATHS $paths
  }

  if {$options(-debug)} {
    puts " -I- find_hold_violations: [llength [filter $paths {USER_UNCERTAINTY == {}}]] failing hold paths without user uncertainty"
    puts " -I- find_hold_violations: [llength [filter $paths {USER_UNCERTAINTY != {}}]] failing hold paths with user uncertainty"
  }

  set sps [get_property -quiet STARTPOINT_PIN $FAILING_HOLD_PATHS]
  set eps [get_property -quiet ENDPOINT_PIN $FAILING_HOLD_PATHS]
  set slacks [get_property -quiet SLACK $FAILING_HOLD_PATHS]
  set uncertainties [get_property -quiet USER_UNCERTAINTY $FAILING_HOLD_PATHS]
  # Intra-site paths that have a negative slack caused by the user clock uncertainty: ok to waive
  catch {unset arrIntOK}
  # Intra-site paths that have a negative slack that is not entirely caused by the user clock uncertainty: shouldn't be waived
  catch {unset arrInt}
  # Cascaded paths that have a negative slack caused by the user clock uncertainty: ok to waive
  catch {unset arrCasOK}
  # Cascaded paths that have a negative slack that is not entirely caused by the user clock uncertainty: shouldn't be waived
  catch {unset arrCas}
  catch {unset arrVio}
  set arrIntOK(-) [list]
  set arrInt(-) [list]
  set arrCasOK(-) [list]
  set arrCas(-) [list]
  set arrVio(-) [list]
  set idx -1
  foreach PATH $FAILING_HOLD_PATHS {
    incr idx
    set sp [lindex $sps $idx]
    set ep [lindex $eps $idx]
    # Get the absolute value of the path slack to make it easier to compare to the user clock uncertainty
    set slack [expr abs([lindex $slacks $idx])]
    set uncertainty [lindex $uncertainties $idx]
    set NETS [get_nets -of $PATH]
    if {[lsort -unique [get_property -quiet ROUTE_STATUS $NETS]] == "INTRASITE"} {
      # Intra-site path
      if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is INTRASITE" }
      set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
      if {$slack <= $uncertainty} {
        # Safe
        lappend arrIntOK($key) $PATH
        lappend arrIntOK(-) $PATH
      } else {
        # Unsafe
        lappend arrInt($key) $PATH
        lappend arrInt(-) $PATH
      }
    } elseif {[llength $NETS] == 1} {
      # Single net: verify the endpoint pin name
      switch -regexp -- [get_property -quiet REF_PIN_NAME $ep] {
        {^ACIN.+$} -
        {^BCIN.+$} -
        {^PCIN.+$} -
        {^CARRYCAS.+$} -
        {^CAS.+$} {
          # Cascaded path
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          if {$slack <= $uncertainty} {
            # Safe
            lappend arrCasOK($key) $PATH
            lappend arrCasOK(-) $PATH
          } else {
            # Unsafe
            lappend arrCas($key) $PATH
            lappend arrCas(-) $PATH
          }
        }
        default {
          # Valid hold violation
          if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is not INTRASITE" }
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          lappend arrVio($key) $PATH
          lappend arrVio(-) $PATH
        }
      }
    } else {
      # Multiple nets
      set CELLS [get_cells -quiet -of $PATH]
      if {([llength [lsort -unique [get_property -quiet PARENT $CELLS]]] == 2) && ([lsort -unique [get_property -quiet PRIMITIVE_SUBGROUP $CELLS]] == {DSP})} {
        # The path is made of 2 different DSP macros
        # If one of the ACIN/BCIN/CARRYCASCIN/PCIN is found on the path, it means that the 2 DSPs macros are cascaded
        set PINS [get_property -quiet BUS_NAME [get_pins -quiet -of $PATH]]
        if {[regexp {(ACIN|BCIN|CARRYCASCIN|PCIN)} $PINS]} {
          # The 2 DSPs macros are cascaded
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          if {$slack <= $uncertainty} {
            # Safe
            lappend arrCasOK($key) $PATH
            lappend arrCasOK(-) $PATH
          } else {
            # Unsafe
            lappend arrCas($key) $PATH
            lappend arrCas(-) $PATH
          }
        } else {
          # The 2 DSPs macros are not cascaded -> valid hold violation
          if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is not INTRASITE" }
          set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
          lappend arrVio($key) $PATH
          lappend arrVio(-) $PATH
        }
      } else {
        # Valid hold violation
        if {$options(-verbose)} { puts "Failing hold path ([get_property -quiet SLACK $PATH]) $PATH is not INTRASITE" }
        set key [get_property -quiet REF_NAME $sp]/[get_property -quiet REF_PIN_NAME $sp]->[get_property -quiet REF_NAME $ep]/[regsub {\[[0-9]+\]$} [get_property -quiet REF_PIN_NAME $ep] {}]
        lappend arrVio($key) $PATH
        lappend arrVio(-) $PATH
      }
    }
  }

  set db(arrVio)   [array get arrVio]
  set db(arrCasOK) [array get arrCasOK]
  set db(arrCas)   [array get arrCas]
  set db(arrIntOK) [array get arrIntOK]
  set db(arrInt)   [array get arrInt]

  return -code ok
}

proc waive_hold_violations { args } {

  if {$args != {}} {
    puts " -I- waive_hold_violations: $args"
  }

  array set defaults [list \
      -verbose 0 \
      -debug 0 \
      -uncertainty 0.050 \
      -max 10000 \
      -paths {} \
    ]
  array set options [array get defaults]
  array set options $args

  # Paths passed by reference: &paths -> variable name in caller space
  set &paths $options(-paths)
  upvar 1 ${&paths} paths

  set max $options(-max)
  set uncertainty $options(-uncertainty)
  set verbose $options(-verbose)
  set debug $options(-debug)
  set start [clock seconds]

  if {![info exists paths]} {
    set paths {}
  }

  if {$paths != {}} {
    puts " -I- waive_hold_violations: using user timing paths"
    set FAILING_HOLD_PATHS $paths
  } elseif {$max == -1} {
    puts " -I- waive_hold_violations: running report_timing_summary"
    set report [report_timing_summary -quiet -no_check_timing -no_detailed_paths -return_string]
    set report [split $report \n]
    if {[set i [lsearch -regexp $report {Design Timing Summary}]] != -1} {
       foreach {wns tns tnsFallingEp tnsTotalEp whs ths thsFallingEp thsTotalEp wpws tpws tpwsFailingEp tpwsTotalEp} [regexp -inline -all -- {\S+} [lindex $report [expr $i + 6]]] { break }
    }
    if {[info exist thsFallingEp] && ($thsFallingEp != 0)} {
      puts " -I- waive_hold_violations: $thsFallingEp failing Hold endpoints extracted from report_timing_summary"
      set FAILING_HOLD_PATHS [get_timing_paths -quiet -hold -slack_less_than 0.0 -max_paths $thsFallingEp -nworst 1]
    } else {
      puts " -I- waive_hold_violations: no failing Hold path found"
      return -code ok
    }
  } else {
    set FAILING_HOLD_PATHS [get_timing_paths -quiet -hold -slack_less_than 0.0 -max_paths $max -nworst 1]
  }
#   set FAILING_HOLD_PATHS [get_timing_paths -quiet -hold -slack_less_than 0.0 -max_paths $max -nworst 1]

  puts " -I- waive_hold_violations: [llength $FAILING_HOLD_PATHS] failing hold paths found"

  catch {unset db}
  catch {unset arrIntOK}
  catch {unset arrInt}
  catch {unset arrCasOK}
  catch {unset arrCas}
  catch {unset arrVio}
  find_hold_violations FAILING_HOLD_PATHS db -uncertainty $uncertainty -verbose $verbose -debug $debug
  array set arrIntOK $db(arrIntOK)
  array set arrInt $db(arrInt)
  array set arrCasOK $db(arrCasOK)
  array set arrCas $db(arrCas)
  array set arrVio $db(arrVio)

  puts " -I- waive_hold_violations: [llength $arrVio(-)] failing hold paths are real violations"
  puts " -I- waive_hold_violations: [llength $arrCasOK(-)] failing hold paths that are safe to waive (CASCADED)"
  puts " -I- waive_hold_violations: [llength $arrCas(-)] failing hold paths that are UNSAFE to waive (CASCADED)"
  puts " -I- waive_hold_violations: [llength $arrIntOK(-)] failing hold paths that are safe to waive (INTRASITE)"
  puts " -I- waive_hold_violations: [llength $arrInt(-)] failing hold paths that are UNSAFE to waive (INTRASITE)"

  # Adding Min Delay constraint on safe path endpoint
  if {[llength $arrCasOK(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrCasOK(-)] u [get_property -quiet USER_UNCERTAINTY $arrCasOK(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] safe endpoints (CASCADED)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  # Adding Min Delay constraint on unsafe path endpoint
  if {[llength $arrCas(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrCas(-)] u [get_property -quiet USER_UNCERTAINTY $arrCas(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] unsafe endpoints (CASCADED)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  # Adding Min Delay constraint on safe path endpoint
  if {[llength $arrIntOK(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrIntOK(-)] u [get_property -quiet USER_UNCERTAINTY $arrIntOK(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] safe endpoints (INTRASITE)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  # Adding Min Delay constraint on unsafe path endpoint
  if {[llength $arrInt(-)]} {
    catch {unset arr}
    foreach p [get_property -quiet ENDPOINT_PIN $arrInt(-)] u [get_property -quiet USER_UNCERTAINTY $arrInt(-)] {
      lappend arr($u) $p
    }
    foreach u [lsort -real [array names arr]] {
      puts " -I- waive_hold_violations: Added Min Delay constraint of $u on [llength $arr($u)] unsafe endpoints (INTRASITE)"
      # Change the sign to convert the user uncertainty to the Min Delay value
      set_min_delay [expr -1.0 * $u] -to $arr($u)
    }
  }

  set end [clock seconds]
  puts " -I- waive_hold_violations completed in [expr $end - $start] seconds"
  return -code ok
}

waive_hold_violations
