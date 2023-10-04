#
# Tcl srcipt for standalone ISS.
#

proc usage { } {
    puts ""
    puts "Usage: <ISS-NAME> -T -P<libdir> -t \"iss.tcl <program> \""
    puts ""
}

if { [llength $::iss::tcl_script_args] == 0 } {
    puts "ERROR (standalone.tcl) script argument <program> is missing"
    usage
    iss close
    exit 5
} elseif { [llength $::iss::tcl_script_args] == 1 } {
    set program $::iss::tcl_script_args
} elseif { [llength $::iss::tcl_script_args] == 2 } {
    set args [split $::iss::tcl_script_args]
    set program [lindex $args 0]
    set fileio [lindex $args 1]
} else {
    puts "ERROR (standalone.tcl) too many arguments"
    usage
    iss close
    exit 5
}

# Create ISS
set procname [lindex [::iss::processors] 0]
::iss::create $procname iss

set procdir [iss info processor_dir {} 0]

set mem_file $program.mem
set rcd_file $program.iss.rcd
set prf_human      profile.prf
set prf_xml        profile.xml
set prf_instr_xml  profile_instr.xml
set ipr_file       profile.ipr

# Load program
puts [llength $::iss::tcl_main_argv_strings]
puts $::iss::tcl_main_argv_strings
if {[llength $::iss::tcl_main_argv_strings] > 0} {
    iss program load $program \
                        -nmlpath $procdir \
                        -dwarf2 \
                        -do_not_load_sp 1 \
                        -disassemble \
                        -sourcepath {.} \
                        -load_main_argv_strings -main_argv_strings $::iss::tcl_main_argv_strings
} else {
    iss program load $program \
                        -nmlpath $procdir \
                        -dwarf2 \
                        -do_not_load_sp 1 \
                        -disassemble \
                        -sourcepath {.} \
}

if {[info exists fileio]} {
    source $fileio
}

# File outputs for ISS versus RTL tests
#iss fileoutput go -file $rcd_file \
#                   -format go_verilog  \
#                   -registers true \
#                   -memories true \
#                   -skip_list { PC }

# Simulate until end of main
catch { iss step -1 } msg
puts $msg

# CRVO-3521 Detect assert in C and exit with an error
set rt 0
foreach bp [iss breakpoint mic get] {
    set d [::tclutils::list2dict $::iss::mic_breakpoint_keys $bp]
    if {[dict get $d hit_last_cycle]} {
        if {[dict get $d chess_assert]} {
            puts "ASSERT"
            set rt 123
            exit $rt
            break
        }
    }
}

catch { iss step 3 } msg
puts $msg

# Save instruction profile in human readable form
iss profile save $prf_human
# Save instruction profile in xml form
iss profile save $prf_xml       -type function_profiling -xml 1 -function_details 0  -one_file 1 -call_details Off
iss profile save $prf_instr_xml -type function_details   -xml 1 -user_cycle_count On -one_file 1 -source_refs  Off

# Generate instruction profile in form usable for coverage analysis
iss profile save -type profile-Risk $ipr_file

iss close
exit

