proc my_load_program {file} {
    set me_DIR $::env(me_DIR)
    iss program load $file -nmlpath $me_DIR -do_not_set_entry_pc 1 -pm_check first -load_offsets {}
}

iss::create %PROCESSORNAME% iss
my_load_program [lindex $::iss::tcl_script_args 0]
iss step -1
set retcode [iss program query exit_code]
puts -nonewline "@@ EXIT STATUS "
puts $retcode
exit $retcode

