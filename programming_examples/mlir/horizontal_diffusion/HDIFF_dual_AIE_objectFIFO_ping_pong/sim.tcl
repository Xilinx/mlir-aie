iss::create %PROCESSORNAME% iss
iss program load ./work/Release_LLVM/test.prx/test -disassemble -dwarf -nmlpath /proj/xbuilds/2021.1_daily_latest/installs/lin64/Vitis/2021.1/aietools/data/versal_prod/lib -extradisassembleopts +Mdec -do_not_set_entry_pc 1 -do_not_load_sp 1 -pm_check first -load_offsets {} -software_breakpoints_allowed on -hardware_breakpoints_allowed on
iss fileinput add SCD 0 -field -file ./dataset_256x256x64.txt -interval_files {} -position 0 -type {} -radix decimal -filter {} -break_on_wrap 0 -cycle_based 0 -format integer -gen_vcd_event 0 -structured 0 -bin_nbr_bytes 1 -bin_lsb_first 0
iss fileoutput add MCD 0 -field -file ./TestOutputS.txt -radix decimal -format integer 
#iss fileoutput add MCD 0 -field -file ./TestOutputS.txt -interval_files {} -type {} -radix decimal -format integer -bin_nbr_bytes 1 -bin_lsb_first 0
iss step -1
iss close