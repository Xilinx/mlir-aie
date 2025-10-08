
iss::create %PROCESSORNAME% iss
#iss program load ./work/Release_LLVM/test.prx/test -disassemble -dwarf -nmlpath /proj/xbuilds/SWIP/2020.1_0602_1208/installs/lin64/Vitis/2020.1/cardano/data/versal_prod/lib -extradisassembleopts +Mdec -do_not_set_entry_pc 1 -do_not_load_sp 1 -pm_check first -load_offsets {} -software_breakpoints_allowed on -hardware_breakpoints_allowed on
iss program load ./work/Release_LLVM/test.prx/test -disassemble -dwarf -nmlpath %XILINX_VITIS%/aietools/data/aie_ml/lib -extradisassembleopts +Mdec -do_not_set_entry_pc 1 -do_not_load_sp 1 -pm_check first -load_offsets {} -software_breakpoints_allowed on -hardware_breakpoints_allowed on
iss step -1
iss profile save test.prf
iss close
exit

