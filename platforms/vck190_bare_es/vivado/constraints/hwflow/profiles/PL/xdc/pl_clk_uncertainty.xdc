#
# Copyright (C) 2021 Xilinx, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 

set_clock_uncertainty -hold  0.050 -from [get_clocks *pl_0] -to [get_clocks *pl_0]
#set_clock_uncertainty -setup  3.0 -from [get_clocks *pl_0] -to [get_clocks *pl_0]

set_clock_uncertainty -hold  0.050 -from [get_clocks *pl_1] -to [get_clocks *pl_1]
#set_clock_uncertainty -setup  3.0 -from [get_clocks *pl_1] -to [get_clocks *pl_1]

set_clock_uncertainty -hold  0.050 -from [get_clocks *pl_2] -to [get_clocks *pl_2]
#set_clock_uncertainty -setup  3.0 -from [get_clocks *pl_2] -to [get_clocks *pl_2]

set_clock_uncertainty -hold  0.050 -from [get_clocks *pl_3] -to [get_clocks *pl_3]
#set_clock_uncertainty -setup  3.0 -from [get_clocks *pl_3] -to [get_clocks *pl_3]

