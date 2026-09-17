# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

        .text
        .globl helper_cycle
        .type helper_cycle,@function
helper_cycle:
        .long entry_real
        .size helper_cycle, .-helper_cycle

        .globl entry_real
        .type entry_real,@function
entry_real:
        .long inlined_entry
        .size entry_real, .-entry_real

        .globl inlined_entry
        .type inlined_entry,@function
        .set inlined_entry, helper_cycle
        .size inlined_entry, 0

        .pushsection .stack_sizes,"",@progbits
        .long helper_cycle
        .byte 0
        .long entry_real
        .byte 0
        .popsection
