// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
/*  (c) Copyright 2014 - 2018 Xilinx, Inc. All rights reserved.

    This file contains confidential and proprietary information
    of Xilinx, Inc. and is protected under U.S. and
    international copyright and other intellectual property
    laws.

    DISCLAIMER
    This disclaimer is not a license and does not grant any
    rights to the materials distributed herewith. Except as
    otherwise provided in a valid license issued to you by
    Xilinx, and to the maximum extent permitted by applicable
    law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
    WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
    AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
    BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
    INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
    (2) Xilinx shall not be liable (whether in contract or tort,
    including negligence, or under any other theory of
    liability) for any loss or damage of any kind or nature
    related to, arising under or in connection with these
    materials, including for any direct, or any indirect,
    special, incidental, or consequential loss or damage
    (including loss of data, profits, goodwill, or any type of
    loss or damage suffered as a result of any action brought
    by a third party) even if such damage or loss was
    reasonably foreseeable or Xilinx had been advised of the
    possibility of the same.

    CRITICAL APPLICATIONS
    Xilinx products are not designed or intended to be fail-
    safe, or for use in any application requiring fail-safe
    performance, such as life-support or safety devices or
    systems, Class III medical devices, nuclear facilities,
    applications related to the deployment of airbags, or any
    other applications that could lead to death, personal
    injury, or severe property or environmental damage
    (individually and collectively, "Critical
    Applications"). Customer assumes the sole risk and
    liability of any use of Xilinx products in Critical
    Applications, subject only to applicable laws and
    regulations governing limitations on product liability.

    THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
    PART OF THIS FILE AT ALL TIMES.                       */

/*
-- File : me_basic.c
--
-- Contents : Initial function that calls main for Math Engine (ME).
*/

#include <stdlib.h>

extern "C" {
typedef void (*thunk)();
extern thunk _ctors_start; // first element
extern thunk _ctors_end;   // past-the-last element
extern thunk _dtors_start; // first element
extern thunk _dtors_end;   // past-the-last element

// the compiler is allowed to consider _start and _end as distinct, but they
// may be overlayed in reality; access one of them through a chess_copy to
// prevent optimization of the initial _start/_end non-equality test

static inline void _init() {
  // constructors are called in reverse order of the list
  for (thunk *t = &_ctors_end; t-- != chess_copy(&_ctors_start);)
    (*t)();
}

void _fini() {
  // destructors in forward order
  for (thunk *t = &_dtors_start; t != chess_copy(&_dtors_end); ++t)
    (*t)();
}

int main(int argc, char *argv[]);

int _main_init(int argc, char **argv) property(envelope);

// clang-format off
inline assembly void chess_envelope_open() {
  asm_begin
  .label __AIE_ARCH_MODEL_VERSION__20010000 global
    MOVXM sp, #_sp_start_value_DM_stack // init SP
  asm_end
}
// clang-format on

int _main_init(int argc, char **argv) property(envelope) {
  _init();

  // Statically initialized in atexit.c in libc
  // atexit(_fini);

  exit(main(argc, argv)); // run program and stop simulation (never returns)
  return 0;
}
}
