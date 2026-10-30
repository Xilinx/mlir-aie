# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

# readable() turns each op's location into a short trailing comment: the
# names on it (iron: dropped, py: kept) and its innermost source line.

from aie.utils.readable_ir import readable

IR = """\
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2) loc(#loc)
    aie.objectfifo @act_in(%tile_0_2, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> loc(#loc3)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end loc(#loc5)
    } loc(#loc6)
    %core_0_2 = aie.core(%tile_0_2) {
      ^bb0(%arg0: i32 loc("blk.py":3:0)):
      aie.end loc(#loc4)
    } {link_with = "has loc(inside).o"} loc(unknown)
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4xi32> loc("/abs/path/inline.py":9:1)
    func.call @get_loc(%b) : (memref<4xi32>) -> () loc("d\\C3\\A9sign.py":5:0)
  } loc(#loc)
} loc(#loc)
#loc = loc("design.py":1:0)
#loc1 = loc("/home/me/helpers.py":12:4)
#loc2 = loc("/home/me/design.py":30:2)
#loc3 = loc("iron:act_in"(callsite(#loc1 at #loc2)))
#loc4 = loc("py:core_fn"("design.py":20:8))
#loc5 = loc("iron:of_out"("design.py":40:0))
#loc6 = loc(fused[#loc3, #loc5])
"""

print(readable(IR))

# CHECK:      %tile_0_2 = aie.tile(0, 2)  // design.py:1
# CHECK-NEXT: aie.objectfifo @act_in(%tile_0_2, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>  // act_in · helpers.py:12
# CHECK-NEXT: %mem_0_2 = aie.mem(%tile_0_2) {
# CHECK-NEXT:   aie.end  // of_out · design.py:40
# Fused: every name, the first source line.
# CHECK-NEXT: }  // act_in, of_out · helpers.py:12

# CHECK-NEXT: %core_0_2 = aie.core(%tile_0_2) {
# A block argument's location is dropped from the middle of the line.
# CHECK-NEXT:   ^bb0(%arg0: i32):  // blk.py:3
# CHECK-NEXT:   aie.end  // py:core_fn · design.py:20
# "loc(" inside a string is left alone; unknown has nothing to show.
# CHECK-NEXT: } {link_with = "has loc(inside).o"}{{$}}
# CHECK-NEXT: %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4xi32>  // inline.py:9
# Only a whole `loc(` starts a location; MLIR's escapes are undone.
# CHECK-NEXT: func.call @get_loc(%b) : (memref<4xi32>) -> ()  // désign.py:5
# Alias definitions are gone.
# CHECK-NOT: #loc
