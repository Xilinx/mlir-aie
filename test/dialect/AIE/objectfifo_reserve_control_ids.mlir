//===- objectfifo_reserve_control_ids.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The ctrl-pkt overlay's controller ids on an occupied column are {15, 26,
// 27, 29, 30, 31} (npu1, aie.getTileToControllerIdMap(columnWiseUniqueIDs=
// true)). With 16 unpinned packet flows, plain auto-assignment counts up from
// 0 and reaches 15 -- the shim controller id. With `reserve-control-ids=true`,
// the allocator pre-reserves those ids on every occupied column so data ids
// skip over them instead.
//
// Two modules cover both occupancy shapes: the first spreads its 16 flows over
// all four columns, the second confines all 16 to column 0. The column-scoped
// reservation is a no-op in the column-wise-unique mode the overlay uses (every
// column carries the same id set), so both shapes must skip the reserved ids;
// the single-column module documents that the subset shape behaves identically.

// RUN: aie-opt %s -split-input-file --aie-objectfifo-allocate | FileCheck %s --check-prefix=OFF
// RUN: aie-opt %s -split-input-file --aie-objectfifo-allocate="reserve-control-ids=true" | FileCheck %s --check-prefix=ON

module @reserve_control_ids_all_columns {
  aie.device(npu1) {
    %t00 = aie.tile(0, 2)
    %t01 = aie.tile(0, 3)
    %t02 = aie.tile(0, 4)
    %t03 = aie.tile(0, 5)
    %t10 = aie.tile(1, 2)
    %t11 = aie.tile(1, 3)
    %t12 = aie.tile(1, 4)
    %t13 = aie.tile(1, 5)
    %t20 = aie.tile(2, 2)
    %t21 = aie.tile(2, 3)
    %t22 = aie.tile(2, 4)
    %t23 = aie.tile(2, 5)
    %t30 = aie.tile(3, 2)
    %t31 = aie.tile(3, 3)
    %t32 = aie.tile(3, 4)
    %t33 = aie.tile(3, 5)

    // One stream-port pair per tile, each pair its own unpinned packet flow.
    // 16 flows in total, so auto-assignment must reach id 15.
    aie.route_endpoint @src00(%t00) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst00(%t00) Core {channelIndex = 0 : i32}
    aie.route from @src00 to [@dst00] {packet}

    aie.route_endpoint @src01(%t01) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst01(%t01) Core {channelIndex = 0 : i32}
    aie.route from @src01 to [@dst01] {packet}

    aie.route_endpoint @src02(%t02) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst02(%t02) Core {channelIndex = 0 : i32}
    aie.route from @src02 to [@dst02] {packet}

    aie.route_endpoint @src03(%t03) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst03(%t03) Core {channelIndex = 0 : i32}
    aie.route from @src03 to [@dst03] {packet}

    aie.route_endpoint @src10(%t10) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst10(%t10) Core {channelIndex = 0 : i32}
    aie.route from @src10 to [@dst10] {packet}

    aie.route_endpoint @src11(%t11) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst11(%t11) Core {channelIndex = 0 : i32}
    aie.route from @src11 to [@dst11] {packet}

    aie.route_endpoint @src12(%t12) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst12(%t12) Core {channelIndex = 0 : i32}
    aie.route from @src12 to [@dst12] {packet}

    aie.route_endpoint @src13(%t13) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst13(%t13) Core {channelIndex = 0 : i32}
    aie.route from @src13 to [@dst13] {packet}

    aie.route_endpoint @src20(%t20) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst20(%t20) Core {channelIndex = 0 : i32}
    aie.route from @src20 to [@dst20] {packet}

    aie.route_endpoint @src21(%t21) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst21(%t21) Core {channelIndex = 0 : i32}
    aie.route from @src21 to [@dst21] {packet}

    aie.route_endpoint @src22(%t22) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst22(%t22) Core {channelIndex = 0 : i32}
    aie.route from @src22 to [@dst22] {packet}

    aie.route_endpoint @src23(%t23) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst23(%t23) Core {channelIndex = 0 : i32}
    aie.route from @src23 to [@dst23] {packet}

    aie.route_endpoint @src30(%t30) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst30(%t30) Core {channelIndex = 0 : i32}
    aie.route from @src30 to [@dst30] {packet}

    aie.route_endpoint @src31(%t31) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst31(%t31) Core {channelIndex = 0 : i32}
    aie.route from @src31 to [@dst31] {packet}

    aie.route_endpoint @src32(%t32) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst32(%t32) Core {channelIndex = 0 : i32}
    aie.route from @src32 to [@dst32] {packet}

    aie.route_endpoint @src33(%t33) Core {channelIndex = 0 : i32}
    aie.route_endpoint @dst33(%t33) Core {channelIndex = 0 : i32}
    aie.route from @src33 to [@dst33] {packet}
  }
}

// -----

// Subset-occupancy shape: every flow lives on column 0, so columns 1-3 are
// unoccupied. Four core tiles carry four stream-port pairs each, 16 flows in
// all, so auto-assignment still reaches id 15 and the ON run must still skip
// the reserved controller ids.
module @reserve_control_ids_single_column {
  aie.device(npu1) {
    %c02 = aie.tile(0, 2)
    %c03 = aie.tile(0, 3)
    %c04 = aie.tile(0, 4)
    %c05 = aie.tile(0, 5)

    aie.route_endpoint @s02_0(%c02) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d02_0(%c02) Core {channelIndex = 0 : i32}
    aie.route from @s02_0 to [@d02_0] {packet}
    aie.route_endpoint @s02_1(%c02) Core {channelIndex = 1 : i32}
    aie.route_endpoint @d02_1(%c02) Core {channelIndex = 1 : i32}
    aie.route from @s02_1 to [@d02_1] {packet}
    aie.route_endpoint @s02_2(%c02) Core {channelIndex = 2 : i32}
    aie.route_endpoint @d02_2(%c02) Core {channelIndex = 2 : i32}
    aie.route from @s02_2 to [@d02_2] {packet}
    aie.route_endpoint @s02_3(%c02) Core {channelIndex = 3 : i32}
    aie.route_endpoint @d02_3(%c02) Core {channelIndex = 3 : i32}
    aie.route from @s02_3 to [@d02_3] {packet}

    aie.route_endpoint @s03_0(%c03) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d03_0(%c03) Core {channelIndex = 0 : i32}
    aie.route from @s03_0 to [@d03_0] {packet}
    aie.route_endpoint @s03_1(%c03) Core {channelIndex = 1 : i32}
    aie.route_endpoint @d03_1(%c03) Core {channelIndex = 1 : i32}
    aie.route from @s03_1 to [@d03_1] {packet}
    aie.route_endpoint @s03_2(%c03) Core {channelIndex = 2 : i32}
    aie.route_endpoint @d03_2(%c03) Core {channelIndex = 2 : i32}
    aie.route from @s03_2 to [@d03_2] {packet}
    aie.route_endpoint @s03_3(%c03) Core {channelIndex = 3 : i32}
    aie.route_endpoint @d03_3(%c03) Core {channelIndex = 3 : i32}
    aie.route from @s03_3 to [@d03_3] {packet}

    aie.route_endpoint @s04_0(%c04) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d04_0(%c04) Core {channelIndex = 0 : i32}
    aie.route from @s04_0 to [@d04_0] {packet}
    aie.route_endpoint @s04_1(%c04) Core {channelIndex = 1 : i32}
    aie.route_endpoint @d04_1(%c04) Core {channelIndex = 1 : i32}
    aie.route from @s04_1 to [@d04_1] {packet}
    aie.route_endpoint @s04_2(%c04) Core {channelIndex = 2 : i32}
    aie.route_endpoint @d04_2(%c04) Core {channelIndex = 2 : i32}
    aie.route from @s04_2 to [@d04_2] {packet}
    aie.route_endpoint @s04_3(%c04) Core {channelIndex = 3 : i32}
    aie.route_endpoint @d04_3(%c04) Core {channelIndex = 3 : i32}
    aie.route from @s04_3 to [@d04_3] {packet}

    aie.route_endpoint @s05_0(%c05) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d05_0(%c05) Core {channelIndex = 0 : i32}
    aie.route from @s05_0 to [@d05_0] {packet}
    aie.route_endpoint @s05_1(%c05) Core {channelIndex = 1 : i32}
    aie.route_endpoint @d05_1(%c05) Core {channelIndex = 1 : i32}
    aie.route from @s05_1 to [@d05_1] {packet}
    aie.route_endpoint @s05_2(%c05) Core {channelIndex = 2 : i32}
    aie.route_endpoint @d05_2(%c05) Core {channelIndex = 2 : i32}
    aie.route from @s05_2 to [@d05_2] {packet}
    aie.route_endpoint @s05_3(%c05) Core {channelIndex = 3 : i32}
    aie.route_endpoint @d05_3(%c05) Core {channelIndex = 3 : i32}
    aie.route from @s05_3 to [@d05_3] {packet}
  }
}

// Both modules reach id 15 when the reservation is off.
// OFF: aie.packet_flow(15)
// OFF-LABEL: @reserve_control_ids_single_column
// OFF: aie.packet_flow(15)

// Neither module -- all-columns nor single-column -- emits a reserved
// controller id when the reservation is on, and the 16th flow still gets
// allocated: it lands on the next free id (16) instead of being silently
// dropped when 15 is skipped.
// ON-NOT: aie.packet_flow(15)
// ON-NOT: aie.packet_flow(26)
// ON-NOT: aie.packet_flow(27)
// ON-NOT: aie.packet_flow(29)
// ON-NOT: aie.packet_flow(30)
// ON-NOT: aie.packet_flow(31)
// ON: aie.packet_flow(16)
// ON-LABEL: @reserve_control_ids_single_column
// ON-NOT: aie.packet_flow(15)
// ON-NOT: aie.packet_flow(26)
// ON-NOT: aie.packet_flow(27)
// ON-NOT: aie.packet_flow(29)
// ON-NOT: aie.packet_flow(30)
// ON-NOT: aie.packet_flow(31)
// ON: aie.packet_flow(16)
