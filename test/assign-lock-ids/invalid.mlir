// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-lock-ids %s  -split-input-file -verify-diagnostics

aie.device(xcve2802) {
  // expected-note @below {{because only 16 locks available in this tile}}
  %tMemTile = aie.tile(4,4)
  aie.lock @l0(%tMemTile)
  aie.lock @l1(%tMemTile)
  aie.lock @l2(%tMemTile)
  aie.lock @l3(%tMemTile)
  aie.lock @l4(%tMemTile)
  aie.lock @l5(%tMemTile)
  aie.lock @l6(%tMemTile)
  aie.lock @l7(%tMemTile)
  aie.lock @l8(%tMemTile)
  aie.lock @l9(%tMemTile)
  aie.lock @l10(%tMemTile)
  aie.lock @l11(%tMemTile)
  aie.lock @l12(%tMemTile)
  aie.lock @l13(%tMemTile)
  aie.lock @l14(%tMemTile)
  aie.lock @l15(%tMemTile)
  // expected-error @below {{not allocated a lock}}
  aie.lock @l16(%tMemTile)
  aie.lock @l17(%tMemTile)
  aie.lock @l18(%tMemTile)
  aie.lock @l19(%tMemTile)
}

// -----

aie.device(xcve2802) {
  // expected-note @below {{because only 16 locks available in this tile}}
  %tMemTile = aie.tile(4,4)
  aie.lock @l0(%tMemTile)
  aie.lock @l1(%tMemTile, 1)
  aie.lock @l2(%tMemTile)
  aie.lock @l3(%tMemTile)
  aie.lock @l4(%tMemTile)
  aie.lock @l5(%tMemTile)
  aie.lock @l6(%tMemTile)
  aie.lock @l7(%tMemTile, 2)
  aie.lock @l8(%tMemTile)
  aie.lock @l9(%tMemTile)
  aie.lock @l10(%tMemTile, 15)
  aie.lock @l11(%tMemTile)
  aie.lock @l12(%tMemTile)
  aie.lock @l13(%tMemTile)
  aie.lock @l14(%tMemTile)
  // expected-error @below {{not allocated a lock}}
  aie.lock @l15(%tMemTile)
  aie.lock @l16(%tMemTile)
  aie.lock @l17(%tMemTile, 3)
  aie.lock @l18(%tMemTile)
  aie.lock @l19(%tMemTile)
}

// -----

aie.device(xcve2802) {
 //expected-note @below {{tile has lock ops assigned to same lock}}
  %t22 = aie.tile(2,2)
  aie.lock @l0(%t22, 7)
  // expected-error @below {{is assigned to the same lock (7) as another op}}
  aie.lock @l1(%t22, 7)
}
