// RUN: aie-opt --aie-assign-lock-ids %s  -split-input-file -verify-diagnostics

aie.device(xcve2802) {
  // expected-note @below {{because only 16 locks available in this tile}}
  %tMemTile = aie.tile(4,4)
  %l0 = aie.lock(%tMemTile)
  %l1 = aie.lock(%tMemTile)
  %l2 = aie.lock(%tMemTile)
  %l3 = aie.lock(%tMemTile)
  %l4 = aie.lock(%tMemTile)
  %l5 = aie.lock(%tMemTile)
  %l6 = aie.lock(%tMemTile)
  %l7 = aie.lock(%tMemTile)
  %l8 = aie.lock(%tMemTile)
  %l9 = aie.lock(%tMemTile)
  %l10 = aie.lock(%tMemTile)
  %l11 = aie.lock(%tMemTile)
  %l12 = aie.lock(%tMemTile)
  %l13 = aie.lock(%tMemTile)
  %l14 = aie.lock(%tMemTile)
  %l15 = aie.lock(%tMemTile)
  // expected-error @below {{not allocated a lock}}
  %l16 = aie.lock(%tMemTile)
  %l17 = aie.lock(%tMemTile)
  %l18 = aie.lock(%tMemTile)
  %l19 = aie.lock(%tMemTile)
}

// -----

aie.device(xcve2802) {
  // expected-note @below {{because only 16 locks available in this tile}}
  %tMemTile = aie.tile(4,4)
  %l0 = aie.lock(%tMemTile)
  %l1 = aie.lock(%tMemTile, 1)
  %l2 = aie.lock(%tMemTile)
  %l3 = aie.lock(%tMemTile)
  %l4 = aie.lock(%tMemTile)
  %l5 = aie.lock(%tMemTile)
  %l6 = aie.lock(%tMemTile)
  %l7 = aie.lock(%tMemTile, 2)
  %l8 = aie.lock(%tMemTile)
  %l9 = aie.lock(%tMemTile)
  %l10 = aie.lock(%tMemTile, 15)
  %l11 = aie.lock(%tMemTile)
  %l12 = aie.lock(%tMemTile)
  %l13 = aie.lock(%tMemTile)
  %l14 = aie.lock(%tMemTile)
  // expected-error @below {{not allocated a lock}}
  %l15 = aie.lock(%tMemTile)
  %l16 = aie.lock(%tMemTile)
  %l17 = aie.lock(%tMemTile, 3)
  %l18 = aie.lock(%tMemTile)
  %l19 = aie.lock(%tMemTile)
}

// -----

aie.device(xcve2802) {
 //expected-note @below {{tile has lock ops assigned to same lock}}
  %t22 = aie.tile(2,2)
  %l0 = aie.lock(%t22, 7)
  // expected-error @below {{is assigned to the same lock (7) as another op}}
  %l1 = aie.lock(%t22, 7)
}
