// RUN: aie-opt --aie-assign-lock-ids %s  -split-input-file -verify-diagnostics

AIE.device(xcve2802) {
  //expected-note @below {{has only 16 locks}}
  %tMemTile = AIE.tile(4,4)
  %l0 = AIE.lock(%tMemTile)
  %l1 = AIE.lock(%tMemTile)
  %l2 = AIE.lock(%tMemTile)
  %l3 = AIE.lock(%tMemTile)
  %l4 = AIE.lock(%tMemTile)
  %l5 = AIE.lock(%tMemTile)
  %l6 = AIE.lock(%tMemTile)
  %l7 = AIE.lock(%tMemTile)
  %l8 = AIE.lock(%tMemTile)
  %l9 = AIE.lock(%tMemTile)
  %l10 = AIE.lock(%tMemTile)
  %l11 = AIE.lock(%tMemTile)
  %l12 = AIE.lock(%tMemTile)
  %l13 = AIE.lock(%tMemTile)
  %l14 = AIE.lock(%tMemTile)
  %l15 = AIE.lock(%tMemTile)
  //expected-error @below {{not allocated a lock}}
  %l16 = AIE.lock(%tMemTile)
  %l17 = AIE.lock(%tMemTile)
  %l18 = AIE.lock(%tMemTile)
  %l19 = AIE.lock(%tMemTile)
}

// -----

AIE.device(xcve2802) {
 //expected-note @below {{tile has lock ops assigned to same lock}}
  %t22 = AIE.tile(2,2)
  %l0 = AIE.lock(%t22, 7)
  // expected-error @below {{is assigned to the same lock (7) as another op}}
  %l1 = AIE.lock(%t22, 7)
}
