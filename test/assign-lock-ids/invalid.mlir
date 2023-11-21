// RUN: aie-opt --aie-assign-lock-ids %s  -split-input-file -verify-diagnostics

AIE.device(xcve2802) {
  //expected-error @+1 {{op can have a maximum of 16 locks. No more available IDs.}}
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
  %l16 = AIE.lock(%tMemTile)
  %l17 = AIE.lock(%tMemTile)
  %l18 = AIE.lock(%tMemTile)
  %l19 = AIE.lock(%tMemTile)
}

//  -----

AIE.device(xcve2802) {
  //expected-error @+1 {{has multiple locks with ID 12.}}
  %t44 = AIE.tile(4,4)
  %l0 = AIE.lock(%t44, 12)
  %l1 = AIE.lock(%t44)
  %l2 = AIE.lock(%t44, 3)
  %l3 = AIE.lock(%t44, 12)
  %l4 = AIE.lock(%t44)
}

