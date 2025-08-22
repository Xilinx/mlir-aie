// RUN: aie-opt %s --convert-vector-to-aievec -split-input-file | FileCheck %s

aie.device(xcvc1902) {
   // CHECK-LABEL: func @unaligned_read
   // CHECK: %[[C0:.*]] = arith.constant 0 : index
   // CHECK: %[[V0B:.*]] = aievec.upd %{{.*}}[%[[C0]]] {index = 0 : i8, offset = 0 : i32} : memref<64xi32>, vector<16xi32>
   // CHECK: %[[V0T:.*]] = aievec.upd %{{.*}}[%[[C0]]], %[[V0B]] {index = 1 : i8, offset = 256 : i32} : memref<64xi32>, vector<16xi32>
   // CHECK: %[[V0ROT:.*]] = aievec_aie1.select %[[V0T]] {select = "0", xoffsets = "0x76543210", xsquare = "0x3210", xstart = "3",
   // CHECK-SAME:                                                  yoffsets = "0", ysquare = "0", ystart = "0"}
   // CHECK-SAME:                                   : vector<16xi32>, vector<16xi32>
   // CHECK: %[[V0:.*]] = aievec_aie1.ext %[[V0ROT]] {index = 0 : i8} : vector<16xi32>, vector<8xi32>
   // CHECK: %[[V1ROT:.*]] = aievec_aie1.select %[[V0T]] {select = "0", xoffsets = "0x76543210", xsquare = "0x3210", xstart = "6",
   // CHECK-SAME:                                                  yoffsets = "0", ysquare = "0", ystart = "0"}
   // CHECK-SAME:                                   : vector<16xi32>, vector<16xi32>
   // CHECK: %[[V1:.*]] = aievec_aie1.ext %[[V1ROT]] {index = 0 : i8} : vector<16xi32>, vector<8xi32>
   // CHECK: return %[[V0]], %[[V1]] : vector<8xi32>, vector<8xi32>
   func.func @unaligned_read(%m: memref<64xi32>) -> (vector<8xi32>, vector<8xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c3 = arith.constant 3 : index
      %c6 = arith.constant 6 : index
      %0 = vector.transfer_read %m[%c3], %c0_i32 : memref<64xi32>, vector<8xi32>
      %1 = vector.transfer_read %m[%c6], %c0_i32 : memref<64xi32>, vector<8xi32>
      return %0, %1 : vector<8xi32>, vector<8xi32>
   }
}

// -----

aie.device(xcvc1902) {
   // CHECK-LABEL: func @unaligned_read
   // CHECK: %[[C0:.*]] = arith.constant 0 : index
   // CHECK: %[[V0B:.*]] = aievec.upd %{{.*}}[%[[C0]]] {index = 0 : i8, offset = 0 : i32} : memref<64xi16>, vector<32xi16>
   // CHECK: %[[V0T:.*]] = aievec.upd %{{.*}}[%[[C0]]], %[[V0B]] {index = 1 : i8, offset = 256 : i32} : memref<64xi16>, vector<32xi16>
   // CHECK: %[[V0ROT:.*]] = aievec_aie1.select %[[V0T]] {select = "0x11111111", xoffsets = "0x06040200", xoffsets_hi = "0x0e0c0a08", xsquare = "0x2103", xstart = "4",
   // CHECK-SAME:                                                           yoffsets = "0x0503010f", yoffsets_hi = "0x0d0b0907", ysquare = "0x2103", ystart = "2"}
   // CHECK-SAME:                                   : vector<32xi16>, vector<32xi16>
   // CHECK: %[[V0:.*]] = aievec_aie1.ext %[[V0ROT]] {index = 0 : i8} : vector<32xi16>, vector<16xi16>
   // CHECK: %[[V1ROT:.*]] = aievec_aie1.select %[[V0T]] {select = "0", xoffsets = "0x06040200", xoffsets_hi = "0x0e0c0a08", xsquare = "0x3210", xstart = "6",
   // CHECK-SAME:                                                  yoffsets = "0", yoffsets_hi = "0", ysquare = "0", ystart = "0"}
   // CHECK-SAME:                                   : vector<32xi16>, vector<32xi16>
   // CHECK: %[[V1:.*]] = aievec_aie1.ext %[[V1ROT]] {index = 0 : i8} : vector<32xi16>, vector<16xi16>
   // CHECK: return %[[V0]], %[[V1]] : vector<16xi16>, vector<16xi16>
   func.func @unaligned_read(%m: memref<64xi16>) -> (vector<16xi16>, vector<16xi16>) {
      %c0_i16 = arith.constant 0 : i16
      %c3 = arith.constant 3 : index
      %c6 = arith.constant 6 : index
      %0 = vector.transfer_read %m[%c3], %c0_i16 : memref<64xi16>, vector<16xi16>
      %1 = vector.transfer_read %m[%c6], %c0_i16 : memref<64xi16>, vector<16xi16>
      return %0, %1 : vector<16xi16>, vector<16xi16>
   }
}
