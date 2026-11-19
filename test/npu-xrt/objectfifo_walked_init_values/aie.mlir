//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// @y's 30 objects and @z's link need 50 BD ids of the mem tile's 48, so @y is
// walked whole: one buffer that must start with every object's initial
// value. The host reads all 30 objects back.
module {
  aie.device(NPUDEVICE) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.objectfifo @y(%mem, {%shim}, 30 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : memref<16xi32>, dense<[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]> : memref<16xi32>, dense<1002> : memref<16xi32>, dense<[49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]> : memref<16xi32>, dense<[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]> : memref<16xi32>, dense<1005> : memref<16xi32>, dense<[97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]> : memref<16xi32>, dense<[113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]> : memref<16xi32>, dense<1008> : memref<16xi32>, dense<[145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160]> : memref<16xi32>, dense<[161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176]> : memref<16xi32>, dense<1011> : memref<16xi32>, dense<[193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208]> : memref<16xi32>, dense<[209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224]> : memref<16xi32>, dense<1014> : memref<16xi32>, dense<[241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]> : memref<16xi32>, dense<[257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272]> : memref<16xi32>, dense<1017> : memref<16xi32>, dense<[289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304]> : memref<16xi32>, dense<[305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320]> : memref<16xi32>, dense<1020> : memref<16xi32>, dense<[337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352]> : memref<16xi32>, dense<[353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368]> : memref<16xi32>, dense<1023> : memref<16xi32>, dense<[385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400]> : memref<16xi32>, dense<[401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416]> : memref<16xi32>, dense<1026> : memref<16xi32>, dense<[433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448]> : memref<16xi32>, dense<[449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464]> : memref<16xi32>, dense<1029> : memref<16xi32>]
    aie.objectfifo @z(%shim, {%mem}, 10 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @z_out(%mem, {%shim}, 10 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@z] -> [@z_out] ([] [])
    aie.runtime_sequence(%out: memref<480xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c480 = arith.constant 480 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c480][%c0, %c0, %c0, %c1]) {id = 0 : i64, metadata = @y, issue_token = true} : memref<480xi32>
      aiex.npu.dma_wait {symbol = @y}
    }
  }
}
