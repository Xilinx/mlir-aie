//===- AIE2_cyclostatic_l2.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// In this example, an AIE core pushes data into a memtile, in a one-by-one
// fashion. The memtile forwards this one-by-one to a consumer tile. The 
// consumer tile cyclostatically consumes {1, 2, 1} elements at a time.

// The way this gets lowered is as follows:
//
// - On the producer tile, two buffers get allocated. Each time the producer
//   wishes to push onto the objectFifo, the implementation alternates between
//   the two buffers (ping-pong). This way, the previous buffer remains 
//   untouched while it is being pushed onto the stream. The other one can
//   meanwhile be filled with the next object.
// 
// - On the memory tile, objects are read in from the stream one-by-one. Since
//   the objectFifo is allocated to hold _up to_ 4 elements, four buffers are
//   provisioned on the memory tile, into which data from the stream is
//   received. The "_cons" locks are used to notify the memory tile whenever
//   a single new object is ready on the stream. As the objects get pushed
//   from memory back out onto the stream, backpressure makes sure that no more
//   elements are written to the stream than are read on the receiving end.
//   Therefore, this boils down to forwarding objects one-by-one through the
//   memory tile (irrespective of what chunk size the consumer consumes).
//
// - On the receiving consumer end, four buffers are also preallocated, into 
//   which the DMA copies objects arriving from the stream. This again is done 
//   object-by-object. If the consumer needs more than one object at once, it 
//   acquires the consumer locks multiple times.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @aie2_cyclostatic_l2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[t0:.*]] = aie.tile(2, 2)
// CHECK:     %[[t1:.*]] = aie.tile(2, 1)
// CHECK:     %[[t2:.*]] = aie.tile(8, 3)

// CHECK:     %[[fifo1_cons_buff_0:.*]] = aie.buffer(%[[t2]]) {sym_name = "fifo1_cons_buff_0"} : memref<1xi32>
// CHECK:     %[[fifo1_cons_buff_1:.*]] = aie.buffer(%[[t2]]) {sym_name = "fifo1_cons_buff_1"} : memref<1xi32>
// CHECK:     %[[fifo1_cons_buff_2:.*]] = aie.buffer(%[[t2]]) {sym_name = "fifo1_cons_buff_2"} : memref<1xi32>
// CHECK:     %[[fifo1_cons_buff_3:.*]] = aie.buffer(%[[t2]]) {sym_name = "fifo1_cons_buff_3"} : memref<1xi32>
// CHECK:     %[[fifo1_cons_prod_lock:.*]] = aie.lock(%[[t2]], 0) {init = 4 : i32, sym_name = "fifo1_cons_prod_lock"}
// CHECK:     %[[fifo1_cons_cons_lock:.*]] = aie.lock(%[[t2]], 1) {init = 0 : i32, sym_name = "fifo1_cons_cons_lock"}

// The consume buffers are used at the receiving end of a stream to notify the
// sender to send more objects once they have been consumed. In this case,
// the (intermediary) consumer is the memtile.
// CHECK:     %[[fifo0_cons_buff_0:.*]] = aie.buffer(%[[t1]]) {sym_name = "fifo0_cons_buff_0"} : memref<1xi32>
// CHECK:     %[[fifo0_cons_buff_1:.*]] = aie.buffer(%[[t1]]) {sym_name = "fifo0_cons_buff_1"} : memref<1xi32>
// CHECK:     %[[fifo0_cons_buff_2:.*]] = aie.buffer(%[[t1]]) {sym_name = "fifo0_cons_buff_2"} : memref<1xi32>
// CHECK:     %[[fifo0_cons_buff_3:.*]] = aie.buffer(%[[t1]]) {sym_name = "fifo0_cons_buff_3"} : memref<1xi32>

// CHECK:     %[[fifo0_cons_prod_lock:.*]] = aie.lock(%[[t1]], 0) {init = 4 : i32, sym_name = "fifo0_cons_prod_lock"}
// CHECK:     %[[fifo0_cons_cons_lock:.*]] = aie.lock(%[[t1]], 1) {init = 0 : i32, sym_name = "fifo0_cons_cons_lock"}

// The objectFifo lowering creates two buffers (for ping-pong) on the producer
// side to which elements are written.
// CHECK:     %[[fifo0_buff_0:.*]] = aie.buffer(%[[t0]]) {sym_name = "fifo0_buff_0"} : memref<1xi32>
// CHECK:     %[[fifo0_buff_1:.*]] = aie.buffer(%[[t0]]) {sym_name = "fifo0_buff_1"} : memref<1xi32>

// Whenever the prod lock can be acquired, the core can proceed to put another
// object into the fifo, i.e. there is space in the queue.
// CHECK:     %[[fifo0_prod_lock:.*]] = aie.lock(%[[t0]], 0) {init = 2 : i32, sym_name = "fifo0_prod_lock"}

// Whenever the cons lock can be acquired, there is an object available in the
// queue to be consumed.
// CHECK:     %[[fifo0_cons_lock:.*]] = aie.lock(%[[t0]], 1) {init = 0 : i32, sym_name = "fifo0_cons_lock"}

// CHECK:     %[[buf83:.*]] = aie.buffer(%[[t2]]) {sym_name = "buf83"} : memref<1xi32>

// We expect a flow out of t0's core into the memtile:
// CHECK:     aie.flow(%[[t0]], DMA : 0, %[[t1]], DMA : 0)

// Flow out of the memtile into t2's DMA. This is mostly analogous to the
// flow from t0 to the memtile.
// CHECK:     aie.flow(%[[t1]], DMA : 0, %[[t2]], DMA : 0)


// ////////////////////////////////////////////////////////////////////////// //
// Producer core:
// ////////////////////////////////////////////////////////////////////////// //

// CHECK:     %[[c0:.*]] = aie.core(%[[t0]]) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       aie.use_lock(%[[fifo0_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       memref.store %c55_i32, %[[fifo0_buff_0]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo0_cons_lock]], Release, 1)
// CHECK:       aie.use_lock(%[[fifo0_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       memref.store %c66_i32, %[[fifo0_buff_1]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo0_cons_lock]], Release, 1)
// CHECK:       aie.use_lock(%[[fifo0_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       memref.store %c77_i32, %[[fifo0_buff_0]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo0_cons_lock]], Release, 1)
// CHECK:       aie.use_lock(%[[fifo0_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       memref.store %c88_i32, %[[fifo0_buff_1]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo0_cons_lock]], Release, 1)
// CHECK:       aie.end
// CHECK:     }


// ////////////////////////////////////////////////////////////////////////// //
// Consumer core:
// ////////////////////////////////////////////////////////////////////////// //

// CHECK:     %[[c2:.*]] = aie.core(%[[t2]]) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c2 = arith.constant 2 : index
// CHECK:       %c3 = arith.constant 3 : index

// The fifo1_cons_cons_lock will be released with a value of 1 whenever the
// DMA received an object from the stream and wrote it to the buffer. First,
// we only want to consume one object, so it suffices to acquire this lock
// with a value of 1:
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       %[[load0:.*]] = memref.load %[[fifo1_cons_buff_0]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], Release, 1)

// We released the lock above, meaning we are done with the one object we
// received. Now we want 2 _new_ objects, so the cons_cons lock is acquired 
// twice, meaning it has to be released twice before both acquires succeed;
// this, again, meaning that the DMA has received two objects on the stream
// and put them in the respective buffers.
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], AcquireGreaterEqual, 2)
// CHECK:       %[[load1:.*]] = memref.load %[[fifo1_cons_buff_1]][%c0] : memref<1xi32>
// CHECK:       %[[load2:.*]] = memref.load %[[fifo1_cons_buff_2]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], Release, 2)

// Lastly, receive just one object:
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       %[[load3:.*]] = memref.load %[[fifo1_cons_buff_3]][%c0] : memref<1xi32>
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], Release, 1)
// CHECK:       aie.end
// CHECK:     }


// ////////////////////////////////////////////////////////////////////////// //
// Producer tile's DMA:
// ////////////////////////////////////////////////////////////////////////// //

// CHECK:     %[[mem0:.*]] = aie.mem(%[[t0]]) {
// CHECK:       %[[dma0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)

// Memory to stream: As soon as we get an object in fifo0_buff_0, put it onto
// the stream, then move on to bb2.
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[fifo0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_buff_0]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb2

// Now, if we get 4 bytes in fifo0_buff_1, put that on the stream, then
// go back to bb1. Ping-pong.
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[fifo0_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_buff_1]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb1

// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }


// ////////////////////////////////////////////////////////////////////////// //
// Mem tile:
// ////////////////////////////////////////////////////////////////////////// //

// CHECK:     %[[memtile:.*]] = aie.memtile_dma(%[[t1]]) {
// CHECK:       %[[VAL_25:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)

// Fill our four buffers, fifo0_cons_buff_0 through fif0_cons_buff_3, 
// allocated inside the memory tile, one by one (round robin) as we receive
// things through the stream:
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_0]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_1]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_2]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_3]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb1

// Now map everything we read in back out onto the stream towards tile 2:
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       %[[VAL_26:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb10)
// CHECK:     ^bb6:  // 2 preds: ^bb5, ^bb9
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_0]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_1]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_2]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb9:  // pred: ^bb8
// CHECK:       aie.use_lock(%[[fifo0_cons_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo0_cons_buff_3]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo0_cons_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb10:  // pred: ^bb5
// CHECK:       aie.end
// CHECK:     }


// ////////////////////////////////////////////////////////////////////////// //
// Consumer tile's DMA:
// ////////////////////////////////////////////////////////////////////////// //

// Things are read from the stream into memory object-by-object, 
// irrespective of the number of objects that the consumer wants to consume
// at a time. This uses the separate _cons locks, which increase/decrease
// by one.

// CHECK:     %[[mem2:.*]] = aie.mem(%[[t2]]) {
// CHECK:       %[[dma2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo1_cons_buff_0]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo1_cons_buff_1]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo1_cons_buff_2]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[fifo1_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo1_cons_buff_3]] : memref<1xi32>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo1_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }


// ////////////////////////////////////////////////////////////////////////// //
// Test input:
// ////////////////////////////////////////////////////////////////////////// //

module @aie2_cyclostatic_l2 {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %memtile = aie.tile(2, 1) // mem tile
        %tile83 = aie.tile(8, 3)  // consumer tile
        %buf83  = aie.buffer(%tile83) {sym_name = "buf83"} : memref<1xi32>

        // ObjectFifo that can hold 4 memref<1xi32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo0 (%tile22, {%memtile}, 4 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo @fifo1 (%memtile, {%tile83}, [4, 4]) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo.link [@fifo0] -> [@fifo1] ()

        // Producer core
        %core22 = aie.core(%tile22) {
            %i0 = arith.constant 0 : index
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            
            // Push 55
            %subview0 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c55, %subview0_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)

            // Push 66
            %subview1 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c66, %subview1_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)

            // Push 77
            %subview2 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c77, %subview2_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)

            // Push 88
            %subview3 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview3_obj = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %c88, %subview3_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0 (Produce, 1)

            aie.end
        }

        // Consumer core
        %core28 = aie.core(%tile83) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Pop 1 object off queue
            %subview0 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %v55 = memref.load %subview0_obj[%i0] : memref<1xi32>
            memref.store %v55, %buf83[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo1 (Consume, 1)

            // Pop 2 objects off queue
            %subview1 = aie.objectfifo.acquire @fifo1 (Consume, 2) : !aie.objectfifosubview<memref<1xi32>>
            %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %v66 = memref.load %subview1_obj0[%i0] : memref<1xi32>
            %v77 = memref.load %subview1_obj1[%i0] : memref<1xi32>
            memref.store %v66, %buf83[%i1] : memref<1xi32>
            memref.store %v77, %buf83[%i2] : memref<1xi32>
            aie.objectfifo.release @fifo1 (Consume, 2)

            // Pop 1 object off queue
            %subview2 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            %v88 = memref.load %subview2_obj[%i0] : memref<1xi32>
            memref.store %v88, %buf83[%i3] : memref<1xi32>
            aie.objectfifo.release @fifo1 (Consume, 1)

            aie.end
        }

    }
}
