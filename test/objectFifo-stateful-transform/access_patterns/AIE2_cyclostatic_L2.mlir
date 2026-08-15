//===- AIE2_cyclostatic_L2.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s



// The consume buffers are used at the receiving end of a stream to notify the
// sender to send more objects once they have been consumed. In this case,
// the (intermediary) consumer is the memtile.


// The objectFifo lowering creates two buffers (for ping-pong) on the producer
// side to which elements are written.

// Whenever the prod lock can be acquired, the core can proceed to put another
// object into the fifo, i.e. there is space in the queue.

// Whenever the cons lock can be acquired, there is an object available in the
// queue to be consumed.


// We expect a flow out of t0's core into the memtile:

// Flow out of the memtile into t2's DMA. This is mostly analogous to the
// flow from t0 to the memtile.


// ////////////////////////////////////////////////////////////////////////// //
// Producer core:
// ////////////////////////////////////////////////////////////////////////// //



// ////////////////////////////////////////////////////////////////////////// //
// Consumer core:
// ////////////////////////////////////////////////////////////////////////// //


// The fifo1_cons_cons_lock will be released with a value of 1 whenever the
// DMA received an object from the stream and wrote it to the buffer. First,
// we only want to consume one object, so it suffices to acquire this lock
// with a value of 1:

// We released the lock above, meaning we are done with the one object we
// received. Now we want 2 _new_ objects, so the cons_cons lock is acquired
// twice, meaning it has to be released twice before both acquires succeed;
// this, again, meaning that the DMA has received two objects on the stream
// and put them in the respective buffers.

// Lastly, receive just one object:


// ////////////////////////////////////////////////////////////////////////// //
// Producer tile's DMA:
// ////////////////////////////////////////////////////////////////////////// //


// Memory to stream: As soon as we get an object in fifo0_buff_0, put it onto
// the stream, then move on to bb2.

// Now, if we get 4 bytes in fifo0_buff_1, put that on the stream, then
// go back to bb1. Ping-pong.



// ////////////////////////////////////////////////////////////////////////// //
// Mem tile:
// ////////////////////////////////////////////////////////////////////////// //


// Fill our four buffers, fifo0_cons_buff_0 through fif0_cons_buff_3,
// allocated inside the memory tile, one by one (round robin) as we receive
// things through the stream:

// Now map everything we read in back out onto the stream towards tile 2:


// ////////////////////////////////////////////////////////////////////////// //
// Consumer tile's DMA:
// ////////////////////////////////////////////////////////////////////////// //

// Things are read from the stream into memory object-by-object,
// irrespective of the number of objects that the consumer wants to consume
// at a time. This uses the separate _cons locks, which increase/decrease
// by one.



// ////////////////////////////////////////////////////////////////////////// //
// Test input:
// ////////////////////////////////////////////////////////////////////////// //

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_0"} : memref<1xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_1"} : memref<1xi32>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_0]]) {init = 2 : i32, sym_name = "fifo0_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo0_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo0_cons_buff_0"} : memref<1xi32>
// CHECK:           %[[VAL_7:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo0_cons_buff_1"} : memref<1xi32>
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo0_cons_buff_2"} : memref<1xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_5]]) {sym_name = "fifo0_cons_buff_3"} : memref<1xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_5]]) {init = 4 : i32, sym_name = "fifo0_cons_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_5]]) {init = 0 : i32, sym_name = "fifo0_cons_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.tile(8, 3)
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "fifo1_cons_buff_0"} : memref<1xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "fifo1_cons_buff_1"} : memref<1xi32>
// CHECK:           %[[VAL_15:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "fifo1_cons_buff_2"} : memref<1xi32>
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "fifo1_cons_buff_3"} : memref<1xi32>
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_12]]) {init = 4 : i32, sym_name = "fifo1_cons_prod_lock_0"}
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_12]]) {init = 0 : i32, sym_name = "fifo1_cons_cons_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.buffer(%[[VAL_12]]) {sym_name = "buf83"} : memref<1xi32>
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_5]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_5]], DMA : 0, %[[VAL_12]], DMA : 0)
// CHECK:           %[[VAL_20:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_21:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_22:.*]] = arith.constant 88 : i32
// CHECK:             %[[VAL_23:.*]] = arith.constant 77 : i32
// CHECK:             %[[VAL_24:.*]] = arith.constant 66 : i32
// CHECK:             %[[VAL_25:.*]] = arith.constant 55 : i32
// CHECK:             %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             memref.store %[[VAL_25]], %[[VAL_1]]{{\[}}%[[VAL_26]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_21]])
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             memref.store %[[VAL_24]], %[[VAL_2]]{{\[}}%[[VAL_26]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_21]])
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             memref.store %[[VAL_23]], %[[VAL_1]]{{\[}}%[[VAL_26]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_21]])
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_21]])
// CHECK:             memref.store %[[VAL_22]], %[[VAL_2]]{{\[}}%[[VAL_26]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_21]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = aie.core(%[[VAL_12]]) {
// CHECK:             %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_29:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_30:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_31:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_32:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_33:.*]] = arith.constant 2 : i32
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_32]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_34]], %[[VAL_19]]{{\[}}%[[VAL_32]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_28]])
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_33]])
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_32]]] : memref<1xi32>
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_32]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_35]], %[[VAL_19]]{{\[}}%[[VAL_31]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_36]], %[[VAL_19]]{{\[}}%[[VAL_30]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_33]])
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, %[[VAL_28]])
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_32]]] : memref<1xi32>
// CHECK:             memref.store %[[VAL_37]], %[[VAL_19]]{{\[}}%[[VAL_29]]] : memref<1xi32>
// CHECK:             aie.use_lock(%[[VAL_17]], Release, %[[VAL_28]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_38:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_39:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_40:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_1]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, %[[VAL_39]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_3]], Release, %[[VAL_39]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = aie.memtile_dma(%[[VAL_5]]) {
// CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_43:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_44:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb10)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_7]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, %[[VAL_42]])
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, %[[VAL_42]])
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb10:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = aie.mem(%[[VAL_12]]) {
// CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_47:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_46]])
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_46]])
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_46]])
// CHECK:             aie.dma_bd(%[[VAL_15]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_17]], AcquireGreaterEqual, %[[VAL_46]])
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<1xi32> offset = 0 len = 1)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, %[[VAL_46]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @aie2_cyclostatic_L2 {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %memtile = aie.tile(2, 1) // mem tile
        %tile83 = aie.tile(8, 3)  // consumer tile
        %buf83  = aie.buffer(%tile83) {sym_name = "buf83"} : memref<1xi32>

        // ObjectFifo that can hold 4 memref<1xi32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo0 (%tile22, {%memtile}, 4 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo @fifo1 (%memtile, {%tile83}, [4, 4]) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo.link [@fifo0] -> [@fifo1] ([] [])

        // Producer core
        %core22 = aie.core(%tile22) {
            %i0 = arith.constant 0 : index
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32

            // Push 55
            %subview0_obj = aie.objectfifo.acquire @fifo0(Produce) : memref<1xi32>
            memref.store %c55, %subview0_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0(Produce) [1]

            // Push 66
            %subview1_obj = aie.objectfifo.acquire @fifo0(Produce) : memref<1xi32>
            memref.store %c66, %subview1_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0(Produce) [1]

            // Push 77
            %subview2_obj = aie.objectfifo.acquire @fifo0(Produce) : memref<1xi32>
            memref.store %c77, %subview2_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0(Produce) [1]

            // Push 88
            %subview3_obj = aie.objectfifo.acquire @fifo0(Produce) : memref<1xi32>
            memref.store %c88, %subview3_obj[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo0(Produce) [1]

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
            %subview0_obj = aie.objectfifo.acquire @fifo1(Consume) : memref<1xi32>
            %v55 = memref.load %subview0_obj[%i0] : memref<1xi32>
            memref.store %v55, %buf83[%i0] : memref<1xi32>
            aie.objectfifo.release @fifo1(Consume) [1]

            // Pop 2 objects off queue
            %subview1_obj0, %subview1_obj1 = aie.objectfifo.acquire @fifo1(Consume) : memref<1xi32>, memref<1xi32>
            %v66 = memref.load %subview1_obj0[%i0] : memref<1xi32>
            %v77 = memref.load %subview1_obj1[%i0] : memref<1xi32>
            memref.store %v66, %buf83[%i1] : memref<1xi32>
            memref.store %v77, %buf83[%i2] : memref<1xi32>
            aie.objectfifo.release @fifo1(Consume) [2]

            // Pop 1 object off queue
            %subview2_obj = aie.objectfifo.acquire @fifo1(Consume) : memref<1xi32>
            %v88 = memref.load %subview2_obj[%i0] : memref<1xi32>
            memref.store %v88, %buf83[%i3] : memref<1xi32>
            aie.objectfifo.release @fifo1(Consume) [1]

            aie.end
        }

    }
}
