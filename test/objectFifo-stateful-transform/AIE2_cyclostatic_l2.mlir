// In this example, an AIE core pushes data into a memtile, in a one-by-one
// fashion. The memtile forwards this one-by-one to a consumer tile. The 
// consumer tile cyclostatically consumes {1, 2, 1} elements at a time.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// This test does not currently pass, because objectFifo.release cannot be
// used in conjunction with objectFifo.link.

module @aie2_cyclostatic_l2 {
    AIE.device(xcve2302) {

        %tile22 = AIE.tile(2, 2)  // producer tile
        %memtile = AIE.tile(1, 2) // mem tile
        %tile83 = AIE.tile(8, 3)  // consumer tile
        %buf83  = AIE.buffer(%tile83) {sym_name = "buf83"} : memref<4xi32>

        // ObjectFifo that can hold 4 memref<4xi32>s, populated by tile22 and
        // consumed by tile23
        %fifo0 = AIE.objectFifo.createObjectFifo(%tile22, {%memtile}, 4 : i32) {sym_name = "fifo0"} : !AIE.objectFifo<memref<4xi32>>
        %fifo1 = AIE.objectFifo.createObjectFifo(%memtile, {%tile83}, 4 : i32) {sym_name = "fifo1"} : !AIE.objectFifo<memref<4xi32>>
        AIE.objectFifo.link(%fifo0, {%fifo1}) : (!AIE.objectFifo<memref<4xi32>>, !AIE.objectFifo<memref<4xi32>>)

        // Producer core
        %core22 = AIE.core(%tile22) {
            %i0 = arith.constant 0 : index
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            
            // Push 55
            %subview0 = AIE.objectFifo.acquire<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            memref.store %c55, %subview0_obj[%i0] : memref<4xi32>
            AIE.objectFifo.release<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1)

            // Push 66
            %subview1 = AIE.objectFifo.acquire<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview1_obj = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            memref.store %c66, %subview1_obj[%i0] : memref<4xi32>
            AIE.objectFifo.release<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1)

            // Push 77
            %subview2 = AIE.objectFifo.acquire<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview2_obj = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            memref.store %c77, %subview2_obj[%i0] : memref<4xi32>
            AIE.objectFifo.release<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1)

            // Push 88
            %subview3 = AIE.objectFifo.acquire<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview3_obj = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            memref.store %c88, %subview3_obj[%i0] : memref<4xi32>
            AIE.objectFifo.release<Produce>(%fifo0 : !AIE.objectFifo<memref<4xi32>>, 1)

            AIE.end
        }

        // Consumer core
        %core28 = AIE.core(%tile83) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Pop 1 object off queue
            %subview0 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<4xi32>>, 1) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            %v55 = memref.load %subview0_obj[%i0] : memref<4xi32>
            memref.store %v55, %buf83[%i0] : memref<4xi32>
            AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<4xi32>>, 1)

            // Pop 2 objects off queue
            %subview1 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<4xi32>>, 2) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview1_obj0 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            %subview1_obj1 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            %v66 = memref.load %subview1_obj0[%i0] : memref<4xi32>
            %v77 = memref.load %subview1_obj1[%i0] : memref<4xi32>
            memref.store %v66, %buf83[%i1] : memref<4xi32>
            memref.store %v77, %buf83[%i2] : memref<4xi32>
            AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<4xi32>>, 2)

            // Pop 1 object off queue
            %subview2 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<4xi32>>, 1) : !AIE.objectFifoSubview<memref<4xi32>>
            %subview2_obj = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<4xi32>> -> memref<4xi32>
            %v88 = memref.load %subview2_obj[%i0] : memref<4xi32>
            memref.store %v88, %buf83[%i3] : memref<4xi32>
            AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<4xi32>>, 1)

            AIE.end
        }

    }
}