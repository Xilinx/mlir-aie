// HW: PASS
// router_properties.py npu2 seed 276 routed with tree 27's branches on arbiters of their own. On HW, with every tile receiver
// observed: 20 PASS.
module {
  aie.device(npu2) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %l_0_1_0 = aie.lock(%t_0_1, 0) {init = 6 : i32, sym_name = "l_0_1_0"}
    %l_0_1_1 = aie.lock(%t_0_1, 1) {init = 0 : i32, sym_name = "l_0_1_1"}
    %l_0_3_0 = aie.lock(%t_0_3, 0) {init = 2 : i32, sym_name = "l_0_3_0"}
    %l_0_3_1 = aie.lock(%t_0_3, 1) {init = 0 : i32, sym_name = "l_0_3_1"}
    %l_0_4_0 = aie.lock(%t_0_4, 0) {init = 1 : i32, sym_name = "l_0_4_0"}
    %l_0_4_1 = aie.lock(%t_0_4, 1) {init = 0 : i32, sym_name = "l_0_4_1"}
    %l_1_3_0 = aie.lock(%t_1_3, 0) {init = 2 : i32, sym_name = "l_1_3_0"}
    %l_1_3_1 = aie.lock(%t_1_3, 1) {init = 0 : i32, sym_name = "l_1_3_1"}
    %l_1_4_0 = aie.lock(%t_1_4, 0) {init = 2 : i32, sym_name = "l_1_4_0"}
    %l_1_4_1 = aie.lock(%t_1_4, 1) {init = 0 : i32, sym_name = "l_1_4_1"}
    %l_1_5_0 = aie.lock(%t_1_5, 0) {init = 1 : i32, sym_name = "l_1_5_0"}
    %l_1_5_1 = aie.lock(%t_1_5, 1) {init = 0 : i32, sym_name = "l_1_5_1"}
    %l_1_5_2 = aie.lock(%t_1_5, 2) {init = 0 : i32, sym_name = "l_1_5_2"}
    %l_1_5_3 = aie.lock(%t_1_5, 3) {init = 1 : i32, sym_name = "l_1_5_3"}
    %l_1_5_4 = aie.lock(%t_1_5, 4) {init = 0 : i32, sym_name = "l_1_5_4"}
    %l_1_5_5 = aie.lock(%t_1_5, 5) {init = 0 : i32, sym_name = "l_1_5_5"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<64xi32> = dense<[1509949440, 1509949441, 1509949442, 1509949443, 1509949444, 1509949445, 1509949446, 1509949447, 1509949448, 1509949449, 1509949450, 1509949451, 1509949452, 1509949453, 1509949454, 1509949455, 1509949456, 1509949457, 1509949458, 1509949459, 1509949460, 1509949461, 1509949462, 1509949463, 1509949464, 1509949465, 1509949466, 1509949467, 1509949468, 1509949469, 1509949470, 1509949471, 1509949472, 1509949473, 1509949474, 1509949475, 1509949476, 1509949477, 1509949478, 1509949479, 1509949480, 1509949481, 1509949482, 1509949483, 1509949484, 1509949485, 1509949486, 1509949487, 1509949488, 1509949489, 1509949490, 1509949491, 1509949492, 1509949493, 1509949494, 1509949495, 1509949496, 1509949497, 1509949498, 1509949499, 1509949500, 1509949501, 1509949502, 1509949503]>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<64xi32> = dense<[1510014976, 1510014977, 1510014978, 1510014979, 1510014980, 1510014981, 1510014982, 1510014983, 1510014984, 1510014985, 1510014986, 1510014987, 1510014988, 1510014989, 1510014990, 1510014991, 1510014992, 1510014993, 1510014994, 1510014995, 1510014996, 1510014997, 1510014998, 1510014999, 1510015000, 1510015001, 1510015002, 1510015003, 1510015004, 1510015005, 1510015006, 1510015007, 1510015008, 1510015009, 1510015010, 1510015011, 1510015012, 1510015013, 1510015014, 1510015015, 1510015016, 1510015017, 1510015018, 1510015019, 1510015020, 1510015021, 1510015022, 1510015023, 1510015024, 1510015025, 1510015026, 1510015027, 1510015028, 1510015029, 1510015030, 1510015031, 1510015032, 1510015033, 1510015034, 1510015035, 1510015036, 1510015037, 1510015038, 1510015039]>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<256xi32>
    %b_0_1_3 = aie.buffer(%t_0_1) {sym_name = "b_0_1_3"} : memref<128xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 5, ^p0b0, ^p1, repeat_count = 2)
    ^p0b0:
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_1_0 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
      aie.use_lock(%l_0_1_1, Release, %c1)
      aie.next_bd ^p0b1
    ^p0b1:
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_1_1 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 22>}
      aie.use_lock(%l_0_1_1, Release, %c1)
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^p2)
    ^p1b0:
      aie.dma_bd(%b_0_1_2 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(S2MM, 2, ^p2b0, ^end)
    ^p2b0:
      aie.dma_bd(%b_0_1_3 : memref<128xi32> offset = 0 len = 128)
      aie.next_bd ^p2b0
    ^end:
      aie.end
    }
    %b_0_3_4 = aie.buffer(%t_0_3) {sym_name = "b_0_3_4"} : memref<64xi32> = dense<[1510080512, 1510080513, 1510080514, 1510080515, 1510080516, 1510080517, 1510080518, 1510080519, 1510080520, 1510080521, 1510080522, 1510080523, 1510080524, 1510080525, 1510080526, 1510080527, 1510080528, 1510080529, 1510080530, 1510080531, 1510080532, 1510080533, 1510080534, 1510080535, 1510080536, 1510080537, 1510080538, 1510080539, 1510080540, 1510080541, 1510080542, 1510080543, 1510080544, 1510080545, 1510080546, 1510080547, 1510080548, 1510080549, 1510080550, 1510080551, 1510080552, 1510080553, 1510080554, 1510080555, 1510080556, 1510080557, 1510080558, 1510080559, 1510080560, 1510080561, 1510080562, 1510080563, 1510080564, 1510080565, 1510080566, 1510080567, 1510080568, 1510080569, 1510080570, 1510080571, 1510080572, 1510080573, 1510080574, 1510080575]>
    %dma_0_3 = aie.mem(%t_0_3) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end, repeat_count = 1)
    ^p0b0:
      aie.use_lock(%l_0_3_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_3_4 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
      aie.use_lock(%l_0_3_1, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_4_5 = aie.buffer(%t_0_4) {sym_name = "b_0_4_5"} : memref<64xi32> = dense<[1510146048, 1510146049, 1510146050, 1510146051, 1510146052, 1510146053, 1510146054, 1510146055, 1510146056, 1510146057, 1510146058, 1510146059, 1510146060, 1510146061, 1510146062, 1510146063, 1510146064, 1510146065, 1510146066, 1510146067, 1510146068, 1510146069, 1510146070, 1510146071, 1510146072, 1510146073, 1510146074, 1510146075, 1510146076, 1510146077, 1510146078, 1510146079, 1510146080, 1510146081, 1510146082, 1510146083, 1510146084, 1510146085, 1510146086, 1510146087, 1510146088, 1510146089, 1510146090, 1510146091, 1510146092, 1510146093, 1510146094, 1510146095, 1510146096, 1510146097, 1510146098, 1510146099, 1510146100, 1510146101, 1510146102, 1510146103, 1510146104, 1510146105, 1510146106, 1510146107, 1510146108, 1510146109, 1510146110, 1510146111]>
    %dma_0_4 = aie.mem(%t_0_4) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_4_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_4_5 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
      aie.use_lock(%l_0_4_1, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_1_3_6 = aie.buffer(%t_1_3) {sym_name = "b_1_3_6"} : memref<64xi32> = dense<[1510211584, 1510211585, 1510211586, 1510211587, 1510211588, 1510211589, 1510211590, 1510211591, 1510211592, 1510211593, 1510211594, 1510211595, 1510211596, 1510211597, 1510211598, 1510211599, 1510211600, 1510211601, 1510211602, 1510211603, 1510211604, 1510211605, 1510211606, 1510211607, 1510211608, 1510211609, 1510211610, 1510211611, 1510211612, 1510211613, 1510211614, 1510211615, 1510211616, 1510211617, 1510211618, 1510211619, 1510211620, 1510211621, 1510211622, 1510211623, 1510211624, 1510211625, 1510211626, 1510211627, 1510211628, 1510211629, 1510211630, 1510211631, 1510211632, 1510211633, 1510211634, 1510211635, 1510211636, 1510211637, 1510211638, 1510211639, 1510211640, 1510211641, 1510211642, 1510211643, 1510211644, 1510211645, 1510211646, 1510211647]>
    %dma_1_3 = aie.mem(%t_1_3) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end, repeat_count = 1)
    ^p0b0:
      aie.use_lock(%l_1_3_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_3_6 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.use_lock(%l_1_3_1, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_1_4_7 = aie.buffer(%t_1_4) {sym_name = "b_1_4_7"} : memref<64xi32> = dense<[1510277120, 1510277121, 1510277122, 1510277123, 1510277124, 1510277125, 1510277126, 1510277127, 1510277128, 1510277129, 1510277130, 1510277131, 1510277132, 1510277133, 1510277134, 1510277135, 1510277136, 1510277137, 1510277138, 1510277139, 1510277140, 1510277141, 1510277142, 1510277143, 1510277144, 1510277145, 1510277146, 1510277147, 1510277148, 1510277149, 1510277150, 1510277151, 1510277152, 1510277153, 1510277154, 1510277155, 1510277156, 1510277157, 1510277158, 1510277159, 1510277160, 1510277161, 1510277162, 1510277163, 1510277164, 1510277165, 1510277166, 1510277167, 1510277168, 1510277169, 1510277170, 1510277171, 1510277172, 1510277173, 1510277174, 1510277175, 1510277176, 1510277177, 1510277178, 1510277179, 1510277180, 1510277181, 1510277182, 1510277183]>
    %b_1_4_8 = aie.buffer(%t_1_4) {sym_name = "b_1_4_8"} : memref<64xi32> = dense<[1510342656, 1510342657, 1510342658, 1510342659, 1510342660, 1510342661, 1510342662, 1510342663, 1510342664, 1510342665, 1510342666, 1510342667, 1510342668, 1510342669, 1510342670, 1510342671, 1510342672, 1510342673, 1510342674, 1510342675, 1510342676, 1510342677, 1510342678, 1510342679, 1510342680, 1510342681, 1510342682, 1510342683, 1510342684, 1510342685, 1510342686, 1510342687, 1510342688, 1510342689, 1510342690, 1510342691, 1510342692, 1510342693, 1510342694, 1510342695, 1510342696, 1510342697, 1510342698, 1510342699, 1510342700, 1510342701, 1510342702, 1510342703, 1510342704, 1510342705, 1510342706, 1510342707, 1510342708, 1510342709, 1510342710, 1510342711, 1510342712, 1510342713, 1510342714, 1510342715, 1510342716, 1510342717, 1510342718, 1510342719]>
    %b_1_4_9 = aie.buffer(%t_1_4) {sym_name = "b_1_4_9"} : memref<512xi32>
    %dma_1_4 = aie.mem(%t_1_4) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_1_4_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_4_7 : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%l_1_4_1, Release, %c1)
      aie.next_bd ^p0b1
    ^p0b1:
      aie.use_lock(%l_1_4_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_4_8 : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%l_1_4_1, Release, %c1)
      aie.next_bd ^end
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_1_4_9 : memref<512xi32> offset = 0 len = 512)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_1_5_10 = aie.buffer(%t_1_5) {sym_name = "b_1_5_10"} : memref<65xi32>
    %b_1_5_11 = aie.buffer(%t_1_5) {sym_name = "b_1_5_11"} : memref<64xi32> = dense<[1510408192, 1510408193, 1510408194, 1510408195, 1510408196, 1510408197, 1510408198, 1510408199, 1510408200, 1510408201, 1510408202, 1510408203, 1510408204, 1510408205, 1510408206, 1510408207, 1510408208, 1510408209, 1510408210, 1510408211, 1510408212, 1510408213, 1510408214, 1510408215, 1510408216, 1510408217, 1510408218, 1510408219, 1510408220, 1510408221, 1510408222, 1510408223, 1510408224, 1510408225, 1510408226, 1510408227, 1510408228, 1510408229, 1510408230, 1510408231, 1510408232, 1510408233, 1510408234, 1510408235, 1510408236, 1510408237, 1510408238, 1510408239, 1510408240, 1510408241, 1510408242, 1510408243, 1510408244, 1510408245, 1510408246, 1510408247, 1510408248, 1510408249, 1510408250, 1510408251, 1510408252, 1510408253, 1510408254, 1510408255]>
    %b_1_5_12 = aie.buffer(%t_1_5) {sym_name = "b_1_5_12"} : memref<65xi32>
    %b_1_5_13 = aie.buffer(%t_1_5) {sym_name = "b_1_5_13"} : memref<64xi32> = dense<[1510473728, 1510473729, 1510473730, 1510473731, 1510473732, 1510473733, 1510473734, 1510473735, 1510473736, 1510473737, 1510473738, 1510473739, 1510473740, 1510473741, 1510473742, 1510473743, 1510473744, 1510473745, 1510473746, 1510473747, 1510473748, 1510473749, 1510473750, 1510473751, 1510473752, 1510473753, 1510473754, 1510473755, 1510473756, 1510473757, 1510473758, 1510473759, 1510473760, 1510473761, 1510473762, 1510473763, 1510473764, 1510473765, 1510473766, 1510473767, 1510473768, 1510473769, 1510473770, 1510473771, 1510473772, 1510473773, 1510473774, 1510473775, 1510473776, 1510473777, 1510473778, 1510473779, 1510473780, 1510473781, 1510473782, 1510473783, 1510473784, 1510473785, 1510473786, 1510473787, 1510473788, 1510473789, 1510473790, 1510473791]>
    %dma_1_5 = aie.mem(%t_1_5) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_1_5_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_5_10 : memref<65xi32> offset = 0 len = 65)
      aie.use_lock(%l_1_5_1, Release, %c1)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 0, ^p1b0, ^p2, repeat_count = 1)
    ^p1b0:
      aie.use_lock(%l_1_5_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_5_11 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%l_1_5_0, Release, %c1)
      aie.next_bd ^end
    ^p2:
      %d2 = aie.dma_start(S2MM, 1, ^p2b0, ^p3)
    ^p2b0:
      aie.use_lock(%l_1_5_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_5_12 : memref<65xi32> offset = 0 len = 65)
      aie.use_lock(%l_1_5_4, Release, %c1)
      aie.next_bd ^p2b0
    ^p3:
      %d3 = aie.dma_start(MM2S, 1, ^p3b0, ^end, repeat_count = 1)
    ^p3b0:
      aie.use_lock(%l_1_5_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_5_13 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%l_1_5_3, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_2_14 = aie.buffer(%t_0_2) {sym_name = "b_0_2_14"} : memref<256xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_2_14 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_5_15 = aie.buffer(%t_0_5) {sym_name = "b_0_5_15"} : memref<16xi32>
    %b_0_5_16 = aie.buffer(%t_0_5) {sym_name = "b_0_5_16"} : memref<512xi32>
    %dma_0_5 = aie.mem(%t_0_5) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_5_15 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_0_5_16 : memref<512xi32> offset = 0 len = 512)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_1_1_17 = aie.buffer(%t_1_1) {sym_name = "b_1_1_17"} : memref<16xi32>
    %b_1_1_18 = aie.buffer(%t_1_1) {sym_name = "b_1_1_18"} : memref<8xi32>
    %dma_1_1 = aie.memtile_dma(%t_1_1) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_1_1_17 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 3, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_1_1_18 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_1_2_19 = aie.buffer(%t_1_2) {sym_name = "b_1_2_19"} : memref<16xi32>
    %dma_1_2 = aie.mem(%t_1_2) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_1_2_19 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %core_1_5 = aie.core(%t_1_5) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_1_5_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_1_5_2, Release, %c1)
      aie.use_lock(%l_1_5_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_1_5_5, Release, %c1)
      aie.use_lock(%l_1_5_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_1_5_2, Release, %c1)
      aie.use_lock(%l_1_5_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_1_5_5, Release, %c1)
      aie.end
    }
    aie.flow(%t_1_4, DMA : 0, %t_1_1, DMA : 3)
    aie.flow(%t_1_4, DMA : 0, %t_0_5, DMA : 1)
    aie.packet_flow(6) { aie.packet_source<%t_1_3, DMA : 1> aie.packet_dest<%t_1_5, DMA : 0> } {keep_pkt_header = true}
    aie.packet_flow(29) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_1_5, DMA : 1> } {keep_pkt_header = true}
    aie.packet_flow(2) { aie.packet_source<%t_1_5, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%t_1_5, DMA : 1> aie.packet_dest<%t_0_0, DMA : 0> }
    aie.packet_flow(27) { aie.packet_source<%t_1_0, DMA : 1> aie.packet_dest<%t_0_1, DMA : 2> aie.packet_dest<%t_1_2, DMA : 1> aie.packet_dest<%t_1_4, DMA : 1> } {priority_route = true}
    aie.packet_flow(29, mask = 29) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_source<%t_0_4, DMA : 1> aie.packet_dest<%t_0_1, DMA : 1> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_1_1, DMA : 0> } {priority_route = true}
    aie.packet_flow(9) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_1_2, DMA : 1> } {keep_pkt_header = false}
    aie.packet_flow(22) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_5, DMA : 0> aie.packet_dest<%t_1_0, DMA : 0> } {keep_pkt_header = false}
    aie.shim_dma_allocation @out0_0(%t_0_0, S2MM, 0)
    aie.shim_dma_allocation @out1_0(%t_1_0, S2MM, 0)
    aie.runtime_sequence @seq0(%in: memref<256xi32>, %out: memref<448xi32>) {
      %task0 = aiex.dma_configure_task(%t_0_0, MM2S, 1) {
        aie.dma_bd(%in : memref<256xi32> offset = 0 len = 64) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
        aie.next_bd ^bb1
      ^bb1:
        aie.dma_bd(%in : memref<256xi32> offset = 64 len = 64) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
        aie.end
      } {issue_token = true, repeat_count = 2 : i32}
      %task1 = aiex.dma_configure_task(%t_1_0, MM2S, 1) {
        aie.dma_bd(%in : memref<256xi32> offset = 128 len = 64) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
        aie.next_bd ^bb1
      ^bb1:
        aie.dma_bd(%in : memref<256xi32> offset = 192 len = 64) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
        aie.end
      } {issue_token = true}
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 256][1, 1, 1, 192][0, 0, 0, 1]) { metadata = @out1_0, id = 2 : i64, issue_token = true } : memref<448xi32>
      aiex.dma_start_task(%task0)
      aiex.dma_start_task(%task1)
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) { metadata = @out0_0, id = 2 : i64, issue_token = true } : memref<448xi32>
      aiex.dma_await_task(%task0)
      aiex.npu.dma_wait {symbol = @out0_0}
      aiex.dma_await_task(%task1)
      aiex.npu.dma_wait {symbol = @out1_0}
    }
  }
}
// -----

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<4> (3)
      %3 = aie.amsel<5> (3)
      %4 = aie.masterset(South : 2, %0)
      %5 = aie.masterset(North : 3, %3) {is_ctrl_pkt_overlay}
      %6 = aie.masterset(North : 4, %2) {is_ctrl_pkt_overlay}
      %7 = aie.masterset(East : 0, %1)
      %8 = aie.masterset(East : 3, %3) {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 2) {
        aie.rule(31, 22, %1)
      }
      aie.packet_rules(North : 3) {
        aie.rule(29, 29, %3)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 7) {
        aie.rule(29, 29, %3) {priority_route}
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(East : 2) {
        aie.rule(31, 27, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 1) {
        aie.rule(24, 0, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<North : 1, South : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<0> (1)
      %2 = aie.amsel<3> (3)
      %3 = aie.amsel<4> (3)
      %4 = aie.amsel<5> (3)
      %5 = aie.masterset(DMA : 1, %4) {is_ctrl_pkt_overlay}
      %6 = aie.masterset(DMA : 2, %3) {is_ctrl_pkt_overlay}
      %7 = aie.masterset(South : 2, %1)
      %8 = aie.masterset(South : 3, %2) {is_ctrl_pkt_overlay}
      %9 = aie.masterset(North : 3, %4) {is_ctrl_pkt_overlay}
      %10 = aie.masterset(North : 5, %0, %1)
      aie.packet_rules(DMA : 5) {
        aie.rule(31, 22, %1)
        aie.rule(31, 9, %0)
      }
      aie.packet_rules(South : 3) {
        aie.rule(29, 29, %4)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 3) {
        aie.rule(29, 29, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 4) {
        aie.rule(31, 27, %3)
      } {is_ctrl_pkt_overlay}
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<4> (3)
      %4 = aie.amsel<5> (3)
      %5 = aie.masterset(DMA : 0, %4) {is_ctrl_pkt_overlay}
      %6 = aie.masterset(South : 1, %2)
      %7 = aie.masterset(South : 3, %3) {is_ctrl_pkt_overlay}
      %8 = aie.masterset(North : 1, %1)
      %9 = aie.masterset(East : 0, %0)
      aie.packet_rules(South : 5) {
        aie.rule(31, 22, %1)
        aie.rule(31, 9, %0)
      }
      aie.packet_rules(South : 3) {
        aie.rule(29, 29, %4)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 2) {
        aie.rule(29, 29, %3)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(East : 1) {
        aie.rule(24, 0, %2)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(South : 2, %2) {is_ctrl_pkt_overlay}
      %4 = aie.masterset(North : 0, %1)
      %5 = aie.masterset(North : 5, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 22, %1)
      }
      aie.packet_rules(North : 0) {
        aie.rule(29, 29, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 29, %0)
      }
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(South : 0, %2) {is_ctrl_pkt_overlay}
      %4 = aie.masterset(North : 1, %0)
      %5 = aie.masterset(North : 5, %1)
      aie.packet_rules(South : 0) {
        aie.rule(31, 22, %0)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(29, 29, %2) {priority_route}
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 5) {
        aie.rule(31, 29, %1)
      }
    }
    %tile_0_5 = aie.tile(0, 5)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<5> (2)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(South : 2, %0) {keep_pkt_header = false}
      %4 = aie.masterset(West : 2, %2) {is_ctrl_pkt_overlay}
      %5 = aie.masterset(North : 1, %2) {is_ctrl_pkt_overlay}
      %6 = aie.masterset(North : 3, %1) {is_ctrl_pkt_overlay}
      %7 = aie.masterset(North : 5, %2) {is_ctrl_pkt_overlay}
      aie.packet_rules(West : 0) {
        aie.rule(31, 22, %0)
      }
      aie.packet_rules(West : 3) {
        aie.rule(29, 29, %1)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 7) {
        aie.rule(31, 27, %2) {priority_route}
      } {is_ctrl_pkt_overlay}
    }
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %l_0_1_0 = aie.lock(%mem_tile_0_1, 0) {init = 6 : i32, sym_name = "l_0_1_0"}
    %l_0_1_1 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "l_0_1_1"}
    %l_0_3_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "l_0_3_0"}
    %l_0_3_1 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "l_0_3_1"}
    %l_0_4_0 = aie.lock(%tile_0_4, 0) {init = 1 : i32, sym_name = "l_0_4_0"}
    %l_0_4_1 = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "l_0_4_1"}
    %l_1_3_0 = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "l_1_3_0"}
    %l_1_3_1 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "l_1_3_1"}
    %l_1_4_0 = aie.lock(%tile_1_4, 0) {init = 2 : i32, sym_name = "l_1_4_0"}
    %l_1_4_1 = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "l_1_4_1"}
    %l_1_5_0 = aie.lock(%tile_1_5, 0) {init = 1 : i32, sym_name = "l_1_5_0"}
    %l_1_5_1 = aie.lock(%tile_1_5, 1) {init = 0 : i32, sym_name = "l_1_5_1"}
    %l_1_5_2 = aie.lock(%tile_1_5, 2) {init = 0 : i32, sym_name = "l_1_5_2"}
    %l_1_5_3 = aie.lock(%tile_1_5, 3) {init = 1 : i32, sym_name = "l_1_5_3"}
    %l_1_5_4 = aie.lock(%tile_1_5, 4) {init = 0 : i32, sym_name = "l_1_5_4"}
    %l_1_5_5 = aie.lock(%tile_1_5, 5) {init = 0 : i32, sym_name = "l_1_5_5"}
    %b_0_1_0 = aie.buffer(%mem_tile_0_1) {sym_name = "b_0_1_0"} : memref<64xi32> = dense<[1509949440, 1509949441, 1509949442, 1509949443, 1509949444, 1509949445, 1509949446, 1509949447, 1509949448, 1509949449, 1509949450, 1509949451, 1509949452, 1509949453, 1509949454, 1509949455, 1509949456, 1509949457, 1509949458, 1509949459, 1509949460, 1509949461, 1509949462, 1509949463, 1509949464, 1509949465, 1509949466, 1509949467, 1509949468, 1509949469, 1509949470, 1509949471, 1509949472, 1509949473, 1509949474, 1509949475, 1509949476, 1509949477, 1509949478, 1509949479, 1509949480, 1509949481, 1509949482, 1509949483, 1509949484, 1509949485, 1509949486, 1509949487, 1509949488, 1509949489, 1509949490, 1509949491, 1509949492, 1509949493, 1509949494, 1509949495, 1509949496, 1509949497, 1509949498, 1509949499, 1509949500, 1509949501, 1509949502, 1509949503]>
    %b_0_1_1 = aie.buffer(%mem_tile_0_1) {sym_name = "b_0_1_1"} : memref<64xi32> = dense<[1510014976, 1510014977, 1510014978, 1510014979, 1510014980, 1510014981, 1510014982, 1510014983, 1510014984, 1510014985, 1510014986, 1510014987, 1510014988, 1510014989, 1510014990, 1510014991, 1510014992, 1510014993, 1510014994, 1510014995, 1510014996, 1510014997, 1510014998, 1510014999, 1510015000, 1510015001, 1510015002, 1510015003, 1510015004, 1510015005, 1510015006, 1510015007, 1510015008, 1510015009, 1510015010, 1510015011, 1510015012, 1510015013, 1510015014, 1510015015, 1510015016, 1510015017, 1510015018, 1510015019, 1510015020, 1510015021, 1510015022, 1510015023, 1510015024, 1510015025, 1510015026, 1510015027, 1510015028, 1510015029, 1510015030, 1510015031, 1510015032, 1510015033, 1510015034, 1510015035, 1510015036, 1510015037, 1510015038, 1510015039]>
    %b_0_1_2 = aie.buffer(%mem_tile_0_1) {sym_name = "b_0_1_2"} : memref<256xi32> 
    %b_0_1_3 = aie.buffer(%mem_tile_0_1) {sym_name = "b_0_1_3"} : memref<128xi32> 
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 5, ^bb1, ^bb3, repeat_count = 2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_0_1_0 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 9>}
      aie.use_lock(%l_0_1_1, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_0_1_1 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 22>}
      aie.use_lock(%l_0_1_1, Release, %c1_i32)
      aie.next_bd ^bb7
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.dma_bd(%b_0_1_2 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 2, ^bb6, ^bb7)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.dma_bd(%b_0_1_3 : memref<128xi32> offset = 0 len = 128)
      aie.next_bd ^bb6
    ^bb7:  // 2 preds: ^bb2, ^bb5
      aie.end
    }
    %b_0_3_4 = aie.buffer(%tile_0_3) {sym_name = "b_0_3_4"} : memref<64xi32> = dense<[1510080512, 1510080513, 1510080514, 1510080515, 1510080516, 1510080517, 1510080518, 1510080519, 1510080520, 1510080521, 1510080522, 1510080523, 1510080524, 1510080525, 1510080526, 1510080527, 1510080528, 1510080529, 1510080530, 1510080531, 1510080532, 1510080533, 1510080534, 1510080535, 1510080536, 1510080537, 1510080538, 1510080539, 1510080540, 1510080541, 1510080542, 1510080543, 1510080544, 1510080545, 1510080546, 1510080547, 1510080548, 1510080549, 1510080550, 1510080551, 1510080552, 1510080553, 1510080554, 1510080555, 1510080556, 1510080557, 1510080558, 1510080559, 1510080560, 1510080561, 1510080562, 1510080563, 1510080564, 1510080565, 1510080566, 1510080567, 1510080568, 1510080569, 1510080570, 1510080571, 1510080572, 1510080573, 1510080574, 1510080575]>
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2, repeat_count = 1)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%l_0_3_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_0_3_4 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
      aie.use_lock(%l_0_3_1, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %b_0_4_5 = aie.buffer(%tile_0_4) {sym_name = "b_0_4_5"} : memref<64xi32> = dense<[1510146048, 1510146049, 1510146050, 1510146051, 1510146052, 1510146053, 1510146054, 1510146055, 1510146056, 1510146057, 1510146058, 1510146059, 1510146060, 1510146061, 1510146062, 1510146063, 1510146064, 1510146065, 1510146066, 1510146067, 1510146068, 1510146069, 1510146070, 1510146071, 1510146072, 1510146073, 1510146074, 1510146075, 1510146076, 1510146077, 1510146078, 1510146079, 1510146080, 1510146081, 1510146082, 1510146083, 1510146084, 1510146085, 1510146086, 1510146087, 1510146088, 1510146089, 1510146090, 1510146091, 1510146092, 1510146093, 1510146094, 1510146095, 1510146096, 1510146097, 1510146098, 1510146099, 1510146100, 1510146101, 1510146102, 1510146103, 1510146104, 1510146105, 1510146106, 1510146107, 1510146108, 1510146109, 1510146110, 1510146111]>
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%l_0_4_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_0_4_5 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
      aie.use_lock(%l_0_4_1, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %b_1_3_6 = aie.buffer(%tile_1_3) {sym_name = "b_1_3_6"} : memref<64xi32> = dense<[1510211584, 1510211585, 1510211586, 1510211587, 1510211588, 1510211589, 1510211590, 1510211591, 1510211592, 1510211593, 1510211594, 1510211595, 1510211596, 1510211597, 1510211598, 1510211599, 1510211600, 1510211601, 1510211602, 1510211603, 1510211604, 1510211605, 1510211606, 1510211607, 1510211608, 1510211609, 1510211610, 1510211611, 1510211612, 1510211613, 1510211614, 1510211615, 1510211616, 1510211617, 1510211618, 1510211619, 1510211620, 1510211621, 1510211622, 1510211623, 1510211624, 1510211625, 1510211626, 1510211627, 1510211628, 1510211629, 1510211630, 1510211631, 1510211632, 1510211633, 1510211634, 1510211635, 1510211636, 1510211637, 1510211638, 1510211639, 1510211640, 1510211641, 1510211642, 1510211643, 1510211644, 1510211645, 1510211646, 1510211647]>
    %mem_1_3 = aie.mem(%tile_1_3) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2, repeat_count = 1)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%l_1_3_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_3_6 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.use_lock(%l_1_3_1, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %b_1_4_7 = aie.buffer(%tile_1_4) {sym_name = "b_1_4_7"} : memref<64xi32> = dense<[1510277120, 1510277121, 1510277122, 1510277123, 1510277124, 1510277125, 1510277126, 1510277127, 1510277128, 1510277129, 1510277130, 1510277131, 1510277132, 1510277133, 1510277134, 1510277135, 1510277136, 1510277137, 1510277138, 1510277139, 1510277140, 1510277141, 1510277142, 1510277143, 1510277144, 1510277145, 1510277146, 1510277147, 1510277148, 1510277149, 1510277150, 1510277151, 1510277152, 1510277153, 1510277154, 1510277155, 1510277156, 1510277157, 1510277158, 1510277159, 1510277160, 1510277161, 1510277162, 1510277163, 1510277164, 1510277165, 1510277166, 1510277167, 1510277168, 1510277169, 1510277170, 1510277171, 1510277172, 1510277173, 1510277174, 1510277175, 1510277176, 1510277177, 1510277178, 1510277179, 1510277180, 1510277181, 1510277182, 1510277183]>
    %b_1_4_8 = aie.buffer(%tile_1_4) {sym_name = "b_1_4_8"} : memref<64xi32> = dense<[1510342656, 1510342657, 1510342658, 1510342659, 1510342660, 1510342661, 1510342662, 1510342663, 1510342664, 1510342665, 1510342666, 1510342667, 1510342668, 1510342669, 1510342670, 1510342671, 1510342672, 1510342673, 1510342674, 1510342675, 1510342676, 1510342677, 1510342678, 1510342679, 1510342680, 1510342681, 1510342682, 1510342683, 1510342684, 1510342685, 1510342686, 1510342687, 1510342688, 1510342689, 1510342690, 1510342691, 1510342692, 1510342693, 1510342694, 1510342695, 1510342696, 1510342697, 1510342698, 1510342699, 1510342700, 1510342701, 1510342702, 1510342703, 1510342704, 1510342705, 1510342706, 1510342707, 1510342708, 1510342709, 1510342710, 1510342711, 1510342712, 1510342713, 1510342714, 1510342715, 1510342716, 1510342717, 1510342718, 1510342719]>
    %b_1_4_9 = aie.buffer(%tile_1_4) {sym_name = "b_1_4_9"} : memref<512xi32> 
    %mem_1_4 = aie.mem(%tile_1_4) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%l_1_4_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_4_7 : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%l_1_4_1, Release, %c1_i32)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%l_1_4_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_4_8 : memref<64xi32> offset = 0 len = 64)
      aie.use_lock(%l_1_4_1, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.dma_bd(%b_1_4_9 : memref<512xi32> offset = 0 len = 512)
      aie.next_bd ^bb4
    ^bb5:  // 2 preds: ^bb2, ^bb3
      aie.end
    }
    %b_1_5_10 = aie.buffer(%tile_1_5) {sym_name = "b_1_5_10"} : memref<65xi32> 
    %b_1_5_11 = aie.buffer(%tile_1_5) {sym_name = "b_1_5_11"} : memref<64xi32> = dense<[1510408192, 1510408193, 1510408194, 1510408195, 1510408196, 1510408197, 1510408198, 1510408199, 1510408200, 1510408201, 1510408202, 1510408203, 1510408204, 1510408205, 1510408206, 1510408207, 1510408208, 1510408209, 1510408210, 1510408211, 1510408212, 1510408213, 1510408214, 1510408215, 1510408216, 1510408217, 1510408218, 1510408219, 1510408220, 1510408221, 1510408222, 1510408223, 1510408224, 1510408225, 1510408226, 1510408227, 1510408228, 1510408229, 1510408230, 1510408231, 1510408232, 1510408233, 1510408234, 1510408235, 1510408236, 1510408237, 1510408238, 1510408239, 1510408240, 1510408241, 1510408242, 1510408243, 1510408244, 1510408245, 1510408246, 1510408247, 1510408248, 1510408249, 1510408250, 1510408251, 1510408252, 1510408253, 1510408254, 1510408255]>
    %b_1_5_12 = aie.buffer(%tile_1_5) {sym_name = "b_1_5_12"} : memref<65xi32> 
    %b_1_5_13 = aie.buffer(%tile_1_5) {sym_name = "b_1_5_13"} : memref<64xi32> = dense<[1510473728, 1510473729, 1510473730, 1510473731, 1510473732, 1510473733, 1510473734, 1510473735, 1510473736, 1510473737, 1510473738, 1510473739, 1510473740, 1510473741, 1510473742, 1510473743, 1510473744, 1510473745, 1510473746, 1510473747, 1510473748, 1510473749, 1510473750, 1510473751, 1510473752, 1510473753, 1510473754, 1510473755, 1510473756, 1510473757, 1510473758, 1510473759, 1510473760, 1510473761, 1510473762, 1510473763, 1510473764, 1510473765, 1510473766, 1510473767, 1510473768, 1510473769, 1510473770, 1510473771, 1510473772, 1510473773, 1510473774, 1510473775, 1510473776, 1510473777, 1510473778, 1510473779, 1510473780, 1510473781, 1510473782, 1510473783, 1510473784, 1510473785, 1510473786, 1510473787, 1510473788, 1510473789, 1510473790, 1510473791]>
    %mem_1_5 = aie.mem(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%l_1_5_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_5_10 : memref<65xi32> offset = 0 len = 65)
      aie.use_lock(%l_1_5_1, Release, %c1_i32)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4, repeat_count = 1)
    ^bb3:  // pred: ^bb2
      aie.use_lock(%l_1_5_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_5_11 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
      aie.use_lock(%l_1_5_0, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%l_1_5_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_5_12 : memref<65xi32> offset = 0 len = 65)
      aie.use_lock(%l_1_5_4, Release, %c1_i32)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 1, ^bb7, ^bb8, repeat_count = 1)
    ^bb7:  // pred: ^bb6
      aie.use_lock(%l_1_5_5, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%b_1_5_13 : memref<64xi32> offset = 0 len = 64) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
      aie.use_lock(%l_1_5_3, Release, %c1_i32)
      aie.next_bd ^bb8
    ^bb8:  // 3 preds: ^bb3, ^bb6, ^bb7
      aie.end
    }
    %b_0_2_14 = aie.buffer(%tile_0_2) {sym_name = "b_0_2_14"} : memref<256xi32> 
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%b_0_2_14 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %b_0_5_15 = aie.buffer(%tile_0_5) {sym_name = "b_0_5_15"} : memref<16xi32> 
    %b_0_5_16 = aie.buffer(%tile_0_5) {sym_name = "b_0_5_16"} : memref<512xi32> 
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%b_0_5_15 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.dma_bd(%b_0_5_16 : memref<512xi32> offset = 0 len = 512)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %b_1_1_17 = aie.buffer(%mem_tile_1_1) {sym_name = "b_1_1_17"} : memref<16xi32> 
    %b_1_1_18 = aie.buffer(%mem_tile_1_1) {sym_name = "b_1_1_18"} : memref<8xi32> 
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%b_1_1_17 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 3, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.dma_bd(%b_1_1_18 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
    %b_1_2_19 = aie.buffer(%tile_1_2) {sym_name = "b_1_2_19"} : memref<16xi32> 
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(S2MM, 1, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.dma_bd(%b_1_2_19 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %core_1_5 = aie.core(%tile_1_5) {
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%l_1_5_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%l_1_5_2, Release, %c1_i32)
      aie.use_lock(%l_1_5_4, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%l_1_5_5, Release, %c1_i32)
      aie.use_lock(%l_1_5_1, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%l_1_5_2, Release, %c1_i32)
      aie.use_lock(%l_1_5_4, AcquireGreaterEqual, %c1_i32)
      aie.use_lock(%l_1_5_5, Release, %c1_i32)
      aie.end
    }
    aie.shim_dma_allocation @out0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out1_0(%shim_noc_tile_1_0, S2MM, 0)
    aie.runtime_sequence @seq0(%arg0: memref<256xi32>, %arg1: memref<448xi32>) {
      %0 = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 64) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
        aie.next_bd ^bb1
      ^bb1:  // pred: ^bb0
        aie.dma_bd(%arg0 : memref<256xi32> offset = 64 len = 64) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
        aie.end
      } {issue_token = true, repeat_count = 2 : i32}
      %1 = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<256xi32> offset = 128 len = 64) {bd_id = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
        aie.next_bd ^bb1
      ^bb1:  // pred: ^bb0
        aie.dma_bd(%arg0 : memref<256xi32> offset = 192 len = 64) {bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
        aie.end
      } {issue_token = true}
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 256][1, 1, 1, 192][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @out1_0} : memref<448xi32>
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @out0_0} : memref<448xi32>
      aiex.dma_await_task(%0)
      aiex.npu.dma_wait {symbol = @out0_0}
      aiex.dma_await_task(%1)
      aiex.npu.dma_wait {symbol = @out1_0}
    }
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      aie.connect<East : 0, DMA : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0) {keep_pkt_header = false}
      %3 = aie.masterset(East : 3, %1)
      aie.packet_rules(South : 1) {
        aie.rule(31, 22, %0)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 29, %1)
      }
    }
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      aie.connect<North : 1, DMA : 3>
      %0 = aie.amsel<3> (3)
      %1 = aie.amsel<4> (3)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(DMA : 0, %1) {is_ctrl_pkt_overlay}
      %4 = aie.masterset(North : 1, %2) {is_ctrl_pkt_overlay}
      %5 = aie.masterset(North : 5, %0) {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 3) {
        aie.rule(29, 29, %1)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 5) {
        aie.rule(31, 27, %0)
      } {is_ctrl_pkt_overlay}
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<North : 0, South : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<4> (3)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(DMA : 1, %1) {is_ctrl_pkt_overlay, keep_pkt_header = false}
      %4 = aie.masterset(West : 1, %0)
      %5 = aie.masterset(North : 1, %2) {is_ctrl_pkt_overlay}
      aie.packet_rules(West : 0) {
        aie.rule(31, 9, %1)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(South : 5) {
        aie.rule(31, 27, %1)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 2) {
        aie.rule(24, 0, %0)
      }
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<North : 2, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(South : 2, %1)
      %4 = aie.masterset(North : 3, %2) {is_ctrl_pkt_overlay}
      %5 = aie.masterset(North : 5, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 3) {
        aie.rule(24, 0, %1)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 6, %0)
      }
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<DMA : 0, North : 5>
      aie.connect<DMA : 0, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<5> (3)
      %3 = aie.masterset(DMA : 1, %2) {is_ctrl_pkt_overlay}
      %4 = aie.masterset(South : 3, %1)
      %5 = aie.masterset(North : 0, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 27, %2)
      } {is_ctrl_pkt_overlay}
      aie.packet_rules(North : 0) {
        aie.rule(24, 0, %1)
      }
      aie.packet_rules(South : 5) {
        aie.rule(31, 6, %0)
      }
    }
    %switchbox_1_5 = aie.switchbox(%tile_1_5) {
      aie.connect<South : 5, West : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(DMA : 0, %1) {keep_pkt_header = true}
      %4 = aie.masterset(DMA : 1, %2) {keep_pkt_header = true}
      %5 = aie.masterset(South : 0, %0)
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 5, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(West : 3) {
        aie.rule(31, 29, %2)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 6, %1)
      }
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%tile_0_4 : Core, %switchbox_0_4 : Core)
    aie.wire(%tile_0_4 : DMA, %switchbox_0_4 : DMA)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%tile_0_5 : Core, %switchbox_0_5 : Core)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
    aie.wire(%switchbox_0_3 : East, %switchbox_1_3 : West)
    aie.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
    aie.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
    aie.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
    aie.wire(%switchbox_0_4 : East, %switchbox_1_4 : West)
    aie.wire(%tile_1_4 : Core, %switchbox_1_4 : Core)
    aie.wire(%tile_1_4 : DMA, %switchbox_1_4 : DMA)
    aie.wire(%switchbox_1_3 : North, %switchbox_1_4 : South)
    aie.wire(%switchbox_0_5 : East, %switchbox_1_5 : West)
    aie.wire(%tile_1_5 : Core, %switchbox_1_5 : Core)
    aie.wire(%tile_1_5 : DMA, %switchbox_1_5 : DMA)
    aie.wire(%switchbox_1_4 : North, %switchbox_1_5 : South)
  }
}

