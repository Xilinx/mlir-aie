# /scratch/gagandee/mlir-air/utils/mlir-aie/reference_designs/horizontal_diffusion/HDIFF_single_AIE_objectFIFO_ping_pong_scaled/code_hdiff.py -*- Python -*-
#
# (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
#


noc_columns = [2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47]

arraycols = 32  # must be even until 32
broadcast_cores = 0  # only 1-2
arrayrows = 1  # one for processing and one for shimDMA
startrow = 2
startcol = 0
bufsize = 256  # must fit in data memory
dram_bufsize_in = 1536
dram_bufsize_out = 512
iter_i = 0
iter_j = 1
cur_noc_count = 0
cur_noc_count_in = 0
cur_noc_count_out = 0


def main():
    global arrayrows
    global arraycols
    global bufsize

    print(
        "Enabling %d columns with each having %d HDIFF cores"
        % (arraycols, broadcast_cores)
    )

    rows = arrayrows  # row 0 is reserved
    cols = arraycols

    f = open("aie.mlir", "w+")
    # declare tile, column by row

    f.write(
        """//===- aie.mlir ------------------------------------------------*- MLIR -*-===//  
//  
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
// 
//
//===----------------------------------------------------------------------===//\n\n\n"""
    )

    f.write("module @hdiff_large_%d {\n" % (arraycols * broadcast_cores))

    def noc_div_two_channel(core_count):
        print(core_count)
        seg = 2
        if core_count < seg:
            val = noc_columns[0]
        elif core_count >= seg and core_count < 2 * seg:
            val = noc_columns[1]
        elif core_count >= 2 * seg and core_count < 3 * seg:
            val = noc_columns[2]
        elif core_count >= 3 * seg and core_count < 4 * seg:
            val = noc_columns[3]
        elif core_count >= 4 * seg and core_count < 5 * seg:
            val = noc_columns[4]
        elif core_count >= 5 * seg and core_count < 6 * seg:
            val = noc_columns[5]
        elif core_count >= 6 * seg and core_count < 7 * seg:
            val = noc_columns[6]
        elif core_count >= 7 * seg and core_count < 8 * seg:
            val = noc_columns[7]
        elif core_count >= 8 * seg and core_count < 9 * seg:
            val = noc_columns[8]
        elif core_count >= 9 * seg and core_count < 10 * seg:
            val = noc_columns[9]
        elif core_count >= 10 * seg and core_count < 11 * seg:
            val = noc_columns[10]
        elif core_count >= 11 * seg and core_count < 12 * seg:
            val = noc_columns[11]
        elif core_count >= 12 * seg and core_count < 13 * seg:
            val = noc_columns[12]
        elif core_count >= 13 * seg and core_count < 14 * seg:
            val = noc_columns[13]
        elif core_count >= 14 * seg and core_count < 15 * seg:
            val = noc_columns[14]
        elif core_count >= 15 * seg:
            val = noc_columns[15]
        return val

    # setting core tiles

    for col in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        f.write("//---col %d---*-\n" % col)
        for row in range(startrow, startrow + broadcast_cores + arrayrows):  #
            f.write("  %%tile%d_%d = AIE.tile(%d, %d)\n" % (col, row, col, row))
    f.write("\n")

    # setting NOC tiles
    noc_count = 0
    for col in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        for row in range(startrow, startrow + broadcast_cores + arrayrows):  #
            if noc_count % 2 == 0:
                shim_place = noc_div_two_channel(noc_count)
                f.write("//---NOC TILE %d---*-\n" % shim_place)
                f.write(
                    "  %%tile%d_%d = AIE.tile(%d, %d)\n"
                    % (shim_place, 0, shim_place, 0)
                )
            noc_count = noc_count + 1

    f.write("\n")

    # def noc_place_min(cur, noc_i, noc_j):
    #     global iter_i
    #     global iter_j
    #     # print("curre={}, noc_i{} noc_j{}".format(cur, noc_i, noc_j))
    #     # print("iter_j{}".format(iter_j))
    #     if cur>47:
    #         return 47
    #     if cur in [12,13,14]:
    #         return 11
    #     if cur in [20,21,22]:
    #         return 19
    #     elif cur>noc_i and iter_j<len(noc_columns)-1:
    #         iter_i=iter_i+1;
    #         iter_j=iter_j+1
    #         return(noc_place_min(cur,noc_columns[iter_i],noc_columns[iter_j]))
    #     else:
    #       return  noc_i

    # for cur in range (0,50):
    #     global iter_i
    #     global iter_j
    #     iter_i=0
    #     iter_j=1
    #     val=noc_place_min(cur,noc_columns[0],noc_columns[1])
    #     # val=noc_place_cust(cur)
    #     print("{}: NOC {}, Dist={}".format(cur, val,abs(val-cur)))
    # row and column to generate for.  lastrow indicates that we shift one column to the right
    # for the next core.

    def gagan_gen_buffer(col):
        global cur_noc_count
        symbol = "obj_in_%d" % (col)
        # bb=("%%tile%d_%d" %(col, row))
        shim_place = noc_div_two_channel(cur_noc_count)
        bb = ""
        for b_tile in range(startrow, startrow + broadcast_cores + 1):
            bb = bb + ("%%tile%d_%d," % (col, b_tile))
            cur_noc_count = cur_noc_count + 1
        # print(bb)
        bb_sym = bb[:-1]
        print("shim={}".format(shim_place))
        f.write(
            '  %%buf_in_%d_shim_%d = AIE.objectfifo.createObjectFifo(%%tile%d_0,{%s},6) { sym_name = "%s" } : !AIE.objectfifo<memref<%dxi32>>\n'
            % (col, shim_place, shim_place, bb_sym, symbol, bufsize)
        )
        for b_tile in range(startrow, startrow + broadcast_cores + 1):
            symbol = "obj_out_%d_%d" % (col, b_tile)
            f.write(
                '  %%buf_out_%d_%d_shim_%d = AIE.objectfifo.createObjectFifo(%%tile%d_%d,{%%tile%d_0},2) { sym_name = "%s" } : !AIE.objectfifo<memref<%dxi32>>\n'
                % (col, b_tile, shim_place, col, b_tile, shim_place, symbol, bufsize)
            )
        f.write("\n")

        if col == startcol + arraycols - 1:
            print("seeting cur_zero")
            cur_noc_count = 0

    for i in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        gagan_gen_buffer(i)

    def gagan_gen_ddr(col):
        # print(bb)
        f.write(
            '  %%ext_buffer_in_%d = AIE.external_buffer  {sym_name = "ddr_buffer_in_%d"}: memref<%d x i32>\n'
            % (col, col, dram_bufsize_in)
        )
        for b_tile in range(startrow, startrow + broadcast_cores + 1):
            f.write(
                '  %%ext_buffer_out_%d_%d = AIE.external_buffer  {sym_name = "ddr_buffer_out_%d_%d"}: memref<%d x i32>\n'
                % (col, b_tile, col, b_tile, dram_bufsize_out)
            )
        f.write("\n")

    for i in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        gagan_gen_ddr(i)

    def gagan_reg_buffer(col, row):
        global cur_noc_count
        if col == startcol and row == 1:
            print("making zero")
            cur_noc_count = 0
        shim_place = noc_div_two_channel(cur_noc_count)
        print(shim_place)
        f.write(
            "  AIE.objectfifo.register_external_buffers(%%tile%d_0, %%buf_in_%d_shim_%d  : !AIE.objectfifo<memref<%dxi32>>, {%%ext_buffer_in_%d}) : (memref<%dxi32>)\n"
            % (shim_place, col, shim_place, bufsize, col, dram_bufsize_in)
        )
        for b_tile in range(startrow, startrow + broadcast_cores + 1):
            f.write(
                "  AIE.objectfifo.register_external_buffers(%%tile%d_0, %%buf_out_%d_%d_shim_%d  : !AIE.objectfifo<memref<%dxi32>>, {%%ext_buffer_out_%d_%d}) : (memref<%dxi32>)\n"
                % (
                    shim_place,
                    col,
                    b_tile,
                    shim_place,
                    bufsize,
                    col,
                    b_tile,
                    dram_bufsize_out,
                )
            )
            cur_noc_count = cur_noc_count + 1
        f.write("\n")

    f.write("//Registering buffers\n")

    for i in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        for j in range(startrow, startrow + arrayrows):
            gagan_reg_buffer(i, j)

    f.write(
        "\n  func.func private @vec_hdiff(%A: memref<256xi32>,%B: memref<256xi32>, %C:  memref<256xi32>, %D: memref<256xi32>, %E:  memref<256xi32>,  %O: memref<256xi32>) -> ()\n"
    )

    f.write("\n")

    def gagan_gen_core(col, row, shim_place):
        f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" % (col, row, col, row))
        f.write("    %lb = arith.constant 0 : index\n")
        f.write("    %ub = arith.constant 2 : index\n")
        f.write("    %step = arith.constant 1 : index\n")
        f.write("    scf.for %iv = %lb to %ub step %step {  \n")
        f.write(
            "      %%obj_in_subview = AIE.objectfifo.acquire<Consume>(%%buf_in_%d_shim_%d: !AIE.objectfifo<memref<%dxi32>>, 5) : !AIE.objectfifosubview<memref<%dxi32>>\n"
            % (col, shim_place, bufsize, bufsize)
        )
        f.write(
            "      %%row0 = AIE.objectfifo.subview.access %%obj_in_subview[0] : !AIE.objectfifosubview<memref<%dxi32>> -> memref<%dxi32>\n"
            % (bufsize, bufsize)
        )
        f.write(
            "      %%row1 = AIE.objectfifo.subview.access %%obj_in_subview[1] : !AIE.objectfifosubview<memref<%dxi32>> -> memref<%dxi32>\n"
            % (bufsize, bufsize)
        )
        f.write(
            "      %%row2 = AIE.objectfifo.subview.access %%obj_in_subview[2] : !AIE.objectfifosubview<memref<%dxi32>> -> memref<%dxi32>\n"
            % (bufsize, bufsize)
        )
        f.write(
            "      %%row3 = AIE.objectfifo.subview.access %%obj_in_subview[3] : !AIE.objectfifosubview<memref<%dxi32>> -> memref<%dxi32>\n"
            % (bufsize, bufsize)
        )
        f.write(
            "      %%row4 = AIE.objectfifo.subview.access %%obj_in_subview[4] : !AIE.objectfifosubview<memref<%dxi32>> -> memref<%dxi32>\n"
            % (bufsize, bufsize)
        )

        f.write(
            "      %%obj_out_subview = AIE.objectfifo.acquire<Produce>(%%buf_out_%d_%d_shim_%d: !AIE.objectfifo<memref<%dxi32>>, 5) : !AIE.objectfifosubview<memref<%dxi32>>\n"
            % (col, row, shim_place, bufsize, bufsize)
        )
        f.write(
            "      %obj_out = AIE.objectfifo.subview.access %obj_out_subview[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>\n"
        )
        f.write(
            "      func.call @vec_hdiff(%row0,%row1,%row2,%row3,%row4,%obj_out) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>) -> ()\n"
        )
        f.write(
            "      AIE.objectfifo.release<Consume>(%%buf_in_%d_shim_%d: !AIE.objectfifo<memref<%dxi32>>, 1)\n"
            % (col, shim_place, bufsize)
        )
        f.write(
            "      AIE.objectfifo.release<Produce>(%%buf_out_%d_%d_shim_%d: !AIE.objectfifo<memref<%dxi32>>, 1)\n"
            % (col, row, shim_place, bufsize)
        )
        f.write("  }\n\n")
        f.write(
            "  AIE.objectfifo.release<Consume>(%%buf_in_%d_shim_%d: !AIE.objectfifo<memref<%dxi32>>, 4)\n"
            % (col, shim_place, bufsize)
        )
        f.write("  AIE.end\n")
        f.write(' } { link_with="hdiff.o" }\n\n')

    cur_count = 0
    for i in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        for j in range(startrow, startrow + broadcast_cores + 1):
            shim_place = noc_div_two_channel(cur_count)
            gagan_gen_core(i, j, shim_place)
            cur_count = cur_count + 1

    # for i in range (0, 64): # col 0 is reserved in aie
    #     f.write("mlir_aie_sync_mem_dev(_xaie, %d);\n "%(i) )

    # for i in range (0, 32): # col 0 is reserved in aie
    #     f.write("mlir_aie_external_set_addr_ddr_buffer_in_%d((u64)ddr_ptr_in_%d); \n "%(i,i) )

    # for i in range (0, 32): # col 0 is reserved in aie
    #     f.write("mlir_aie_external_set_addr_ddr_buffer_out_%d_2((u64)ddr_ptr_out_%d_2); \n "%(i,i) )

    # for i in range (32,64): # col 0 is reserved in aie
    #     f.write("mlir_aie_sync_mem_cpu(_xaie, %d); //// only used in libaiev2 //sync up with output\n "%(i) )

    f.write("}\n")


if __name__ == "__main__":
    main()
