from re import L
from struct import pack
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
import math

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

from aie.dialects import memref

from aie.dialects._aie_ops_gen import buffer as buffer_raw
from aie.helpers.util import try_convert_np_type_to_mlir_type
import numpy as np
import sys
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from enum import IntEnum
from aie.extras.dialects.ext.arith import constant, index_cast

from aie.ir import *
from aie.ir import MemRefType, IndexType
from aie.dialects import arith, memref
from aie.dialects.memref import AllocaScopeOp

from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.dialects.memref import alloc, store, alloca
from aie.extras import types as T

from aie.dialects.aiex import control_packet

import os
import json
import json

from aie._mlir_libs._mlir.ir import Attribute
from aie.dialects._aiex_ops_gen import _Dialect

from helper_func import generate_packet_attribute, custom_ceil

def setup_CT_0_2(in_data_ty, out_data_ty, control_package_ty,
                 buffer_size:int, total_size:int):

    ComputeTile_0_2 = tile(0,2, allocation_scheme="basic-sequential")
    ComputeTile_0_2.attributes["controlled_id"] = generate_packet_attribute(0, 26)    
    #NOTE: mem_bank flag seem not working anymore after Tile() is configure to basic-sequential address mode
    offset = 1024 # reserve for stack
    assert offset %64 == 0
    in_buffer = [
        buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(in_data_ty), sym_name=f"in_buffer_{0}", address=offset),

    ]
    in_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=8, init=2, sym_name="in_buffer_p_lock")
    in_buffer_con_lock = lock(ComputeTile_0_2, lock_id=9, init=0, sym_name="in_buffer_c_lock")
    offset+= buffer_size*2*4
    
    offset = custom_ceil(offset, 64)
    assert offset %64 == 0


    # out_buffer_address = (64*1024) - (buffer_size_of_out_ping_pong*2*4) # 4 byte per float
    out_buffer = [
        buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(out_data_ty), sym_name=f"out_buffer_{0}", address=offset ), # 
    ]        
    out_buffer_prod_lock = lock(ComputeTile_0_2, lock_id=10, init=2)
    out_buffer_con_lock = lock(ComputeTile_0_2, lock_id=11, init=0)

    assert offset+buffer_size*2*4 <= (64*1024)  # total of less than 64kB
    offset += buffer_size*2*4
    
    CT2_control_out_prod_lock = lock(ComputeTile_0_2, lock_id=0, init=1, sym_name="CT2_control_out_prod_lock")
    CT2_control_out_con_lock = lock(ComputeTile_0_2, lock_id=1, init=0, sym_name="CT2_control_out_con_lock")
    CT2_control_out_buffer = [
        buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(control_package_ty), sym_name="CT2_control_out_buffer", address=offset)
    ]
    
    offset += 16*4
    CT2_control_in_buffer = [
        buffer_raw(tile=ComputeTile_0_2, buffer=try_convert_np_type_to_mlir_type(control_package_ty), sym_name="CT2_control_in_buffer", address=offset)
    ]
    CT2_control_in_prod_lock = lock(ComputeTile_0_2, lock_id=2, init=1, sym_name="CT2_control_in_prod_lock")
    CT2_control_in_con_lock = lock(ComputeTile_0_2, lock_id=3, init=0, sym_name="CT2_control_in_con_lock")
    

    
    @mem(ComputeTile_0_2)
    def m(block):
    
        s0 = dma_start( DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3]) # input DMA0
        
        with block[1]:
            use_lock(in_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(in_buffer[0], offset=0, len=buffer_size)
            use_lock(in_buffer_con_lock, LockAction.Release, value=1)
            next_bd(block[2])
        with block[2]:
            use_lock(in_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(in_buffer[0], offset=buffer_size, len=buffer_size)
            use_lock(in_buffer_con_lock, LockAction.Release, value=1)
            next_bd(block[1])
            
        with block[3]:
            s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
        with block[4]:
            use_lock(out_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(out_buffer[0], offset=0, len=buffer_size, packet=(0,9)) # error to be fixed through control_packet
            # purposefully leave the error, fix by control packet in kernel
            use_lock(out_buffer_prod_lock, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            use_lock(out_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(out_buffer[0], offset=buffer_size, len=buffer_size, packet=(0,9))
            use_lock(out_buffer_prod_lock, LockAction.Release, value=1)
            next_bd(block[4])
        with block[6]:
            s2 = dma_start(DMAChannelDir.MM2S, 1, dest=block[7], chain=block[8])
        with block[7]:
            use_lock(CT2_control_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT2_control_out_buffer[0], offset=0, len=1, packet=(0,13)) # for now as read packet?
            use_lock(CT2_control_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[7])
        with block[8]:
            s3 = dma_start(DMAChannelDir.S2MM, 1, dest=block[9], chain=block[10] )
        with block[9]:
            use_lock(CT2_control_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT2_control_in_buffer[0], offset=0, len=1)
            use_lock(CT2_control_in_con_lock, LockAction.Release, value=1)
            next_bd(block[9])
        with block[10]:
            EndOp()
                            

        
    return ComputeTile_0_2, in_buffer, out_buffer, CT2_control_out_buffer, CT2_control_in_buffer
    