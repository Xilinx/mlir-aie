# These Python bindings are to be merged in #1699 in the future
from typing import Optional, Union
from aie.dialects.aie import *

def get_dma_region_decorator(op_obj_constructor):
    def decorator(f):
        f_sig = inspect.signature(f)
        op = op_obj_constructor()
        entry_block = op.body.blocks.append()
        bds_ctx = bds(op)
        with InsertionPoint(entry_block):
            with bds_ctx as bd:
                if len(f_sig.parameters) == 0:
                    f()
                elif len(f_sig.parameters) == 1:
                    f(bd)
                else:
                    raise RuntimeError(
                        "Expected function to take zero or one argument(s)."
                    )
        return op

    return decorator


def mem(tile):
    return get_dma_region_decorator(lambda: MemOp(T.index(), tile))


def shim_mem(tile):
    return get_dma_region_decorator(lambda: ShimDMAOp(T.index(), tile))


def memtile_dma(tile):
    return get_dma_region_decorator(lambda: MemTileDMAOp(T.index(), tile))


def dma_start(
    channel_dir,
    channel_index,
    *,
    dest: Optional[Union[Successor, Block, ContextManagedBlock]] = None,
    chain: Optional[Union[Successor, Block, ContextManagedBlock]] = None,
    loc=None,
    ip=None,
):
    chain_block = chain.block if isinstance(chain, ContextManagedBlock) else chain
    dest_block = dest.block if isinstance(dest, ContextManagedBlock) else dest
    op = DMAStartOp(
        channel_dir, channel_index, dest=dest_block, chain=chain_block, loc=loc, ip=ip
    )
    return op.dest, op.chain