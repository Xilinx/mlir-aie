import numpy as np

import aie.extras.types as T
from aie.dialects import builtin
from aie.dialects.transform import any_op_t
from aie.dialects.transform.extras import named_sequence, apply_patterns
from aie.extras.util import find_ops
from aie.ir import StringAttr, UnitAttr

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
import aie.extras.dialects.ext.memref
from aie.extras.context import RAIIMLIRContext, ExplicitlyManagedModule
from aie.extras.dialects.ext.bufferization import LayoutMapOption
from aie.dialects.transform.vector import (
    VectorContractLowering,
    VectorMultiReductionLowering,
    VectorTransferSplit,
    VectorTransposeLowering,
)
from aie.extras.dialects.ext import linalg
from aie.extras.dialects.ext.func import func
from aie.extras.dialects.ext.transform import (
    match,
    tile_to_scf_for,
    get_parent_op,
    transform_any_op_t,
)
from aie.extras.dialects.ext import transform
from aie.extras.runtime.passes import Pipeline, run_pipeline
from aie.extras.runtime.refbackend import LLVMJITBackend
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    device,
    generate_bcf,
    generate_cdo,
    ipu_instgen,
    mem,
    memtile_dma,
    tile,
    translate_mlir_to_llvmir,
    translate_aie_vec_to_cpp,
    dma,
    another_bd,
)

ctx = RAIIMLIRContext()
backend = LLVMJITBackend()
module = ExplicitlyManagedModule()

M, K, N = 2, 4, 6


@func
def matmul_tensors(
    A: T.tensor(M, K, T.f32()),
    B: T.tensor(K, N, T.f32()),
    C: T.tensor(M, N, T.f32()),
):
    return linalg.matmul(A, B, C)


@builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
def payload():
    matmul_tensors.emit(force=True)


@builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
def mod_transform():
    @named_sequence("main", [any_op_t()], [])
    def main(module_op: any_op_t()):
        matmul = match(module_op, ops=["linalg.matmul"])
        tiled_matmul, (_, _, inner_loop) = tile_to_scf_for(matmul, sizes=[2, 2, 2])
        transform.structured.vectorize_children_and_apply_patterns(
            get_parent_op(transform_any_op_t(), tiled_matmul, isolated_from_above=True)
        )
        new_mod = transform.bufferization.one_shot_bufferize(
            module_op,
            function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
            bufferize_function_boundaries=True,
        )

        func_op = match(new_mod, ops=["func.func"])

        @apply_patterns(func_op)
        def pats():
            transform.apply_patterns.vector.lower_contraction(
                lowering_strategy=VectorContractLowering.OuterProduct
            )
            transform.apply_patterns.vector.transfer_permutation_patterns()
            transform.apply_patterns.vector.lower_multi_reduction(
                lowering_strategy=VectorMultiReductionLowering.InnerParallel
            )
            transform.apply_patterns.vector.split_transfer_full_partial(
                split_transfer_strategy=VectorTransferSplit.LinalgCopy
            )
            transform.apply_patterns.vector.transfer_to_scf(
                max_transfer_rank=1, full_unroll=True
            )
            transform.apply_patterns.vector.lower_transfer(max_transfer_rank=1)
            transform.apply_patterns.vector.lower_shape_cast()
            transform.apply_patterns.vector.lower_transpose(
                lowering_strategy=VectorTransposeLowering.Shuffle1D
            )


module = module.finish()
# print(module)

vectorized_module = run_pipeline(
    module,
    pipeline=Pipeline()
    .transform_interpreter(entry_point="main", debug_payload_root_tag="payload")
    .add_pass("convert-vector-to-aievec", **{"aie-target": "aieml"})
    .lower_affine(),
)

mod = find_ops(
    vectorized_module.operation,
    lambda x: "transform.target_tag" in x.attributes
    and x.attributes["transform.target_tag"].value == "payload",
    single=True,
)
print(mod)
