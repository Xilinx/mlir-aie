import logging
import os
import sys
import tempfile
from contextlib import ExitStack
from io import StringIO
from typing import Optional, Union

from .util import disable_multithreading
from ..ir import StringAttr, Module
from ..passmanager import PassManager

logger = logging.getLogger(__name__)


class MlirCompilerError(Exception):
    pass


def get_module_name_for_debug_dump(module):
    if "debug_module_name" not in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["debug_module_name"]).value


def run_pipeline(
    module,
    pipeline: Union[str, "Pipeline"],
    description: Optional[str] = None,
    enable_ir_printing=False,
    print_pipeline=False,
    verify=True,
) -> Module:
    module = Module.parse(str(module))

    if isinstance(pipeline, Pipeline):
        pipeline = str(pipeline)
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        # Lower module in place to make it ready for compiler backends.
        with ExitStack() as stack:
            stack.enter_context(module.context)
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10,
                enable_debug_info=True,
            )
            pm = PassManager.parse(pipeline)
            pm.enable_verifier(verify)
            if print_pipeline:
                print(pm)
            if enable_ir_printing:
                stack.enter_context(disable_multithreading())
                pm.enable_ir_printing()

            pm.run(module.operation)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:

            {'*' * 80}
            {sys.stderr.getvalue().strip()}
            {'*' * 80}

            For developers, the error can be reproduced with:
            $ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise MlirCompilerError(trimmed_message)
    finally:
        sys.stderr = original_stderr

    return module


class Pipeline:
    _pipeline: list[str] = []

    def __init__(self, pipeline=None):
        if pipeline is None:
            pipeline = []
        self._pipeline = pipeline

    def Func(self, p: "Pipeline"):
        assert isinstance(p, Pipeline)
        self._pipeline.append(f"func.func({p.materialize(module=False)})")
        return self

    def Spirv(self, p: "Pipeline"):
        assert isinstance(p, Pipeline)
        self._pipeline.append(f"spirv.module({p.materialize(module=False)})")
        return self

    def Gpu(self, p: "Pipeline"):
        assert isinstance(p, Pipeline)
        self._pipeline.append(f"gpu.module({p.materialize(module=False)})")
        return self

    def materialize(self, module=True):
        pipeline_str = ",".join(self._pipeline)
        if module:
            pipeline_str = f"builtin.module({pipeline_str})"
        logger.debug(f"{pipeline_str}")
        return pipeline_str

    def __str__(self):
        return self.materialize()

    def add_pass(self, pass_name, **kwargs):
        kwargs = {
            k.replace("_", "-"): int(v) if isinstance(v, bool) else v
            for k, v in kwargs.items()
            if v is not None
        }
        if kwargs:
            args_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            pass_str = f"{pass_name}{{ {args_str} }}"
        else:
            pass_str = f"{pass_name}"
        self._pipeline.append(pass_str)
        return self

    def lower_to_llvm_(self):
        return any(["to-llvm" in p for p in self._pipeline])

    def bufferize(self):
        return (
            self.Func(scf_bufferize().empty_tensor_to_alloc_tensor().linalg_bufferize())
            .func_bufferize()
            .arith_bufferize()
            .Func(tensor_bufferize().finalizing_bufferize().buffer_deallocation())
        )

    def lower_to_llvm(self):
        return (
            self.cse()
            .Func(lower_affine().arith_expand().convert_math_to_llvm())
            .convert_math_to_libm()
            .expand_strided_metadata()
            .finalize_memref_to_llvm()
            .convert_scf_to_cf()
            .convert_cf_to_llvm()
            .cse()
            .lower_affine()
            .Func(convert_arith_to_llvm())
            .convert_func_to_llvm()
            .canonicalize()
            .convert_openmp_to_llvm()
            .cse()
            .reconcile_unrealized_casts()
        )

    def lower_to_openmp(self):
        return self.convert_scf_to_openmp().Func(lower_affine())

    def sparse_compiler(
        self,
        parallelization_strategy=None,
        enable_runtime_library=None,
        enable_buffer_initialization=None,
        vl=None,
        s2s_strategy=None,
        reassociate_fp_reductions=None,
        enable_index_optimizations=None,
        enable_amx=None,
        enable_arm_neon=None,
        enable_arm_sve=None,
        enable_x86vector=None,
    ):
        self.add_pass(
            "sparse-compiler",
            parallelization_strategy=parallelization_strategy,
            enable_runtime_library=enable_runtime_library,
            enable_buffer_initialization=enable_buffer_initialization,
            vl=vl,
            s2s_strategy=s2s_strategy,
            reassociate_fp_reductions=reassociate_fp_reductions,
            enable_index_optimizations=enable_index_optimizations,
            enable_amx=enable_amx,
            enable_arm_neon=enable_arm_neon,
            enable_arm_sve=enable_arm_sve,
            enable_x86vector=enable_x86vector,
        )
        return self

    def lower_to_vulkan(self, index_bitwidth=None):
        return (
            self.gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .convert_gpu_to_spirv()
            .Spirv(spirv_lower_abi_attrs().spirv_update_vce())
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .Func(llvm_request_c_wrappers())
            .convert_func_to_llvm(index_bitwidth=index_bitwidth)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan()
        )

    def transform_dialect_erase_schedule(self):
        return self.add_pass("test-transform-dialect-erase-schedule")

    def transform_dialect_interpreter(
        self,
        bind_first_extra_to_ops=None,
        bind_first_extra_to_params=None,
        bind_first_extra_to_results_of_ops=None,
        bind_second_extra_to_ops=None,
        bind_second_extra_to_params=None,
        bind_second_extra_to_results_of_ops=None,
        debug_payload_root_tag=None,
        debug_transform_root_tag=None,
        enable_expensive_checks=None,
        transform_file_name=None,
        test_module_generation=None,
    ):
        if bind_first_extra_to_ops is not None and isinstance(
            bind_first_extra_to_ops, (list, tuple)
        ):
            bind_first_extra_to_ops = ",".join(map(str, bind_first_extra_to_ops))
        if bind_first_extra_to_params is not None and isinstance(
            bind_first_extra_to_params, (list, tuple)
        ):
            bind_first_extra_to_params = ",".join(map(str, bind_first_extra_to_params))
        if bind_first_extra_to_results_of_ops is not None and isinstance(
            bind_first_extra_to_results_of_ops, (list, tuple)
        ):
            bind_first_extra_to_results_of_ops = ",".join(
                map(str, bind_first_extra_to_results_of_ops)
            )
        if bind_second_extra_to_ops is not None and isinstance(
            bind_second_extra_to_ops, (list, tuple)
        ):
            bind_second_extra_to_ops = ",".join(map(str, bind_second_extra_to_ops))
        if bind_second_extra_to_params is not None and isinstance(
            bind_second_extra_to_params, (list, tuple)
        ):
            bind_second_extra_to_params = ",".join(
                map(str, bind_second_extra_to_params)
            )
        if bind_second_extra_to_results_of_ops is not None and isinstance(
            bind_second_extra_to_results_of_ops, (list, tuple)
        ):
            bind_second_extra_to_results_of_ops = ",".join(
                map(str, bind_second_extra_to_results_of_ops)
            )
        if debug_payload_root_tag is not None and isinstance(
            debug_payload_root_tag, (list, tuple)
        ):
            debug_payload_root_tag = ",".join(map(str, debug_payload_root_tag))
        if debug_transform_root_tag is not None and isinstance(
            debug_transform_root_tag, (list, tuple)
        ):
            debug_transform_root_tag = ",".join(map(str, debug_transform_root_tag))
        if transform_file_name is not None and isinstance(
            transform_file_name, (list, tuple)
        ):
            transform_file_name = ",".join(map(str, transform_file_name))
        if test_module_generation is not None and isinstance(
            test_module_generation, (list, tuple)
        ):
            test_module_generation = ",".join(map(str, test_module_generation))
        return self.add_pass(
            "test-transform-dialect-interpreter",
            bind_first_extra_to_ops=bind_first_extra_to_ops,
            bind_first_extra_to_params=bind_first_extra_to_params,
            bind_first_extra_to_results_of_ops=bind_first_extra_to_results_of_ops,
            bind_second_extra_to_ops=bind_second_extra_to_ops,
            bind_second_extra_to_params=bind_second_extra_to_params,
            bind_second_extra_to_results_of_ops=bind_second_extra_to_results_of_ops,
            debug_payload_root_tag=debug_payload_root_tag,
            debug_transform_root_tag=debug_transform_root_tag,
            enable_expensive_checks=enable_expensive_checks,
            transform_file_name=transform_file_name,
            test_module_generation=test_module_generation,
        )

    ############################
    # autogen starts
    ############################

    def affine_data_copy_generate(
        self,
        fast_mem_capacity: int = None,
        fast_mem_space: int = None,
        generate_dma: bool = None,
        min_dma_transfer: int = None,
        slow_mem_space: int = None,
        skip_non_unit_stride_loops: bool = None,
        tag_mem_space: int = None,
    ):
        """Generate explicit copying for affine memory operations
        Args:
            fast-mem-capacity: Set fast memory space capacity in KiB (default: unlimited)
            fast-mem-space: Fast memory space identifier for copy generation (default: 1)
            generate-dma: Generate DMA instead of point-wise copy
            min-dma-transfer: Minimum DMA transfer size supported by the target in bytes
            slow-mem-space: Slow memory space identifier for copy generation (default: 0)
            skip-non-unit-stride-loops: Testing purposes: avoid non-unit stride loop choice depths for copy placement
            tag-mem-space: Tag memory space identifier for copy generation (default: 0)
        """
        self.add_pass(
            "affine-data-copy-generate",
            fast_mem_capacity=fast_mem_capacity,
            fast_mem_space=fast_mem_space,
            generate_dma=generate_dma,
            min_dma_transfer=min_dma_transfer,
            slow_mem_space=slow_mem_space,
            skip_non_unit_stride_loops=skip_non_unit_stride_loops,
            tag_mem_space=tag_mem_space,
        )
        return self

    def affine_expand_index_ops(self):
        """Lower affine operations operating on indices into more fundamental operations"""
        self.add_pass("affine-expand-index-ops")
        return self

    def affine_loop_coalescing(self):
        """Coalesce nested loops with independent bounds into a single loop"""
        self.add_pass("affine-loop-coalescing")
        return self

    def affine_loop_fusion(
        self,
        fusion_compute_tolerance: float = None,
        fusion_fast_mem_space: int = None,
        fusion_local_buf_threshold: int = None,
        fusion_maximal: bool = None,
        mode: "FusionMode" = None,
    ):
        """Fuse affine loop nests

        This pass performs fusion of loop nests using a slicing-based approach. The
        transformation works on an MLIR `Block` granularity and applies to all
        blocks of the pass is run on. It combines two fusion strategies:
        producer-consumer fusion and sibling fusion. Producer-consumer fusion is
        aimed at fusing pairs of loops where the first one writes to a memref that
        the second reads. Sibling fusion targets pairs of loops that share no
        dependences between them but that load from the same memref. The fused loop
        nests, when possible, are rewritten to access significantly smaller local
        buffers instead of the original memref's, and the latter are often either
        completely optimized away or contracted. This transformation leads to
        enhanced locality and lower memory footprint through the elimination or
        contraction of temporaries/intermediate memref's. These benefits are
        sometimes achieved at the expense of redundant computation through a cost
        model that evaluates available choices such as the depth at which a source
        slice should be materialized in the designation slice.

        Example 1: Producer-consumer fusion.
        Input:
        ```mlir
        func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
          %0 = memref.alloc() : memref<10xf32>
          %1 = memref.alloc() : memref<10xf32>
          %cst = arith.constant 0.000000e+00 : f32
          affine.for %arg2 = 0 to 10 {
            affine.store %cst, %0[%arg2] : memref<10xf32>
            affine.store %cst, %1[%arg2] : memref<10xf32>
          }
          affine.for %arg2 = 0 to 10 {
            %2 = affine.load %0[%arg2] : memref<10xf32>
            %3 = arith.addf %2, %2 : f32
            affine.store %3, %arg0[%arg2] : memref<10xf32>
          }
          affine.for %arg2 = 0 to 10 {
            %2 = affine.load %1[%arg2] : memref<10xf32>
            %3 = arith.mulf %2, %2 : f32
            affine.store %3, %arg1[%arg2] : memref<10xf32>
          }
          return
        }
        ```
        Output:
        ```mlir
        func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
          %0 = memref.alloc() : memref<1xf32>
          %1 = memref.alloc() : memref<1xf32>
          %cst = arith.constant 0.000000e+00 : f32
          affine.for %arg2 = 0 to 10 {
            affine.store %cst, %0[0] : memref<1xf32>
            affine.store %cst, %1[0] : memref<1xf32>
            %2 = affine.load %1[0] : memref<1xf32>
            %3 = arith.mulf %2, %2 : f32
            affine.store %3, %arg1[%arg2] : memref<10xf32>
            %4 = affine.load %0[0] : memref<1xf32>
            %5 = arith.addf %4, %4 : f32
            affine.store %5, %arg0[%arg2] : memref<10xf32>
          }
          return
        }
        ```

        Example 2: Sibling fusion.
        Input:
        ```mlir
        func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                             %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                             %arg4: memref<10x10xf32>) {
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
              %2 = arith.mulf %0, %1 : f32
              affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
            }
          }
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
              %2 = arith.addf %0, %1 : f32
              affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
            }
          }
          return
        }
        ```
        Output:
        ```mlir
        func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                             %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                             %arg4: memref<10x10xf32>) {
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
              %2 = arith.mulf %0, %1 : f32
              affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
              %3 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %4 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %arg4[%arg5, %arg6] : memref<10x10xf32>
            }
          }
          return
        }
        ```

        Args:
            fusion-compute-tolerance: Fractional increase in additional computation tolerated while fusing
            fusion-fast-mem-space: Faster memory space number to promote fusion buffers to
            fusion-local-buf-threshold: Threshold size (KiB) for promoting local buffers to fast memory space
            fusion-maximal: Enables maximal loop fusion
            mode: fusion mode to attempt
        """
        self.add_pass(
            "affine-loop-fusion",
            fusion_compute_tolerance=fusion_compute_tolerance,
            fusion_fast_mem_space=fusion_fast_mem_space,
            fusion_local_buf_threshold=fusion_local_buf_threshold,
            fusion_maximal=fusion_maximal,
            mode=mode,
        )
        return self

    def affine_loop_invariant_code_motion(self):
        """Hoist loop invariant instructions outside of affine loops"""
        self.add_pass("affine-loop-invariant-code-motion")
        return self

    def affine_loop_normalize(self, promote_single_iter: bool = None):
        """Apply normalization transformations to affine loop-like ops
        Args:
            promote-single-iter: Promote single iteration loops
        """
        self.add_pass("affine-loop-normalize", promote_single_iter=promote_single_iter)
        return self

    def affine_loop_tile(
        self,
        cache_size: int = None,
        separate: bool = None,
        tile_size: int = None,
        tile_sizes: list[int] = None,
    ):
        """Tile affine loop nests
        Args:
            cache-size: Set size of cache to tile for in KiB (default: 512)
            separate: Separate full and partial tiles (default: false)
            tile-size: Use this tile size for all loops
            tile-sizes: List of tile sizes for each perfect nest (overridden by -tile-size)
        """
        if tile_sizes is not None and isinstance(tile_sizes, (list, tuple)):
            tile_sizes = ",".join(map(str, tile_sizes))
        self.add_pass(
            "affine-loop-tile",
            cache_size=cache_size,
            separate=separate,
            tile_size=tile_size,
            tile_sizes=tile_sizes,
        )
        return self

    def affine_loop_unroll(
        self,
        unroll_factor: int = None,
        unroll_up_to_factor: bool = None,
        unroll_full: bool = None,
        unroll_num_reps: int = None,
        unroll_full_threshold: int = None,
        cleanup_unroll: bool = None,
    ):
        """Unroll affine loops
        Args:
            unroll-factor: Use this unroll factor for all loops being unrolled
            unroll-up-to-factor: Allow unrolling up to the factor specified
            unroll-full: Fully unroll loops
            unroll-num-reps: Unroll innermost loops repeatedly this many times
            unroll-full-threshold: Unroll all loops with trip count less than or equal to this
            cleanup-unroll: Fully unroll the cleanup loop when possible.
        """
        self.add_pass(
            "affine-loop-unroll",
            unroll_factor=unroll_factor,
            unroll_up_to_factor=unroll_up_to_factor,
            unroll_full=unroll_full,
            unroll_num_reps=unroll_num_reps,
            unroll_full_threshold=unroll_full_threshold,
            cleanup_unroll=cleanup_unroll,
        )
        return self

    def affine_loop_unroll_jam(self, unroll_jam_factor: int = None):
        """Unroll and jam affine loops
        Args:
            unroll-jam-factor: Use this unroll jam factor for all loops (default 4)
        """
        self.add_pass("affine-loop-unroll-jam", unroll_jam_factor=unroll_jam_factor)
        return self

    def affine_parallelize(
        self, max_nested: int = None, parallel_reductions: bool = None
    ):
        """Convert affine.for ops into 1-D affine.parallel
        Args:
            max-nested: Maximum number of nested parallel loops to produce. Defaults to unlimited (UINT_MAX).
            parallel-reductions: Whether to parallelize reduction loops. Defaults to false.
        """
        self.add_pass(
            "affine-parallelize",
            max_nested=max_nested,
            parallel_reductions=parallel_reductions,
        )
        return self

    def affine_pipeline_data_transfer(self):
        """Pipeline non-blocking data transfers between explicitly managed levels of the memory hierarchy

        This pass performs a transformation to overlap non-blocking DMA operations
        in a loop with computations through double buffering. This is achieved by
        advancing dma_start operations with respect to other operations.

        Input

        ```mlir
        func.func @pipelinedatatransfer() {
          %0 = memref.alloc() : memref<256xf32>
          %1 = memref.alloc() : memref<32xf32, 1>
          %2 = memref.alloc() : memref<1xf32>
          %c0 = arith.constant 0 : index
          %c128 = arith.constant 128 : index
          affine.for %i0 = 0 to 8 {
            affine.dma_start %0[%i0], %1[%i0], %2[%c0], %c128 : memref<256xf32>, memref<32xf32, 1>, memref<1xf32>
            affine.dma_wait %2[%c0], %c128 : memref<1xf32>
            %3 = affine.load %1[%i0] : memref<32xf32, 1>
            %4 = "compute"(%3) : (f32) -> f32
            affine.store %4, %1[%i0] : memref<32xf32, 1>
          }
          return
        }
        ```

        Output

        ```mlir
        module {
          func.func @pipelinedatatransfer() {
            %c8 = arith.constant 8 : index
            %c0 = arith.constant 0 : index
            %0 = memref.alloc() : memref<256xf32>
            %c0_0 = arith.constant 0 : index
            %c128 = arith.constant 128 : index
            %1 = memref.alloc() : memref<2x32xf32, 1>
            %2 = memref.alloc() : memref<2x1xf32>
            affine.dma_start %0[%c0], %1[%c0 mod 2, %c0], %2[%c0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
            affine.for %arg0 = 1 to 8 {
              affine.dma_start %0[%arg0], %1[%arg0 mod 2, %arg0], %2[%arg0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
              %8 = affine.apply #map3(%arg0)
              %9 = affine.apply #map4(%8)
              %10 = affine.apply #map4(%8)
              affine.dma_wait %2[%8 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
              %11 = affine.load %1[%8 mod 2, %8] : memref<2x32xf32, 1>
              %12 = "compute"(%11) : (f32) -> f32
              affine.store %12, %1[%8 mod 2, %8] : memref<2x32xf32, 1>
            }
            %3 = affine.apply #map3(%c8)
            %4 = affine.apply #map4(%3)
            %5 = affine.apply #map4(%3)
            affine.dma_wait %2[%3 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
            %6 = affine.load %1[%3 mod 2, %3] : memref<2x32xf32, 1>
            %7 = "compute"(%6) : (f32) -> f32
            affine.store %7, %1[%3 mod 2, %3] : memref<2x32xf32, 1>
            memref.dealloc %2 : memref<2x1xf32>
            memref.dealloc %1 : memref<2x32xf32, 1>
            return
          }
        }
        ```

        """
        self.add_pass("affine-pipeline-data-transfer")
        return self

    def affine_scalrep(self):
        """Replace affine memref accesses by scalars by forwarding stores to loads and eliminating redundant loads

        This pass performs store to load forwarding and redundant load elimination
        for affine memref accesses and potentially eliminates the entire memref
        if all its accesses are forwarded.

        Input

        ```mlir
        func.func @store_load_affine_apply() -> memref<10x10xf32> {
          %cf7 = arith.constant 7.0 : f32
          %m = memref.alloc() : memref<10x10xf32>
          affine.for %i0 = 0 to 10 {
            affine.for %i1 = 0 to 10 {
              affine.store %cf7, %m[%i0, %i1] : memref<10x10xf32>
              %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
              %v1 = arith.addf %v0, %v0 : f32
            }
          }
          return %m : memref<10x10xf32>
        }
        ```

        Output

        ```mlir
        module {
          func.func @store_load_affine_apply() -> memref<10x10xf32> {
            %cst = arith.constant 7.000000e+00 : f32
            %0 = memref.alloc() : memref<10x10xf32>
            affine.for %arg0 = 0 to 10 {
              affine.for %arg1 = 0 to 10 {
                affine.store %cst, %0[%arg0, %arg1] : memref<10x10xf32>
                %1 = arith.addf %cst, %cst : f32
              }
            }
            return %0 : memref<10x10xf32>
          }
        }
        ```

        """
        self.add_pass("affine-scalrep")
        return self

    def affine_simplify_structures(self):
        """Simplify affine expressions in maps/sets and normalize memrefs"""
        self.add_pass("affine-simplify-structures")
        return self

    def affine_super_vectorize(
        self,
        virtual_vector_size: list[int] = None,
        test_fastest_varying: list[int] = None,
        vectorize_reductions: bool = None,
    ):
        """Vectorize to a target independent n-D vector abstraction
        Args:
            virtual-vector-size: Specify an n-D virtual vector size for vectorization
            test-fastest-varying: Specify a 1-D, 2-D or 3-D pattern of fastest varying memory dimensions to match. See defaultPatterns in Vectorize.cpp for a description and examples. This is used for testing purposes
            vectorize-reductions: Vectorize known reductions expressed via iter_args. Switched off by default.
        """
        if virtual_vector_size is not None and isinstance(
            virtual_vector_size, (list, tuple)
        ):
            virtual_vector_size = ",".join(map(str, virtual_vector_size))
        if test_fastest_varying is not None and isinstance(
            test_fastest_varying, (list, tuple)
        ):
            test_fastest_varying = ",".join(map(str, test_fastest_varying))
        self.add_pass(
            "affine-super-vectorize",
            virtual_vector_size=virtual_vector_size,
            test_fastest_varying=test_fastest_varying,
            vectorize_reductions=vectorize_reductions,
        )
        return self

    def allocate_arm_sme_tiles(self):
        """Allocate SME tiles

        This pass does tile allocation for SME "virtual tiles". It is run at the
        'func.func' op level, replacing 'arm_sme.get_tile_id' ops with (i32) tile
        ids. An error will be emitted when there's no tiles left.

        """
        self.add_pass("allocate-arm-sme-tiles")
        return self

    def amdgpu_emulate_atomics(self, chipset: str = None):
        """Emulate atomic operations on chipsets that do not support them

        This pass rewrites any AMDGPU-specific atomic operation that is not supported
        on the given `chipset` into a compare-and-swap loop.

        Args:
            chipset: Chipset that these operations will run on
        """
        self.add_pass("amdgpu-emulate-atomics", chipset=chipset)
        return self

    def arith_bufferize(self, alignment: int = None):
        """Bufferize Arith dialect ops.

        This pass bufferizes arith dialect ops.

        This pass needs to be a module pass because it inserts memref.global
        ops into the module, which cannot be done safely from a function pass due to
        multi-threading. Most other bufferization passes can run in parallel at
        function granularity.

        Args:
            alignment: Create global memrefs with a specified alignment
        """
        self.add_pass("arith-bufferize", alignment=alignment)
        return self

    def arith_emulate_unsupported_floats(
        self, source_types: list[str] = None, target_type: str = None
    ):
        """Emulate operations on unsupported floats with extf/truncf

        Emulate arith and vector floating point operations that use float types
        which are unspported on a target by inserting extf/truncf pairs around all
        such operations in order to produce arithmetic that can be performed while
        preserving the original rounding behavior.

        This pass does not attempt to reason about the operations being performed
        to determine when type conversions can be elided.

        Args:
            source-types: MLIR types without arithmetic support on a given target
            target-type: MLIR type to convert the unsupported source types to
        """
        if source_types is not None and isinstance(source_types, (list, tuple)):
            source_types = ",".join(map(str, source_types))
        self.add_pass(
            "arith-emulate-unsupported-floats",
            source_types=source_types,
            target_type=target_type,
        )
        return self

    def arith_emulate_wide_int(self, widest_int_supported: int = None):
        """Emulate 2*N-bit integer operations using N-bit operations

        Emulate arith integer operations that use too wide integer types with
        equivalent operations on supported narrow integer types. This is done by
        splitting original integer values into two halves.

        This pass is intended preserve semantics but not necessarily provide the
        most efficient implementation.
        TODO: Optimize op emulation.

        Currently, only power-of-two integer bitwidths are supported.

        Args:
            widest-int-supported: Widest integer type supported by the target
        """
        self.add_pass(
            "arith-emulate-wide-int", widest_int_supported=widest_int_supported
        )
        return self

    def arith_expand(self, include_bf16: bool = None):
        """Legalize Arith ops to be convertible to LLVM.
        Args:
            include-bf16: Enable the BF16 expansion patterns
        """
        self.add_pass("arith-expand", include_bf16=include_bf16)
        return self

    def arith_int_narrowing(self, int_bitwidths_supported: list[int] = None):
        """Reduce integer operation bitwidth

        Reduce bitwidths of integer types used in arith operations. This pass
        prefers the narrowest available integer bitwidths that are guaranteed to
        produce the same results.

        Args:
            int-bitwidths-supported: Integer bitwidths supported
        """
        if int_bitwidths_supported is not None and isinstance(
            int_bitwidths_supported, (list, tuple)
        ):
            int_bitwidths_supported = ",".join(map(str, int_bitwidths_supported))
        self.add_pass(
            "arith-int-narrowing", int_bitwidths_supported=int_bitwidths_supported
        )
        return self

    def arith_unsigned_when_equivalent(self):
        """Replace signed ops with unsigned ones where they are proven equivalent

        Replace signed ops with their unsigned equivalents when integer range analysis
        determines that their arguments and results are all guaranteed to be
        non-negative when interpreted as signed integers. When this occurs,
        we know that the semantics of the signed and unsigned operations are the same,
        since they share the same behavior when their operands and results  are in the
        range [0, signed_max(type)].

        The affect ops include division, remainder, shifts, min, max, and integer
        comparisons.

        """
        self.add_pass("arith-unsigned-when-equivalent")
        return self

    def arm_neon_2d_to_intr(self):
        """Convert Arm NEON structured ops to intrinsics"""
        self.add_pass("arm-neon-2d-to-intr")
        return self

    def async_func_to_async_runtime(self):
        """Lower async.func operations to the explicit async.runtime andasync.coro operations"""
        self.add_pass("async-func-to-async-runtime")
        return self

    def async_parallel_for(
        self,
        async_dispatch: bool = None,
        num_workers: int = None,
        min_task_size: int = None,
    ):
        """Convert scf.parallel operations to multiple async compute ops executed concurrently for non-overlapping iteration ranges
        Args:
            async-dispatch: Dispatch async compute tasks using recursive work splitting. If `false` async compute tasks will be launched using simple for loop in the caller thread.
            num-workers: The number of available workers to execute async operations. If `-1` the value will be retrieved from the runtime.
            min-task-size: The minimum task size for sharding parallel operation.
        """
        self.add_pass(
            "async-parallel-for",
            async_dispatch=async_dispatch,
            num_workers=num_workers,
            min_task_size=min_task_size,
        )
        return self

    def async_runtime_policy_based_ref_counting(self):
        """Policy based reference counting for Async runtime operations

        This pass works at the async runtime abtraction level, after all
        `async.execute` and `async.await` operations are lowered to the async
        runtime API calls, and async coroutine operations.

        This pass doesn't rely on reference counted values liveness analysis, and
        instead uses simple policy to create reference counting operations. If the
        program violates any of the assumptions, then this pass might lead to
        memory leaks or runtime errors.

        The default reference counting policy assumptions:
          1. Async token can be awaited or added to the group only once.
          2. Async value or group can be awaited only once.

        Under these assumptions reference counting only needs to drop reference:
          1. After `async.runtime.await` operation for async tokens and groups
             (until error handling is not implemented for the sync await).
          2. After `async.runtime.is_error` operation for async tokens and groups
             (this is the last operation in the coroutine resume function).
          3. After `async.runtime.load` operation for async values.

        This pass introduces significanly less runtime overhead compared to the
        automatic reference counting.

        """
        self.add_pass("async-runtime-policy-based-ref-counting")
        return self

    def async_runtime_ref_counting(self):
        """Automatic reference counting for Async runtime operations

        This pass works at the async runtime abtraction level, after all
        `async.execute` and `async.await` operations are lowered to the async
        runtime API calls, and async coroutine operations.

        It relies on the LLVM coroutines switched-resume lowering semantics for
        the correct placing of the reference counting operations.

        See: https://llvm.org/docs/Coroutines.html#switched-resume-lowering

        """
        self.add_pass("async-runtime-ref-counting")
        return self

    def async_runtime_ref_counting_opt(self):
        """Optimize automatic reference counting operations for theAsync runtime by removing redundant operations"""
        self.add_pass("async-runtime-ref-counting-opt")
        return self

    def async_to_async_runtime(self):
        """Lower all high level async operations (e.g. async.execute) tothe explicit async.runtime and async.coro operations"""
        self.add_pass("async-to-async-runtime")
        return self

    def buffer_deallocation(self):
        """Adds all required dealloc operations for all allocations in the input program

        This pass implements an algorithm to automatically introduce all required
        deallocation operations for all buffers in the input program. This ensures
        that the resulting program does not have any memory leaks.


        Input

        ```mlir
        #map0 = affine_map<(d0) -> (d0)>
        module {
          func.func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
            cf.cond_br %arg0, ^bb1, ^bb2
          ^bb1:
            cf.br ^bb3(%arg1 : memref<2xf32>)
          ^bb2:
            %0 = memref.alloc() : memref<2xf32>
            linalg.generic {
              args_in = 1 : i64,
              args_out = 1 : i64,
              indexing_maps = [#map0, #map0],
              iterator_types = ["parallel"]} %arg1, %0 {
            ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
              %tmp1 = exp %gen1_arg0 : f32
              linalg.yield %tmp1 : f32
            }: memref<2xf32>, memref<2xf32>
            cf.br ^bb3(%0 : memref<2xf32>)
          ^bb3(%1: memref<2xf32>):
            "memref.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
            return
          }
        }

        ```

        Output

        ```mlir
        #map0 = affine_map<(d0) -> (d0)>
        module {
          func.func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
            cf.cond_br %arg0, ^bb1, ^bb2
          ^bb1:  // pred: ^bb0
            %0 = memref.alloc() : memref<2xf32>
            memref.copy(%arg1, %0) : memref<2xf32>, memref<2xf32>
            cf.br ^bb3(%0 : memref<2xf32>)
          ^bb2:  // pred: ^bb0
            %1 = memref.alloc() : memref<2xf32>
            linalg.generic {
              args_in = 1 : i64,
              args_out = 1 : i64,
              indexing_maps = [#map0, #map0],
              iterator_types = ["parallel"]} %arg1, %1 {
            ^bb0(%arg3: f32, %arg4: f32):
              %4 = exp %arg3 : f32
              linalg.yield %4 : f32
            }: memref<2xf32>, memref<2xf32>
            %2 = memref.alloc() : memref<2xf32>
            memref.copy(%1, %2) : memref<2xf32>, memref<2xf32>
            dealloc %1 : memref<2xf32>
            cf.br ^bb3(%2 : memref<2xf32>)
          ^bb3(%3: memref<2xf32>):  // 2 preds: ^bb1, ^bb2
            memref.copy(%3, %arg2) : memref<2xf32>, memref<2xf32>
            dealloc %3 : memref<2xf32>
            return
          }

        }
        ```


        """
        self.add_pass("buffer-deallocation")
        return self

    def buffer_hoisting(self):
        """Optimizes placement of allocation operations by moving them into common dominators and out of nested regions

        This pass implements an approach to aggressively move allocations upwards
        into common dominators and out of nested regions.

        """
        self.add_pass("buffer-hoisting")
        return self

    def buffer_loop_hoisting(self):
        """Optimizes placement of allocation operations by moving them out of loop nests

        This pass implements an approach to aggressively move allocations upwards
        out of loop nests. It does not move allocations into common dominators.

        """
        self.add_pass("buffer-loop-hoisting")
        return self

    def buffer_results_to_out_params(self):
        """Converts memref-typed function results to out-params

        Some calling conventions prefer to pass output memrefs as "out params". The
        conversion to this calling convention must be done as an atomic
        transformation of the entire program (hence this is a module pass).

        For example, if a call is rewritten, the callee needs to be rewritten
        otherwise the IR will end up invalid. Thus, this transformation
        require an atomic change to the entire program (e.g. the whole module).

        This pass is expected to run immediately after bufferization is finished.
        At that point, tensor-typed results will have been converted to memref-typed
        results, and can be consistently converted to out params.

        All memref-typed results are appended to the function argument list.

        The main issue with this pass (and the out-param calling convention) is that
        buffers for results need to be allocated in the caller. This currently only
        works for static shaped memrefs.

        """
        self.add_pass("buffer-results-to-out-params")
        return self

    def bufferization_bufferize(self):
        """Bufferize the `bufferization` dialect"""
        self.add_pass("bufferization-bufferize")
        return self

    def canonicalize(
        self,
        top_down: bool = None,
        region_simplify: bool = None,
        max_iterations: int = None,
        max_num_rewrites: int = None,
        test_convergence: bool = None,
        disable_patterns: list[str] = None,
        enable_patterns: list[str] = None,
    ):
        """Canonicalize operations

        This pass performs various types of canonicalizations over a set of
        operations by iteratively applying the canonicalization patterns of all
        loaded dialects until either a fixpoint is reached or the maximum number of
        iterations/rewrites is exhausted. Canonicalization is best-effort and does
        not guarantee that the entire IR is in a canonical form after running this
        pass. See [Operation Canonicalization](Canonicalization.md) for more
        details.

        Args:
            top-down: Seed the worklist in general top-down order
            region-simplify: Perform control flow optimizations to the region tree
            max-iterations: Max. iterations between applying patterns / simplifying regions
            max-num-rewrites: Max. number of pattern rewrites within an iteration
            test-convergence: Test only: Fail pass on non-convergence to detect cyclic pattern
            disable-patterns: Labels of patterns that should be filtered out during application
            enable-patterns: Labels of patterns that should be used during application, all other patterns are filtered out
        """
        if disable_patterns is not None and isinstance(disable_patterns, (list, tuple)):
            disable_patterns = ",".join(map(str, disable_patterns))
        if enable_patterns is not None and isinstance(enable_patterns, (list, tuple)):
            enable_patterns = ",".join(map(str, enable_patterns))
        self.add_pass(
            "canonicalize",
            top_down=top_down,
            region_simplify=region_simplify,
            max_iterations=max_iterations,
            max_num_rewrites=max_num_rewrites,
            test_convergence=test_convergence,
            disable_patterns=disable_patterns,
            enable_patterns=enable_patterns,
        )
        return self

    def control_flow_sink(self):
        """Sink operations into conditional blocks

        This pass implements control-flow sink on operations that implement
        `RegionBranchOpInterface` by moving dominating operations whose only uses
        are in a conditionally-executed regions into those regions so that
        executions paths where their results are not needed do not perform
        unnecessary computations.

        This is similar (but opposite) to loop-invariant code motion, which hoists
        operations out of regions executed more than once. The implementation of
        control-flow sink uses a simple and conversative cost model: operations are
        never duplicated and are only moved into singly-executed regions.

        It is recommended to run canonicalization first to remove unreachable
        blocks: ops in unreachable blocks may prevent other operations from being
        sunk as they may contain uses of their results

        """
        self.add_pass("control-flow-sink")
        return self

    def convert_amdgpu_to_rocdl(self, chipset: str = None):
        """Convert AMDGPU dialect to ROCDL dialect

        This pass converts supported AMDGPU ops to ROCDL dialect intrinsics.

        Args:
            chipset: Chipset that these operations will run on
        """
        self.add_pass("convert-amdgpu-to-rocdl", chipset=chipset)
        return self

    def convert_arith_to_llvm(self, index_bitwidth: int = None):
        """Convert Arith dialect to LLVM dialect

        This pass converts supported Arith ops to LLVM dialect instructions.

        Args:
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-arith-to-llvm", index_bitwidth=index_bitwidth)
        return self

    def convert_arith_to_spirv(
        self, emulate_lt_32_bit_scalar_types: bool = None, enable_fast_math: bool = None
    ):
        """Convert Arith dialect to SPIR-V dialect
        Args:
            emulate-lt-32-bit-scalar-types: Emulate narrower scalar types with 32-bit ones if not supported by the target
            enable-fast-math: Enable fast math mode (assuming no NaN and infinity for floating point values) when performing conversion
        """
        self.add_pass(
            "convert-arith-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
            enable_fast_math=enable_fast_math,
        )
        return self

    def convert_async_to_llvm(self, use_opaque_pointers: bool = None):
        """Convert the operations from the async dialect into the LLVM dialect

        Convert `async.execute` operations to LLVM coroutines and use async runtime
        API to execute them.

        Args:
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass("convert-async-to-llvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_bufferization_to_memref(self):
        """Convert operations from the Bufferization dialect to the MemRef dialect


        This pass converts bufferization operations into memref operations.

        In the current state, this pass only transforms a `bufferization.clone`
        operation into `memref.alloc` and `memref.copy` operations. This conversion
        is needed, since some clone operations could remain after applying several
        transformation processes. Currently, only `canonicalize` transforms clone
        operations or even eliminates them. This can lead to errors if any clone op
        survived after all conversion passes (starting from the bufferization
        dialect) are performed.

        See:
        https://llvm.discourse.group/t/bufferization-error-related-to-memref-clone/4665

        To avoid these errors, this pass can be performed as a last clean-up pass to
        transform remaining operations and to proceed in other dialects (memref
        e.g.).

        Note that this pass only transforms the operation without any further
        analyses. This pass does not consider any memory analysis or optimization
        and hence does not resolve any memory leaks.


        """
        self.add_pass("convert-bufferization-to-memref")
        return self

    def convert_cf_to_llvm(
        self, index_bitwidth: int = None, use_opaque_pointers: bool = None
    ):
        """Convert ControlFlow operations to the LLVM dialect

        Convert ControlFlow operations into LLVM IR dialect operations.

        If other operations are present and their results are required by the LLVM
        IR dialect operations, the pass will fail.  Any LLVM IR operations or types
        already present in the IR will be kept as is.

        Args:
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "convert-cf-to-llvm",
            index_bitwidth=index_bitwidth,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_cf_to_spirv(self, emulate_lt_32_bit_scalar_types: bool = None):
        """Convert ControlFlow dialect to SPIR-V dialect
        Args:
            emulate-lt-32-bit-scalar-types: Emulate narrower scalar types with 32-bit ones if not supported by the target
        """
        self.add_pass(
            "convert-cf-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
        )
        return self

    def convert_complex_to_libm(self):
        """Convert Complex dialect to libm calls

        This pass converts supported Complex ops to libm calls.

        """
        self.add_pass("convert-complex-to-libm")
        return self

    def convert_complex_to_llvm(self):
        """Convert Complex dialect to LLVM dialect"""
        self.add_pass("convert-complex-to-llvm")
        return self

    def convert_complex_to_spirv(self):
        """Convert Complex dialect to SPIRV dialect"""
        self.add_pass("convert-complex-to-spirv")
        return self

    def convert_complex_to_standard(self):
        """Convert Complex dialect to standard dialect"""
        self.add_pass("convert-complex-to-standard")
        return self

    def convert_elementwise_to_linalg(self):
        """Convert ElementwiseMappable ops to linalg

        Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

        This pass only converts ops that operate on ranked tensors. It can be
        run on op which contains linalg ops (most commonly a
        FunctionOpInterface op).

        """
        self.add_pass("convert-elementwise-to-linalg")
        return self

    def convert_func_to_llvm(
        self,
        use_bare_ptr_memref_call_conv: bool = None,
        index_bitwidth: int = None,
        data_layout: str = None,
        use_opaque_pointers: bool = None,
    ):
        """Convert from the Func dialect to the LLVM dialect

        Convert Func dialect operations into the LLVM IR dialect operations.

        #### Input invariant

        -   no `tensor` types;
        -   all `vector` are one-dimensional;
        -   all blocks are reachable by following the successors of the first basic
            block;

        If other operations are present and their results are required by the LLVM
        IR dialect operations, the pass will fail.  Any LLVM IR operations or types
        already present in the IR will be kept as is.

        #### Output IR

        Functions converted to LLVM IR. Function arguments types are converted
        one-to-one. Function results are converted one-to-one and, in case more than
        1 value is returned, packed into an LLVM IR struct type. Function calls and
        returns are updated accordingly. Block argument types are updated to use
        LLVM IR types.

        Args:
            use-bare-ptr-memref-call-conv: Replace FuncOp's MemRef arguments with bare pointers to the MemRef element types
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
            data-layout: String description (LLVM format) of the data layout that is expected on the produced module
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "convert-func-to-llvm",
            use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
            index_bitwidth=index_bitwidth,
            data_layout=data_layout,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_func_to_spirv(self, emulate_lt_32_bit_scalar_types: bool = None):
        """Convert Func dialect to SPIR-V dialect
        Args:
            emulate-lt-32-bit-scalar-types: Emulate narrower scalar types with 32-bit ones if not supported by the target
        """
        self.add_pass(
            "convert-func-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
        )
        return self

    def convert_gpu_launch_to_vulkan_launch(self):
        """Convert gpu.launch_func to vulkanLaunch external call

        This pass is only intended for the mlir-vulkan-runner.

        """
        self.add_pass("convert-gpu-launch-to-vulkan-launch")
        return self

    def convert_gpu_to_nvvm(
        self,
        index_bitwidth: int = None,
        has_redux: bool = None,
        use_bare_ptr_memref_call_conv: bool = None,
        use_opaque_pointers: bool = None,
    ):
        """Generate NVVM operations for gpu operations
        Args:
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
            has-redux: Target gpu supports redux
            use-bare-ptr-memref-call-conv: Replace memref arguments in GPU functions with bare pointers. All memrefs must have static shape.
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "convert-gpu-to-nvvm",
            index_bitwidth=index_bitwidth,
            has_redux=has_redux,
            use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_gpu_to_rocdl(
        self,
        chipset: str = None,
        index_bitwidth: int = None,
        use_bare_ptr_memref_call_conv: bool = None,
        runtime: "gpu::amd::Runtime" = None,
        use_opaque_pointers: bool = None,
    ):
        """Generate ROCDL operations for gpu operations
        Args:
            chipset: Chipset that these operations will run on
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
            use-bare-ptr-memref-call-conv: Replace memref arguments in GPU functions with bare pointers.All memrefs must have static shape
            runtime: Runtime code will be run on (default is Unknown, can also use HIP or OpenCl)
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "convert-gpu-to-rocdl",
            chipset=chipset,
            index_bitwidth=index_bitwidth,
            use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
            runtime=runtime,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_gpu_to_spirv(self, use_64bit_index: bool = None):
        """Convert GPU dialect to SPIR-V dialect

        This pass converts supported GPU device ops to SPIR-V ops. It does not
        handle GPU host ops.

        A `gpu.func` op can have parameters to pass in resources. But in SPIR-V
        entry functions cannot take parameters; they use descriptors to access
        resources. By default, parameters to a `gpu.func` op will be converted to
        global variables. These global variables will be assigned sequential binding
        numbers following their order in the original `gpu.func` op, starting from
        0, in set 0. One can attach `spirv.interface_var_abi` to those parameters
        to control the set and binding if wanted.

        Args:
            use-64bit-index: Use 64-bit integers to convert index types
        """
        self.add_pass("convert-gpu-to-spirv", use_64bit_index=use_64bit_index)
        return self

    def convert_index_to_llvm(self, index_bitwidth: int = None):
        """Lower the `index` dialect to the `llvm` dialect.

        This pass lowers Index dialect operations to LLVM dialect operations.
        Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`,
        `ceildivu`, and `floordivs`, which expand to series of LLVM operations.
        Importantly, the index bitwidth should be correctly set to the target
        pointer width via `index-bitwidth`.

        Args:
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-index-to-llvm", index_bitwidth=index_bitwidth)
        return self

    def convert_linalg_to_affine_loops(self):
        """Lower the operations from the linalg dialect into affine loops"""
        self.add_pass("convert-linalg-to-affine-loops")
        return self

    def convert_linalg_to_loops(self):
        """Lower the operations from the linalg dialect into loops"""
        self.add_pass("convert-linalg-to-loops")
        return self

    def convert_linalg_to_parallel_loops(self):
        """Lower the operations from the linalg dialect into parallel loops"""
        self.add_pass("convert-linalg-to-parallel-loops")
        return self

    def convert_linalg_to_std(self):
        """Convert the operations from the linalg dialect into the Standard dialect"""
        self.add_pass("convert-linalg-to-std")
        return self

    def convert_math_to_funcs(
        self, min_width_of_fpowi_exponent: int = None, convert_ctlz: bool = None
    ):
        """Convert Math operations to calls of outlined implementations.

        This pass converts supported Math ops to calls of compiler generated
        functions implementing these operations in software.
        The LLVM dialect is used for LinkonceODR linkage of the generated functions.

        Args:
            min-width-of-fpowi-exponent: Convert FPowI only if the width of its exponent's integer type is greater than or equal to this value
            convert-ctlz: Convert math.ctlz to a software implementation. Enable for targets that do not natively support ctlz.
        """
        self.add_pass(
            "convert-math-to-funcs",
            min_width_of_fpowi_exponent=min_width_of_fpowi_exponent,
            convert_ctlz=convert_ctlz,
        )
        return self

    def convert_math_to_libm(self):
        """Convert Math dialect to libm calls

        This pass converts supported Math ops to libm calls.

        """
        self.add_pass("convert-math-to-libm")
        return self

    def convert_math_to_llvm(self, approximate_log1p: bool = None):
        """Convert Math dialect to LLVM dialect
        Args:
            approximate-log1p: Enable approximation of Log1p.
        """
        self.add_pass("convert-math-to-llvm", approximate_log1p=approximate_log1p)
        return self

    def convert_math_to_spirv(self):
        """Convert Math dialect to SPIR-V dialect"""
        self.add_pass("convert-math-to-spirv")
        return self

    def convert_memref_to_spirv(self, bool_num_bits: int = None):
        """Convert MemRef dialect to SPIR-V dialect
        Args:
            bool-num-bits: The number of bits to store a boolean value
        """
        self.add_pass("convert-memref-to-spirv", bool_num_bits=bool_num_bits)
        return self

    def convert_nvgpu_to_nvvm(self, use_opaque_pointers: bool = None):
        """Convert NVGPU dialect to NVVM dialect

        This pass converts supported NVGPU ops to NVVM dialect intrinsics.

        Args:
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass("convert-nvgpu-to-nvvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_nvvm_to_llvm(self):
        """Convert NVVM dialect to LLVM dialect

        This pass generates inline assembly for the NVVM ops which is not
        implemented in LLVM core.

        """
        self.add_pass("convert-nvvm-to-llvm")
        return self

    def convert_openacc_to_scf(self):
        """Convert the OpenACC ops to OpenACC with SCF dialect"""
        self.add_pass("convert-openacc-to-scf")
        return self

    def convert_openmp_to_llvm(self):
        """Convert the OpenMP ops to OpenMP ops with LLVM dialect"""
        self.add_pass("convert-openmp-to-llvm")
        return self

    def convert_parallel_loops_to_gpu(self):
        """Convert mapped scf.parallel ops to gpu launch operations"""
        self.add_pass("convert-parallel-loops-to-gpu")
        return self

    def convert_pdl_to_pdl_interp(self):
        """Convert PDL ops to PDL interpreter ops"""
        self.add_pass("convert-pdl-to-pdl-interp")
        return self

    def convert_scf_to_cf(self):
        """Convert SCF dialect to ControlFlow dialect, replacing structured control flow with a CFG"""
        self.add_pass("convert-scf-to-cf")
        return self

    def convert_scf_to_openmp(self, use_opaque_pointers: bool = None):
        """Convert SCF parallel loop to OpenMP parallel + workshare constructs.
        Args:
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass("convert-scf-to-openmp", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_scf_to_spirv(self):
        """Convert SCF dialect to SPIR-V dialect.

        Converts SCF ops into SPIR-V structured control flow ops.
        SPIR-V structured control flow ops do not support yielding values.
        So for SCF ops yielding values, SPIR-V variables are created for
        holding the values and load/store operations are emitted for updating
        them.

        """
        self.add_pass("convert-scf-to-spirv")
        return self

    def convert_shape_constraints(self):
        """Convert shape constraint operations to the standard dialect

        This pass eliminates shape constraints from the program, converting them to
        eager (side-effecting) error handling code.

        This pass is separate from the regular convert-shape-to-standard, despite
        converting between the same dialects, because converting shape constraints
        can happen at a different part of the program than general shape
        computation lowering.

        """
        self.add_pass("convert-shape-constraints")
        return self

    def convert_shape_to_std(self):
        """Convert operations from the shape dialect into the standard dialect"""
        self.add_pass("convert-shape-to-std")
        return self

    def convert_spirv_to_llvm(self, use_opaque_pointers: bool = None):
        """Convert SPIR-V dialect to LLVM dialect

        See https://from ....llvm.org/docs/SPIRVToLLVMDialectConversion/
        for more details.

        Args:
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass("convert-spirv-to-llvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_tensor_to_linalg(self):
        """Convert some Tensor dialect ops to Linalg dialect"""
        self.add_pass("convert-tensor-to-linalg")
        return self

    def convert_tensor_to_spirv(self, emulate_lt_32_bit_scalar_types: bool = None):
        """Convert Tensor dialect to SPIR-V dialect
        Args:
            emulate-lt-32-bit-scalar-types: Emulate narrower scalar types with 32-bit ones if not supported by the target
        """
        self.add_pass(
            "convert-tensor-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
        )
        return self

    def convert_ub_to_llvm(self, index_bitwidth: int = None):
        """Convert UB dialect to LLVM dialect

        This pass converts supported UB ops to LLVM dialect instructions.

        Args:
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-ub-to-llvm", index_bitwidth=index_bitwidth)
        return self

    def convert_ub_to_spirv(self):
        """Convert UB dialect to SPIR-V dialect

        This pass converts supported UB ops to SPIR-V dialect ops.

        """
        self.add_pass("convert-ub-to-spirv")
        return self

    def convert_vector_to_arm_sme(self):
        """Lower the operations from the vector dialect into the ArmSME dialect

        Pass that converts vector dialect operations into equivalent ArmSME dialect
        operations.

        """
        self.add_pass("convert-vector-to-arm-sme")
        return self

    def convert_vector_to_gpu(self, use_nvgpu: bool = None):
        """Lower the operations from the vector dialect into the GPU dialect
        Args:
            use-nvgpu: convert to NvGPU ops instead of GPU dialect ops
        """
        self.add_pass("convert-vector-to-gpu", use_nvgpu=use_nvgpu)
        return self

    def convert_vector_to_llvm(
        self,
        reassociate_fp_reductions: bool = None,
        force_32bit_vector_indices: bool = None,
        enable_amx: bool = None,
        enable_arm_neon: bool = None,
        enable_arm_sve: bool = None,
        enable_arm_sme: bool = None,
        enable_x86vector: bool = None,
        use_opaque_pointers: bool = None,
    ):
        """Lower the operations from the vector dialect into the LLVM dialect


        Convert operations from the vector dialect into the LLVM IR dialect
        operations. The lowering pass provides several options to control
        the kinds of optimizations that are allowed. It also provides options
        that enable the use of one or more architectural-specific dialects
        (AMX, X86Vector, ArmNeon, ArmSVE, etc.) in combination with the
        architectural-neutral vector dialect lowering.


        Args:
            reassociate-fp-reductions: Allows llvm to reassociate floating-point reductions for speed
            force-32bit-vector-indices: Allows compiler to assume vector indices fit in 32-bit if that yields faster code
            enable-amx: Enables the use of AMX dialect while lowering the vector dialect.
            enable-arm-neon: Enables the use of ArmNeon dialect while lowering the vector dialect.
            enable-arm-sve: Enables the use of ArmSVE dialect while lowering the vector dialect.
            enable-arm-sme: Enables the use of ArmSME dialect while lowering the vector dialect.
            enable-x86vector: Enables the use of X86Vector dialect while lowering the vector dialect.
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "convert-vector-to-llvm",
            reassociate_fp_reductions=reassociate_fp_reductions,
            force_32bit_vector_indices=force_32bit_vector_indices,
            enable_amx=enable_amx,
            enable_arm_neon=enable_arm_neon,
            enable_arm_sve=enable_arm_sve,
            enable_arm_sme=enable_arm_sme,
            enable_x86vector=enable_x86vector,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_vector_to_scf(
        self,
        full_unroll: bool = None,
        target_rank: int = None,
        lower_tensors: bool = None,
    ):
        """Lower the operations from the vector dialect into the SCF dialect
        Args:
            full-unroll: Perform full unrolling when converting vector transfers to SCF
            target-rank: Target vector rank to which transfer ops should be lowered
            lower-tensors: Lower transfer ops that operate on tensors
        """
        self.add_pass(
            "convert-vector-to-scf",
            full_unroll=full_unroll,
            target_rank=target_rank,
            lower_tensors=lower_tensors,
        )
        return self

    def convert_vector_to_spirv(self):
        """Convert Vector dialect to SPIR-V dialect"""
        self.add_pass("convert-vector-to-spirv")
        return self

    def cse(self):
        """Eliminate common sub-expressions

        This pass implements a generalized algorithm for common sub-expression
        elimination. This pass relies on information provided by the
        `Memory SideEffect` interface to identify when it is safe to eliminate
        operations. See [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
        for more general details on this optimization.

        """
        self.add_pass("cse")
        return self

    def decorate_spirv_composite_type_layout(self):
        """Decorate SPIR-V composite type with layout info

        Module pass that converts composite types used by objects in the
        StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant storage
        classes to attatch layout information.
        Right now this pass only supports Vulkan layout rules.

        """
        self.add_pass("decorate-spirv-composite-type-layout")
        return self

    def drop_equivalent_buffer_results(self):
        """Remove MemRef return values that are equivalent to a bbArg

        This pass removes MemRef return values from functions if they are equivalent
        to a function bbArg. In that case, the return value is redundant and the
        respective CallOp operand can be used at the call site.

        Note: If a bbArg buffer is not returned directly but casted to beforehand,
        the buffer is still considered equivalent.

        """
        self.add_pass("drop-equivalent-buffer-results")
        return self

    def duplicate_function_elimination(self):
        """Deduplicate functions

        Deduplicate functions that are equivalent in all aspects but their symbol
        name. The pass chooses one representative per equivalence class, erases
        the remainder, and updates function calls accordingly.

        """
        self.add_pass("duplicate-function-elimination")
        return self

    def eliminate_empty_tensors(self):
        """Try to eliminate all tensor.empty ops.

        This pass tries to eliminate all insert_slice op-anchored tensor.empty ops.
        I.e., when a value that is equivalent to an tensor.empty op is inserted into
        another tensor, this pass tries to rewrite the IR in such a way that the
        destination tensor of the insert_slice op is used directly instead of the
        tensor.empty result.

        """
        self.add_pass("eliminate-empty-tensors")
        return self

    def empty_tensor_to_alloc_tensor(self):
        """Replace all empty ops by alloc_tensor ops.

        tensor.empty ops return a tensor of unspecified contents who's only purpose
        is to carry the tensor shape. This pass converts such ops to
        bufferization.alloc_tensor ops, which bufferize to buffer allocations.

        """
        self.add_pass("empty-tensor-to-alloc-tensor")
        return self

    def enable_arm_streaming(
        self, mode: "arm_sme::ArmStreaming" = None, enable_za: bool = None
    ):
        """Enable Armv9 Streaming SVE mode

        Enables the Armv9 Streaming SVE mode [1] for func.func ops by annotating
        them with attributes. See options for more details.

        [1] https://developer.arm.com/documentation/ddi0616/aa

        Args:
            mode: Select how streaming-mode is managed at the function-level.
            enable-za: Enable ZA storage array.
        """
        self.add_pass("enable-arm-streaming", mode=mode, enable_za=enable_za)
        return self

    def ensure_debug_info_scope_on_llvm_func(self):
        """Materialize LLVM debug info subprogram attribute on every LLVMFuncOp

        Having a debug info subprogram attribute on a function is required for
        emitting line tables from MLIR FileLocCol locations.

        This is not intended to be a proper replacement for frontends to emit
        complete debug informations, however it is a convenient way to get line
        tables for debugging purposes. This allow to step trough in a debugger
        line-by-line or get a backtrace with line numbers.

        """
        self.add_pass("ensure-debug-info-scope-on-llvm-func")
        return self

    def expand_strided_metadata(self):
        """Expand memref operations into easier to analyze constructs

        The pass expands memref operations that modify the metadata of a memref
        (sizes, offset, strides) into a sequence of easier to analyze constructs.
        In particular, this pass transforms operations into explicit sequence of
        operations that model the effect of this operation on the different metadata.
        This pass uses affine constructs to materialize these effects.

        """
        self.add_pass("expand-strided-metadata")
        return self

    def finalize_memref_to_llvm(
        self,
        use_aligned_alloc: bool = None,
        index_bitwidth: int = None,
        use_generic_functions: bool = None,
        use_opaque_pointers: bool = None,
    ):
        """Finalize MemRef dialect to LLVM dialect conversion

        Finalize the conversion of the operations from the MemRef
        dialect to the LLVM dialect.
        This conversion will not convert some complex MemRef
        operations. Make sure to run `expand-strided-metadata`
        beforehand for these.

        Args:
            use-aligned-alloc: Use aligned_alloc in place of malloc for heap allocations
            index-bitwidth: Bitwidth of the index type, 0 to use size of machine word
            use-generic-functions: Use generic allocation and deallocation functions instead of the classic 'malloc', 'aligned_alloc' and 'free' functions
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "finalize-memref-to-llvm",
            use_aligned_alloc=use_aligned_alloc,
            index_bitwidth=index_bitwidth,
            use_generic_functions=use_generic_functions,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def finalizing_bufferize(self):
        """Finalize a partial bufferization

        A bufferize pass that finalizes a partial bufferization by removing
        remaining `bufferization.to_tensor` and `bufferization.to_buffer` operations.

        The removal of those operations is only possible if the operations only
        exist in pairs, i.e., all uses of `bufferization.to_tensor` operations are
        `bufferization.to_buffer` operations.

        This pass will fail if not all operations can be removed or if any operation
        with tensor typed operands remains.

        """
        self.add_pass("finalizing-bufferize")
        return self

    def fold_memref_alias_ops(self):
        """Fold memref alias ops into consumer load/store ops

        The pass folds loading/storing from/to memref aliasing ops to loading/storing
        from/to the original memref.

        """
        self.add_pass("fold-memref-alias-ops")
        return self

    def fold_tensor_subset_ops(self):
        """Fold tensor subset ops into producer/consumer ops

        The pass folds tensor subset ops into producer/consumer ops.

        At the moment, the following foldings occur when possible:
          - tensor.extract_slice into vector.transfer_read
          - vector.transfer_write into tensor.insert_slice


        """
        self.add_pass("fold-tensor-subset-ops")
        return self

    def func_bufferize(self):
        """Bufferize func/call/return ops

        A bufferize pass that bufferizes func.func and func.call ops.

        Because this pass updates func.func ops, it must be a module pass. It is
        useful to keep this pass separate from other bufferizations so that the
        other ones can be run at function-level in parallel.

        This pass must be done atomically because it changes func op signatures,
        which requires atomically updating calls as well throughout the entire
        module.

        This pass also changes the type of block arguments, which requires that all
        successor arguments of predecessors be converted. This is achieved by
        rewriting terminators based on the information provided by the
        `BranchOpInterface`.
        As this pass rewrites function operations, it also rewrites the
        corresponding return operations. Other return-like operations that
        implement the `ReturnLike` trait are not rewritten in general, as they
        require that the corresponding parent operation is also rewritten.
        Finally, this pass fails for unknown terminators, as we cannot decide
        whether they need rewriting.

        """
        self.add_pass("func-bufferize")
        return self

    def generate_runtime_verification(self):
        """Generate additional runtime op verification checks

        This pass generates op-specific runtime checks using the
        `RuntimeVerifiableOpInterface`. It can be run for debugging purposes after
        passes that are suspected to introduce faulty IR.

        """
        self.add_pass("generate-runtime-verification")
        return self

    def gpu_async_region(self):
        """Make GPU ops async"""
        self.add_pass("gpu-async-region")
        return self

    def gpu_kernel_outlining(self):
        """Outline gpu.launch bodies to kernel functions"""
        self.add_pass("gpu-kernel-outlining")
        return self

    def gpu_launch_sink_index_computations(self):
        """Sink index computations into gpu.launch body"""
        self.add_pass("gpu-launch-sink-index-computations")
        return self

    def gpu_map_parallel_loops(self):
        """Greedily maps loops to GPU hardware dimensions.
        Greedily maps loops to GPU hardware dimensions.
        """
        self.add_pass("gpu-map-parallel-loops")
        return self

    def gpu_to_llvm(
        self,
        use_bare_pointers_for_host: bool = None,
        use_bare_pointers_for_kernels: bool = None,
        gpu_binary_annotation: str = None,
        use_opaque_pointers: bool = None,
    ):
        """Convert GPU dialect to LLVM dialect with GPU runtime calls

        Creates a pass to convert a GPU operations into a sequence of GPU runtime
        calls.

        This pass does not generate code to call GPU runtime APIs directly but
        instead uses a small wrapper library that exports a stable and conveniently
        typed ABI on top of GPU runtimes such as CUDA or ROCm (HIP).

        Args:
            use-bare-pointers-for-host: Use bare pointers to pass memref arguments to host functions. All memrefs must have static shape.
            use-bare-pointers-for-kernels: Use bare pointers to pass memref arguments to kernels. The kernel must use the same setting for this option.
            gpu-binary-annotation: Annotation attribute string for GPU binary
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass(
            "gpu-to-llvm",
            use_bare_pointers_for_host=use_bare_pointers_for_host,
            use_bare_pointers_for_kernels=use_bare_pointers_for_kernels,
            gpu_binary_annotation=gpu_binary_annotation,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def inline(
        self,
        default_pipeline: str = None,
        op_pipelines: list["OpPassManager"] = None,
        max_iterations: int = None,
    ):
        """Inline function calls
        Args:
            default-pipeline: The default optimizer pipeline used for callables
            op-pipelines: Callable operation specific optimizer pipelines (in the form of `dialect.op(pipeline)`)
            max-iterations: Maximum number of iterations when inlining within an SCC
        """
        if op_pipelines is not None and isinstance(op_pipelines, (list, tuple)):
            op_pipelines = ",".join(map(str, op_pipelines))
        self.add_pass(
            "inline",
            default_pipeline=default_pipeline,
            op_pipelines=op_pipelines,
            max_iterations=max_iterations,
        )
        return self

    def int_range_optimizations(self):
        """Do optimizations based on integer range analysis

        This pass runs integer range analysis and apllies optimizations based on its
        results. e.g. replace arith.cmpi with const if it can be inferred from
        args ranges.

        """
        self.add_pass("int-range-optimizations")
        return self

    def launch_func_to_vulkan(self, use_opaque_pointers: bool = None):
        """Convert vulkanLaunch external call to Vulkan runtime external calls

        This pass is only intended for the mlir-vulkan-runner.

        Args:
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass("launch-func-to-vulkan", use_opaque_pointers=use_opaque_pointers)
        return self

    def linalg_bufferize(self):
        """Bufferize the linalg dialect"""
        self.add_pass("linalg-bufferize")
        return self

    def linalg_fold_unit_extent_dims(self, use_rank_reducing_slices: bool = None):
        """Remove unit-extent dimension in Linalg ops on tensors
        Args:
            use-rank-reducing-slices: Generate rank-reducing slices instead of reassociative reshapes
        """
        self.add_pass(
            "linalg-fold-unit-extent-dims",
            use_rank_reducing_slices=use_rank_reducing_slices,
        )
        return self

    def linalg_fuse_elementwise_ops(self):
        """Fuse elementwise operations on tensors"""
        self.add_pass("linalg-fuse-elementwise-ops")
        return self

    def linalg_generalize_named_ops(self):
        """Convert named ops into generic ops"""
        self.add_pass("linalg-generalize-named-ops")
        return self

    def linalg_inline_scalar_operands(self):
        """Inline scalar operands into linalg generic ops"""
        self.add_pass("linalg-inline-scalar-operands")
        return self

    def linalg_named_op_conversion(self):
        """Convert from one named linalg op to another."""
        self.add_pass("linalg-named-op-conversion")
        return self

    def llvm_legalize_for_export(self):
        """Legalize LLVM dialect to be convertible to LLVM IR"""
        self.add_pass("llvm-legalize-for-export")
        return self

    def llvm_optimize_for_nvvm_target(self):
        """Optimize NVVM IR"""
        self.add_pass("llvm-optimize-for-nvvm-target")
        return self

    def llvm_request_c_wrappers(self):
        """Request C wrapper emission for all functions

        Annotate every builtin function in the module with the LLVM dialect
        attribute that instructs the conversion to LLVM to emit the C wrapper for
        the function. This pass is expected to be applied immediately before the
        conversion of builtin functions to LLVM to avoid the attribute being
        dropped by other passes.

        """
        self.add_pass("llvm-request-c-wrappers")
        return self

    def llvm_type_consistency(self, max_vector_split_size: int = None):
        """Rewrites to improve type consistency

        Set of rewrites to improve the coherency of types within an LLVM dialect
        program. This will adjust operations operating on pointers so they interpret
        their associated pointee type as consistently as possible.

        Args:
            max-vector-split-size: Maximum size in bits of a vector value in a load or store operation operating on multiple elements that should still be split
        """
        self.add_pass(
            "llvm-type-consistency", max_vector_split_size=max_vector_split_size
        )
        return self

    def loop_invariant_code_motion(self):
        """Hoist loop invariant instructions outside of the loop"""
        self.add_pass("loop-invariant-code-motion")
        return self

    def lower_affine(self):
        """Lower Affine operations to a combination of Standard and SCF operations


        Convert operations from the affine dialect into operations from the SCF and
        standard dialects.

        `affine.for` operations are converted to `scf.for` operations that are free
        of certain structural restrictions (on their bounds and step). `affine.if`
        is similarly converted to the `scf.if` operation. `affine.apply` operations
        are converted into sequences of primitive arithmetic operations from the
        standard dialect that have the same effect, using operands of the `index`
        type. Consequently, named maps and sets thare are no longer in use may be
        removed from the module.

        For example, `%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 +
        s0)>(%d0, %d1)[%s0]`
        can be converted into:

        ```mlir
        %d0 = <...>
        %d1 = <...>
        %s0 = <...>
        %0 = arith.constant 2 : index
        %1 = arith.muli %0, %d1
        %2 = arith.addi %d0, %1
        %r = arith.addi %2, %s0
        ```

        #### Input invariant

        -   no `Tensor` types;

        These restrictions may be lifted in the future.

        #### Output IR

        Functions with `affine.for` and `affine.if` operations eliminated. These
        functions may contain operations from the Standard dialect in addition to
        those already present before the pass.

        #### Invariants

        -   Functions without a body are not modified.
        -   The semantics of the other functions is preserved.
        -   Individual operations other than those mentioned above are not modified
            if they do not depend on the loop iterator value or on the result of
            `affine.apply`.

        """
        self.add_pass("lower-affine")
        return self

    def lower_host_to_llvm(self, use_opaque_pointers: bool = None):
        """Lowers the host module code and `gpu.launch_func` to LLVM

        Creates a pass to emulate `gpu.launch_func` call in LLVM dialect and lower
        the host module code to LLVM.

        This transformation creates a sequence of global variables that are later
        linked to the variables in the kernel module, and a series of copies to/from
        them to emulate the memory transfer from the host or to the device sides. It
        also converts the remaining Arithmetic, Func, and MemRef dialects into LLVM
        dialect, emitting C wrappers.

        Args:
            use-opaque-pointers: Generate LLVM IR using opaque pointers instead of typed pointers
        """
        self.add_pass("lower-host-to-llvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def lower_vector_mask(self):
        """Lower 'vector.mask' operations"""
        self.add_pass("lower-vector-mask")
        return self

    def map_memref_spirv_storage_class(self, client_api: str = None):
        """Map numeric MemRef memory spaces to SPIR-V storage classes
        Args:
            client-api: The client API to use for populating mappings
        """
        self.add_pass("map-memref-spirv-storage-class", client_api=client_api)
        return self

    def math_uplift_to_fma(self):
        """Uplift arith ops to math.fma.

        Uplift sequence of addf and mulf ops to math.fma if fastmath flags allows it.

        """
        self.add_pass("math-uplift-to-fma")
        return self

    def mem2reg(self, region_simplify: bool = None):
        """Promotes memory slots into values.

        This pass removes loads out of and stores into a memory slot, and turns
        them into direct uses of SSA values. This is done generically using the
        `PromoteAllocationOpInterface`, `PromoteOpInterface` and
        `PromoteMemOpInterface` interfaces.

        This pass will attempt to compute which definitions of the content of
        the memory slot reach operations that use the memory slot pointer. It
        will rewire or remove operations that use the slot pointer so they no
        longer use it. If any of this is not possible, the IR will be left
        without mutation.

        This pass only supports unstructured control-flow. Promotion of operations
        within subregions will not happen.

        Args:
            region-simplify: Perform control flow optimizations to the region tree
        """
        self.add_pass("mem2reg", region_simplify=region_simplify)
        return self

    def memref_emulate_wide_int(self, widest_int_supported: int = None):
        """Emulate 2*N-bit integer operations using N-bit operations

        Emulate memref integer operations that use too wide integer types with
        equivalent operations on supported narrow integer types. This is done by
        splitting original integer values into two halves.

        Currently, only power-of-two integer bitwidths are supported.

        Args:
            widest-int-supported: Widest integer type supported by the target
        """
        self.add_pass(
            "memref-emulate-wide-int", widest_int_supported=widest_int_supported
        )
        return self

    def memref_expand(self):
        """Legalize memref operations to be convertible to LLVM."""
        self.add_pass("memref-expand")
        return self

    def normalize_memrefs(self):
        """Normalize memrefs

          This pass transforms memref types with a non-trivial
          [layout map](https://from ....llvm.org/docs/Dialects/Builtin/#affine-map-layout)
          into memref types with an identity layout map, e.g. (i, j) -> (i, j). This
          pass is inter-procedural, in the sense that it can modify function
          interfaces and call sites that pass memref types. In order to modify
          memref types while preserving the original behavior, users of those
          memref types are also modified to incorporate the resulting layout map.
          For instance, an [AffineLoadOp](https://from ....llvm.org/docs/Dialects/Affine/#affineload-mliraffineloadop)
          will be updated to compose the layout map with with the affine expression
          contained in the op. Operations marked with the
          [MemRefsNormalizable](https://from ....llvm.org/docs/Traits/#memrefsnormalizable)
          trait are expected to be normalizable. Supported operations include affine
          operations, memref.alloc, memref.dealloc, and func.return.

          Given an appropriate layout map specified in the code, this transformation
          can express tiled or linearized access to multi-dimensional data
          structures, but will not modify memref types without an explicit layout
          map.

          Currently this pass is limited to only modify
          functions where all memref types can be normalized. If a function
          contains any operations that are not MemRefNormalizable, then the function
          and any functions that call or call it will not be modified.

          Input

          ```mlir
          #tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
          func.func @matmul(%A: memref<16xf64, #tile>,
                       %B: index, %C: memref<16xf64>) -> (memref<16xf64, #tile>) {
            affine.for %arg3 = 0 to 16 {
                  %a = affine.load %A[%arg3] : memref<16xf64, #tile>
                  %p = arith.mulf %a, %a : f64
                  affine.store %p, %A[%arg3] : memref<16xf64, #tile>
            }
            %c = memref.alloc() : memref<16xf64, #tile>
            %d = affine.load %c[0] : memref<16xf64, #tile>
            return %A: memref<16xf64, #tile>
          }
          ```

          Output

          ```mlir
          func.func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
            -> memref<4x4xf64> {
            affine.for %arg3 = 0 to 16 {
              %3 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
              %4 = arith.mulf %3, %3 : f64
              affine.store %4, %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
            }
            %0 = memref.alloc() : memref<4x4xf64>
            %1 = affine.apply #map1()
            %2 = affine.load %0[0, 0] : memref<4x4xf64>
            return %arg0 : memref<4x4xf64>
          }
          ```

          Input

          ```
          #linear8 = affine_map<(i, j) -> (i * 8 + j)>
          func.func @linearize(%arg0: memref<8x8xi32, #linear8>,
                          %arg1: memref<8x8xi32, #linear8>,
                          %arg2: memref<8x8xi32, #linear8>) {
            %c8 = arith.constant 8 : index
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            affine.for %arg3 = %c0 to %c8  {
            affine.for %arg4 = %c0 to %c8  {
              affine.for %arg5 = %c0 to %c8 {
                %0 = affine.load %arg0[%arg3, %arg5] : memref<8x8xi32, #linear8>
                %1 = affine.load %arg1[%arg5, %arg4] : memref<8x8xi32, #linear8>
                %2 = affine.load %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
                %3 = arith.muli %0, %1 : i32
                %4 = arith.addi %2, %3 : i32
                affine.store %4, %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
              }
            }
            }
            return
          }
          ```

          Output

          ```mlir
          func.func @linearize(%arg0: memref<64xi32>,
                          %arg1: memref<64xi32>,
                          %arg2: memref<64xi32>) {
          %c8 = arith.constant 8 : index
          %c0 = arith.constant 0 : index
          affine.for %arg3 = %c0 to %c8 {
            affine.for %arg4 = %c0 to %c8 {
              affine.for %arg5 = %c0 to %c8 {
                %0 = affine.load %arg0[%arg3 * 8 + %arg5] : memref<64xi32>
                %1 = affine.load %arg1[%arg5 * 8 + %arg4] : memref<64xi32>
                %2 = affine.load %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
                %3 = arith.muli %0, %1 : i32
                %4 = arith.addi %2, %3 : i32
                affine.store %4, %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
              }
            }
          }
          return
        }
        ```

        """
        self.add_pass("normalize-memrefs")
        return self

    def nvgpu_optimize_shared_memory(self):
        """Optimizes accesses to shard memory memrefs in order to reduce bank conflicts."""
        self.add_pass("nvgpu-optimize-shared-memory")
        return self

    def one_shot_bufferize(
        self,
        allow_return_allocs: bool = None,
        allow_unknown_ops: bool = None,
        analysis_fuzzer_seed: int = None,
        analysis_heuristic: str = None,
        bufferize_function_boundaries: bool = None,
        copy_before_write: bool = None,
        create_deallocs: bool = None,
        dialect_filter: list[str] = None,
        dump_alias_sets: bool = None,
        no_analysis_func_filter: list[str] = None,
        function_boundary_type_conversion: str = None,
        must_infer_memory_space: bool = None,
        test_analysis_only: bool = None,
        print_conflicts: bool = None,
        unknown_type_conversion: str = None,
    ):
        """One-Shot Bufferize

        This pass bufferizes all ops that implement `BufferizableOpInterface`. It
        first performs an inplacability analysis on SSA use-def chains of tensor
        values to determine which OpOperands may bufferize in-place, i.e., without
        inserting a buffer copy. It then rewrites the IR, inserting a buffer
        allocation and copy for each OpOperand that was decided to bufferize
        out-of-place.

        One-Shot Bufferize (and `BufferizableOpInterface`) was designed for ops that
        are in destination-passing style. When bufferizing such ops, it is possible
        to reuse the buffer of a tensor OpOperand for a tensor OpResult. In essence,
        a possible destination of an operation is already passed as an SSA value.

        `tensor.insert` is an example for an op in destination-passing style. E.g.,
        when bufferizing `%t0 = tensor.insert %f into %dest[%idx]`, `buffer(%t0)` is
        identical to `buffer(%dest)` in the absence of RaW conflicts. As a counter
        example, `tensor.generate` is not in destination-passing style and always
        results in a new buffer allocation.

        One-Shot Bufferize deallocates all buffers that it allocates. Yielding newly
        allocated buffers from a block can lead to bad performance because
        additional buffer copies are often needed to make sure that every buffer
        allocation is also deallocated again. By default, such IR is rejected by
        One-Shot Bufferize. Such IR can be allowed with `allow-return-allocs`. In
        that case, the `-buffer-deallocation` pass should be run after One-Shot
        Bufferize. Note that new buffer allocations that are returned from a
        function can currently not be deallocated by `-buffer-deallocation` and
        leak.

        One-Shot Bufferize will by default reject IR that contains non-bufferizable
        op, i.e., ops that do not implemement BufferizableOpInterface. Such IR can
        be allowed with `allow-unknown-ops=1`. In that case, to_memref and to_tensor
        ops will be generated at the bufferization boundary. This is useful for
        compatibility with existing partial bufferization passes: These can
        bufferize the remaining IR after running One-Shot Bufferize.

        Note: Running One-Shot Bufferize after a partial bufferization pass is
        currently not supported. Running partial bufferization passes after running
        One-Shot Bufferize is supported and the recommended way to gradually
        migrate from partial bufferization to One-Shot Bufferize.

        With `dialect-filter`, bufferization can be restricted to a set of dialects.
        If no filter is specified, all ops that implement `BufferizableOpInterface`
        are bufferized. Ops from the `std` dialect are an exception: These ops are
        always ignored, even if no filter is specified. When specifying a dialect
        filter and `allow-unknown-ops` is not turned on, bufferization would fail
        when encountering an op that is not included in the filter (even if it is
        bufferizable).

        One-Shot Bufferize will by default assume memref types with fully dynamic
        layout maps when a precise layout cannot be inferred. E.g., this is the case
        when wrapping a non-bufferizable op in to_memref/to_tensor ops. This
        behavior can be overridden with `unknown-type-conversion`. Valid values are
        `fully-dynamic-layout-map` and `identity-layout-map`.

        For testing/debugging purposes, `test-analysis-only=1 print-conflicts=1`
        prints analysis results and explains why an OpOperand was decided to
        bufferize out-of-place. This is useful for understanding why One-Shot
        Bufferize chose to insert a certain buffer copy.

        `bufferize-function-boundaries` is an experimental flag for bufferizing
        `FuncOp`, `ReturnOp` and `CallOp`. This feature is still under development
        and supports only simple cases at the moment. In particular:

        * Recursive or circular function call graphs are not supported.
        * External functions (without bodies) that return a tensor are not
          supported.
        * Function with multiple blocks or multiple ReturnOps are not supported.
        * Layout maps on function signatures can be controlled with a separate
          `function-boundary-type-conversion` option, which is similar to
          `unknown-type-conversion` but supports an additional `infer-layout-map`
          option. `fully-dynamic-layout-map` and `identity-layout-map` ensure that
          function signatures bufferize to easily predictable types, potentially at
          the cost of additional casts and copies, respectively. When layout maps
          are inferred, function return types may be more precise, but less
          predictable. Function argument types cannot be inferred and always have
          fully dynamic layout maps with `infer-layout-map`.

        One-Shot Bufferize implements the following contract around function calls:
        The buffer of function arguments is always writable (unless annotated with
        `bufferization.writable = false`). A buffer copy may be inserted at the call
        site where necessary. Alias sets and equivalence info is propagated through
        function calls. Whenever a function is bufferized, all other functions that
        are being called were already analyzed and bufferized, so exact alias and
        equivalence information is available. This is why recursive function calls
        are not yet supported.

        One-Shot Bufferize gathers additional information during the analysis phase
        when function boundary bufferization is activated. E.g., whether a function
        argument is read/written and which returned values are aliasing/equivalent.
        For debugging purposes, such information can be printed with
        `test-analysis-only`.

        Args:
            allow-return-allocs: Allows returning/yielding new allocations from a block.
            allow-unknown-ops: Allows unknown (not bufferizable) ops in the input IR.
            analysis-fuzzer-seed: Test only: Analyze ops in random order with a given seed (fuzzer)
            analysis-heuristic: Heuristic that control the IR traversal during analysis
            bufferize-function-boundaries: Bufferize function boundaries (experimental).
            copy-before-write: Skip the analysis. Make a buffer copy on every write.
            create-deallocs: Specify if buffers should be deallocated. For compatibility with core bufferization passes.
            dialect-filter: Restrict bufferization to ops from these dialects.
            dump-alias-sets: Test only: Annotate tensor IR with alias sets
            no-analysis-func-filter: Skip analysis of functions with these symbol names.Set copyBeforeWrite to true when bufferizing them.
            function-boundary-type-conversion: Controls layout maps when bufferizing function signatures.
            must-infer-memory-space: The memory space of an memref types must always be inferred. If unset, a default memory space of 0 is used otherwise.
            test-analysis-only: Test only: Only run inplaceability analysis and annotate IR
            print-conflicts: Test only: Annotate IR with RaW conflicts. Requires test-analysis-only.
            unknown-type-conversion: Controls layout maps for non-inferrable memref types.
        """
        if dialect_filter is not None and isinstance(dialect_filter, (list, tuple)):
            dialect_filter = ",".join(map(str, dialect_filter))
        if no_analysis_func_filter is not None and isinstance(
            no_analysis_func_filter, (list, tuple)
        ):
            no_analysis_func_filter = ",".join(map(str, no_analysis_func_filter))
        self.add_pass(
            "one-shot-bufferize",
            allow_return_allocs=allow_return_allocs,
            allow_unknown_ops=allow_unknown_ops,
            analysis_fuzzer_seed=analysis_fuzzer_seed,
            analysis_heuristic=analysis_heuristic,
            bufferize_function_boundaries=bufferize_function_boundaries,
            copy_before_write=copy_before_write,
            create_deallocs=create_deallocs,
            dialect_filter=dialect_filter,
            dump_alias_sets=dump_alias_sets,
            no_analysis_func_filter=no_analysis_func_filter,
            function_boundary_type_conversion=function_boundary_type_conversion,
            must_infer_memory_space=must_infer_memory_space,
            test_analysis_only=test_analysis_only,
            print_conflicts=print_conflicts,
            unknown_type_conversion=unknown_type_conversion,
        )
        return self

    def opt_reduction_pass(
        self, opt_pass: str = None, test: str = None, test_arg: list[str] = None
    ):
        """A wrapper pass that reduces the file with optimization passes
        Args:
            opt-pass: The optimization passes used for reduction, e.g., symbol-dce
            test: The location of the tester which tests the file interestingness
            test-arg: arguments of the tester
        """
        if test_arg is not None and isinstance(test_arg, (list, tuple)):
            test_arg = ",".join(map(str, test_arg))
        self.add_pass(
            "opt-reduction-pass", opt_pass=opt_pass, test=test, test_arg=test_arg
        )
        return self

    def outline_shape_computation(self):
        """Using shape.func to preserve shape computation

        This pass outlines the shape computation part in high level IR by adding
        shape.func and populate corresponding mapping infoemation into
        ShapeMappingAnalysis. The shape computation part is usually introduced by
        shape reification, and each single dynamic shape is denoted by shape.with_shape.

        There're two main reasons this shape-outline pass is needed:
        1. Many passes don't take shape reification part into consideration.
           Therefore we need to "remove" the shape reification part temporarily for
           these passes.
        2. Sometimes we cannot redo shape reification after converting from dialect
           A to dialect B. Because op-level shape reification is only implemented
           on A.

        Input:

        ```mlir
        func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
          tensor<?x4x?xf32> {
          %c2 = arith.constant 2 : index
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
          %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
          %2 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
          %3 = shape.with_shape %2, %0 : tensor<?x4x?xf32>, tensor<3xindex>
          %4 = shape.value_of %3 : tensor<?x4x?xf32>
          %5 = "test.concat"(%4, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
                tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
          %6 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
          %7 = arith.addi %6, %c2 : index
          %8 = shape.from_extents %7, %c4, %1 : index, index, index
          %9 = shape.with_shape %5, %8 : tensor<?x4x?xf32>, !shape.shape
          %10 = shape.value_of %9 : tensor<?x4x?xf32>
          return %10 : tensor<?x4x?xf32>
        }
        ```

        Output
        ```mlir
        func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
          tensor<?x4x?xf32> {
          %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
          %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
                tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
          return %1 : tensor<?x4x?xf32>
        }
        shape.func private @shape_cal_1(%arg0: tensor<?x4x?xf32>) -> !shape.shape {
          %c2 = arith.constant 2 : index
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
          %1 = get_extent %0, %c2 : tensor<3xindex>, index -> index
          %2 = get_extent %0, %c0 : tensor<3xindex>, index -> index
          %3 = arith.addi %2, %c2 : index
          %4 = from_extents %3, %c4, %1 : index, index, index
          return %4 : !shape.shape
        }
        shape.func private @shape_cal_0(%arg0: tensor<?x4x?xf32>) -> tensor<3xindex> {
          %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
          return %0 : tensor<3xindex>
        }
        ```

        For the above example, the shape computation is inlined in the input IR,
        which is used for two values' (test.abs and test.concat) shape. And the shape
        compuatation part is outlined in the output IR.

        And the shape mapping infomation will be:

        ```
        // ---- Shape Mapping Infomation -----
        // - Shape for: %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_0(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
        // - Shape for: %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_1(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
        ```

        """
        self.add_pass("outline-shape-computation")
        return self

    def post_sparsification_rewrite(
        self,
        enable_runtime_library: bool = None,
        enable_foreach: bool = None,
        enable_convert: bool = None,
    ):
        """Applies sparse tensor rewriting rules after sparsification

        A pass that applies rewriting rules to sparse tensor operations after
        running the actual sparsification pass.

        Args:
            enable-runtime-library: Enable runtime library for manipulating sparse tensors
            enable-foreach: Enable rewriting rules for the foreach operator
            enable-convert: Enable rewriting rules for the convert operator
        """
        self.add_pass(
            "post-sparsification-rewrite",
            enable_runtime_library=enable_runtime_library,
            enable_foreach=enable_foreach,
            enable_convert=enable_convert,
        )
        return self

    def pre_sparsification_rewrite(self):
        """Applies sparse tensor rewriting rules prior to sparsification

        A pass that applies rewriting rules to sparse tensor operations prior
        to running the actual sparsification pass.

        """
        self.add_pass("pre-sparsification-rewrite")
        return self

    def print_ir(self, label: str = None):
        """Print IR on the debug stream

        Print the entire IR on the debug stream. This is meant for debugging
        purposes to inspect the IR at a specific point in the pipeline.

        Args:
            label: Label
        """
        self.add_pass("print-ir", label=label)
        return self

    def print_op_stats(self, json: bool = None):
        """Print statistics of operations
        Args:
            json: print the stats as JSON
        """
        self.add_pass("print-op-stats", json=json)
        return self

    def promote_buffers_to_stack(
        self,
        max_alloc_size_in_bytes: int = None,
        max_rank_of_allocated_memref: int = None,
    ):
        """Promotes heap-based allocations to automatically managed stack-based allocations

        This pass implements a simple algorithm to convert heap-based memory
        allocations to stack-based ones. It uses a built-in heuristic to decide
        whether it makes sense to convert an allocation. Furthermore, dynamic
        shaped buffers that are limited by the rank of the tensor can be
        converted. They are only transformed if they are considered to be small.

        Args:
            max-alloc-size-in-bytes: Maximal size in bytes to promote allocations to stack.
            max-rank-of-allocated-memref: Maximal memref rank to promote dynamic buffers.
        """
        self.add_pass(
            "promote-buffers-to-stack",
            max_alloc_size_in_bytes=max_alloc_size_in_bytes,
            max_rank_of_allocated_memref=max_rank_of_allocated_memref,
        )
        return self

    def reconcile_unrealized_casts(self):
        """Simplify and eliminate unrealized conversion casts

        Eliminate `unrealized_conversion_cast` operations, commonly introduced by
        partial dialect conversions, that transitively convert a value to another
        value of the same type, that is:

        ```
        %0 = "producer.op"() : () -> !type.A
        %1 = unrealized_conversion_cast %0 : !type.A to !type.B
        %2 = unrealized_conversion_cast %1 : !type.B to !type.C
        %3 = unrealized_conversion_cast %2 : !type.C to !type.A
        "consumer.op"(%3) : (!type.A) -> ()
        ```

        Such situations appear when the consumer operation is converted by one pass
        and the producer operation is converted by another pass, each of which
        produces an unrealized cast. This pass can be used to clean up the IR.

        """
        self.add_pass("reconcile-unrealized-casts")
        return self

    def reduction_tree(
        self, traversal_mode: int = None, test: str = None, test_arg: list[str] = None
    ):
        """Reduce the input with reduction-tree algorithm
        Args:
            traversal-mode: The graph traversal mode, the default is single-path mode
            test: The location of the tester which tests the file interestingness
            test-arg: arguments of the tester
        """
        if test_arg is not None and isinstance(test_arg, (list, tuple)):
            test_arg = ",".join(map(str, test_arg))
        self.add_pass(
            "reduction-tree",
            traversal_mode=traversal_mode,
            test=test,
            test_arg=test_arg,
        )
        return self

    def remove_shape_constraints(self):
        """Replace all cstr_ ops with a true witness"""
        self.add_pass("remove-shape-constraints")
        return self

    def resolve_ranked_shaped_type_result_dims(self):
        """Resolve memref.dim of result values of ranked shape type

        The pass resolves memref.dim of result of operations that
        implement the `ReifyRankedShapedTypeOpInterface` in terms of
        shapes of its operands.

        """
        self.add_pass("resolve-ranked-shaped-type-result-dims")
        return self

    def resolve_shaped_type_result_dims(self):
        """Resolve memref.dim of result values

        The pass resolves memref.dim of result of operations that
        implement the `InferShapedTypeOpInterface` or
        `ReifyRankedShapedTypeOpInterface` in terms of shapes of its
        operands.

        """
        self.add_pass("resolve-shaped-type-result-dims")
        return self

    def sccp(self):
        """Sparse Conditional Constant Propagation

        This pass implements a general algorithm for sparse conditional constant
        propagation. This algorithm detects values that are known to be constant and
        optimistically propagates this throughout the IR. Any values proven to be
        constant are replaced, and removed if possible.

        This implementation is based on the algorithm described by Wegman and Zadeck
        in [“Constant Propagation with Conditional Branches”](https://dl.acm.org/doi/10.1145/103135.103136) (1991).

        """
        self.add_pass("sccp")
        return self

    def scf_bufferize(self):
        """Bufferize the scf dialect."""
        self.add_pass("scf-bufferize")
        return self

    def scf_for_loop_canonicalization(self):
        """Canonicalize operations within scf.for loop bodies"""
        self.add_pass("scf-for-loop-canonicalization")
        return self

    def scf_for_loop_peeling(self, skip_partial: bool = None):
        """Peel `for` loops at their upper bounds.
        Args:
            skip-partial: Do not peel loops inside of the last, partial iteration of another already peeled loop.
        """
        self.add_pass("scf-for-loop-peeling", skip_partial=skip_partial)
        return self

    def scf_for_loop_range_folding(self):
        """Fold add/mul ops into loop range"""
        self.add_pass("scf-for-loop-range-folding")
        return self

    def scf_for_loop_specialization(self):
        """Specialize `for` loops for vectorization"""
        self.add_pass("scf-for-loop-specialization")
        return self

    def scf_for_to_while(self):
        """Convert SCF for loops to SCF while loops

        This pass transforms SCF.ForOp operations to SCF.WhileOp. The For loop
        condition is placed in the 'before' region of the while operation, and the
        induction variable incrementation and loop body in the 'after' region.
        The loop carried values of the while op are the induction variable (IV) of
        the for-loop + any iter_args specified for the for-loop.
        Any 'yield' ops in the for-loop are rewritten to additionally yield the
        (incremented) induction variable.

        ```mlir
        # Before:
          scf.for %i = %c0 to %arg1 step %c1 {
            %0 = arith.addi %arg2, %arg2 : i32
            memref.store %0, %arg0[%i] : memref<?xi32>
          }

        # After:
          %0 = scf.while (%i = %c0) : (index) -> index {
            %1 = arith.cmpi slt, %i, %arg1 : index
            scf.condition(%1) %i : index
          } do {
          ^bb0(%i: index):
            %1 = arith.addi %i, %c1 : index
            %2 = arith.addi %arg2, %arg2 : i32
            memref.store %2, %arg0[%i] : memref<?xi32>
            scf.yield %1 : index
          }
        ```

        """
        self.add_pass("scf-for-to-while")
        return self

    def scf_parallel_loop_fusion(self):
        """Fuse adjacent parallel loops"""
        self.add_pass("scf-parallel-loop-fusion")
        return self

    def scf_parallel_loop_specialization(self):
        """Specialize parallel loops for vectorization"""
        self.add_pass("scf-parallel-loop-specialization")
        return self

    def scf_parallel_loop_tiling(
        self, parallel_loop_tile_sizes: list[int] = None, no_min_max_bounds: bool = None
    ):
        """Tile parallel loops
        Args:
            parallel-loop-tile-sizes: Factors to tile parallel loops by
            no-min-max-bounds: Perform tiling with fixed upper bound with inbound check inside the internal loops
        """
        if parallel_loop_tile_sizes is not None and isinstance(
            parallel_loop_tile_sizes, (list, tuple)
        ):
            parallel_loop_tile_sizes = ",".join(map(str, parallel_loop_tile_sizes))
        self.add_pass(
            "scf-parallel-loop-tiling",
            parallel_loop_tile_sizes=parallel_loop_tile_sizes,
            no_min_max_bounds=no_min_max_bounds,
        )
        return self

    def shape_bufferize(self):
        """Bufferize the shape dialect."""
        self.add_pass("shape-bufferize")
        return self

    def shape_to_shape_lowering(self):
        """Legalize Shape dialect to be convertible to Arith"""
        self.add_pass("shape-to-shape-lowering")
        return self

    def snapshot_op_locations(self, filename: str = None, tag: str = None):
        """Generate new locations from the current IR

        This pass allows for generating new locations from the IR during any stage
        of compilation, by snapshotting the IR to a file and using that file to
        generate new locations for the operations.

        Depending on the value of the `tag` option, different resulting locations
        may be generated:

        * If unset, the original location of the operation is replaced.

        Example:

        ```mlir
        // old:
        ... loc("original_source.cpp":1:1)

        // new:
        ... loc("snapshot_source.mlir":10:10)
        ```

        * If set, the new location is fused with the original location in the form
        of a [`Name Location`](Dialects/Builtin.md/#nameloc) with the specified tag.

        Example:

        ```mlir
        // old:
        ... loc("original_source.cpp":1:1)

        // new:
        ... loc(fused["original_source.cpp":1:1, "snapshot"("snapshot_source.mlir":10:10)])
        ```

        Args:
            filename: The filename to print the generated IR
            tag: A tag to use when fusing the new locations with the original. If unset, the locations are replaced.
        """
        self.add_pass("snapshot-op-locations", filename=filename, tag=tag)
        return self

    def sparse_buffer_rewrite(self, enable_buffer_initialization: bool = None):
        """Rewrite sparse primitives on buffers to actual code

        A pass that rewrites sparse primitives on buffers to the MLIR implementation
        of the primitives. For example, sparse_tensor.sort operator is implemented
        in this pass.

        Args:
            enable-buffer-initialization: Enable zero-initialization of the memory buffers
        """
        self.add_pass(
            "sparse-buffer-rewrite",
            enable_buffer_initialization=enable_buffer_initialization,
        )
        return self

    def sparse_gpu_codegen(self, num_threads: int = None):
        """Generates GPU code during sparsification

        Enables sparse compiler to use GPU acceleration.

        Args:
            num_threads: Sets the number of GPU threads
        """
        self.add_pass("sparse-gpu-codegen", num_threads=num_threads)
        return self

    def sparse_storage_specifier_to_llvm(self):
        """Lower sparse storage specifer to llvm structure

        This pass rewrites sparse tensor storage specifier-related operations into
        LLVMDialect, and converts sparse tensor storage specifier into an llvm.struct.

        Example of the conversion:
        ```mlir
        Before:
          %0 = sparse_tensor.storage_specifier.get %arg0 dim_sz at 0
          : !sparse_tensor.storage_specifier<#CSR> to i64

        After:
          %0 = llvm.extractvalue %arg0[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
        ```

        """
        self.add_pass("sparse-storage-specifier-to-llvm")
        return self

    def sparse_tensor_codegen(
        self,
        enable_buffer_initialization: bool = None,
        create_sparse_deallocs: bool = None,
    ):
        """Convert sparse tensors and primitives to actual code

        A pass that converts sparse tensor types and primitives to actual
        compiler visible buffers and compiler IR that implements these
        primitives on the selected sparse tensor storage schemes.

        This pass provides an alternative to the SparseTensorConversion pass,
        eliminating the dependence on a runtime support library, and providing
        much more opportunities for subsequent compiler optimization of the
        generated code.

        Example of the conversion:

        ```mlir
          Before:
            func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
              %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
                 : tensor<8x8xf32, #CSR> to memref<?xindex>
              return %0 : memref<?xindex>
            }

          After:
            func.func @foo(%arg0: memref<2xindex>,
                           %arg1: memref<3xindex>,
                           %arg2: memref<?xindex>,
                           %arg3: memref<?xindex>,
                           %arg4: memref<?xf32>) -> memref<?xindex> {
              return %arg2 : memref<?xindex>
            }
        ```

        Args:
            enable-buffer-initialization: Enable zero-initialization of the memory buffers
            create-sparse-deallocs: Specify if the temporary buffers created by the sparse compiler should be deallocated. For compatibility with core bufferization passes. This option is only used when enable-runtime-library=false. See also create-deallocs for BufferizationOption.
        """
        self.add_pass(
            "sparse-tensor-codegen",
            enable_buffer_initialization=enable_buffer_initialization,
            create_sparse_deallocs=create_sparse_deallocs,
        )
        return self

    def sparse_tensor_conversion(self, s2s_strategy: int = None):
        """Convert sparse tensors and primitives to library calls

        A pass that converts sparse tensor primitives into calls into a runtime
        support library. Sparse tensor types are converted into opaque pointers
        to the underlying sparse storage schemes.

        The use of opaque pointers together with runtime support library keeps
        the conversion relatively simple, but at the expense of IR opacity,
        which obscures opportunities for subsequent optimization of the IR.
        An alternative is provided by the SparseTensorCodegen pass.

        Example of the conversion:

        ```mlir
          Before:
            func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
              %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
                 : tensor<8x8xf32, #CSR> to memref<?xindex>
              return %0 : memref<?xindex>
            }

          After:
            func.func @foo(%arg0: !llvm.ptr<i8>) -> memref<?xindex> {
              %c1 = arith.constant 1 : index
              %0 = call @sparsePointers0(%arg0, %c1)
                 : (!llvm.ptr<i8>, index) -> memref<?xindex>
              return %0 : memref<?xindex>
            }
        ```

        Args:
            s2s-strategy: Set the strategy for sparse-to-sparse conversion
        """
        self.add_pass("sparse-tensor-conversion", s2s_strategy=s2s_strategy)
        return self

    def sparse_vectorization(
        self,
        vl: int = None,
        enable_vla_vectorization: bool = None,
        enable_simd_index32: bool = None,
    ):
        """Vectorizes loops after sparsification

        A pass that converts loops after sparsification into vector loops.
        The vector dialect is used as target to provide an architectural
        neutral way of exploiting any platform that supports SIMD instructions.

        The vector length (viz. `vl`) describes the number of packed data elements
        (e.g. both vector<16xf32> and vector<16xf64> have a vector length of 16 even
        though the actual bitwidths differ). A small multiple of the actual lengths
        supported in hardware typically results in efficient SIMD code, since the
        backend will map longer vectors to multiple vector registers, thereby
        effectively unrolling an addition level within the generated for-loop.

        Example of the conversion:

        ```mlir
          Before:
            %3 = memref.load %2[] : memref<f32>
            %4 = scf.for %arg3 = %c0 to %c1024 step %c1 iter_args(%arg4 = %3) -> (f32) {
              %6 = memref.load %0[%arg3] : memref<?xf32>
              %7 = memref.load %1[%arg3] : memref<1024xf32>
              %8 = arith.mulf %6, %7 : f32
              %9 = arith.addf %arg4, %8 : f32
              scf.yield %9 : f32
            }
            memref.store %4, %2[] : memref<f32>

          After:
            %3 = memref.load %2[] : memref<f32>
            %4 = vector.insertelement %3, %cst[%c0 : index] : vector<32xf32>
            %5 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %4) -> (vector<32xf32>) {
              %8 = vector.load %0[%arg3] : memref<?xf32>, vector<32xf32>
              %9 = vector.load %1[%arg3] : memref<1024xf32>, vector<32xf32>
              %10 = arith.mulf %8, %9 : vector<32xf32>
              %11 = arith.addf %arg4, %10 : vector<32xf32>
              scf.yield %11 : vector<32xf32>
            }
            %6 = vector.reduction <add>, %5 : vector<32xf32> into f32
            memref.store %6, %2[] : memref<f32>
        ```

        Args:
            vl: Set the vector length (use 0 to disable vectorization)
            enable-vla-vectorization: Enable vector length agnostic vectorization
            enable-simd-index32: Enable i32 indexing into vectors (for efficient gather/scatter)
        """
        self.add_pass(
            "sparse-vectorization",
            vl=vl,
            enable_vla_vectorization=enable_vla_vectorization,
            enable_simd_index32=enable_simd_index32,
        )
        return self

    def sparsification(
        self,
        enable_index_reduction: bool = None,
        parallelization_strategy: "SparseParallelizationStrategy" = None,
        enable_gpu_libgen: bool = None,
        enable_runtime_library: bool = None,
    ):
        """Automatically generate sparse tensor code from sparse tensor types

        A pass that implements the core functionality of a **sparse compiler**.
        Each Linalg operation (MLIR's tensor index notation) that operates on
        sparse tensor types is converted into code in which the sparsity is
        explicit both in terms of co-iterating looping logic as well as
        selected sparse storage schemes.

        See the `SparseTensor` dialect documentation for more background.

        Example input:

        ```mlir
        #matvec = {
          indexing_maps = [
            affine_map<(i,j) -> (i,j)>, // A
            affine_map<(i,j) -> (j)>,   // b
            affine_map<(i,j) -> (i)>    // x (out)
          ],
          iterator_types = ["parallel", "reduction"],
          doc = "X(i) += A(i,j) * B(j)"
        }

        // Multiply a sparse matrix A with a dense vector b into a dense vector x.
        func.func @kernel_matvec(%arga: tensor<?x?xf64, #SparseMatrix>,
                                 %argb: tensor<?xf64>,
                                 %argx: tensor<?xf64>) -> tensor<?xf64> {
          %0 = linalg.generic #matvec
            ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?xf64>)
            outs(%argx: tensor<?xf64>) {
            ^bb(%a: f64, %b: f64, %x: f64):
              %0 = arith.mulf %a, %b : f64
              %1 = arith.addf %x, %0 : f64
              linalg.yield %1 : f64
          } -> tensor<?xf64>
          return %0 : tensor<?xf64>
        }
        ```

        Args:
            enable-index-reduction: Enable dependent index reduction based algorithm to handle non-trivial index expressions on sparse inputs (experimental features)
            parallelization-strategy: Set the parallelization strategy
            enable-gpu-libgen: Enable GPU acceleration by means of direct library calls (like cuSPARSE)
            enable-runtime-library: Enable runtime library for manipulating sparse tensors
        """
        self.add_pass(
            "sparsification",
            enable_index_reduction=enable_index_reduction,
            parallelization_strategy=parallelization_strategy,
            enable_gpu_libgen=enable_gpu_libgen,
            enable_runtime_library=enable_runtime_library,
        )
        return self

    def spirv_canonicalize_gl(self):
        """Canonicalize GLSL ops

        Pass to run canoncalization patterns that involve GL ops.
        These patterns cannot be run in default canonicalization because GL ops
        aren't always available. So they should be involed specifically when needed.

        """
        self.add_pass("spirv-canonicalize-gl")
        return self

    def spirv_lower_abi_attrs(self):
        """Decorate SPIR-V composite type with layout info

        Operation pass that lowers the ABI attributes specified during
        SPIR-V Lowering. Specifically:
        1. Creates the global variables for arguments of entry point function using
          the specification in the `spirv.interface_var_abi` attribute for each
          argument.
        2. Inserts the EntryPointOp and the ExecutionModeOp for entry point
          functions using the specification in the `spirv.entry_point_abi`
          attribute.

        """
        self.add_pass("spirv-lower-abi-attrs")
        return self

    def spirv_rewrite_inserts(self):
        """Rewrite sequential chains of `spirv.CompositeInsert` operations into `spirv.CompositeConstruct` operations"""
        self.add_pass("spirv-rewrite-inserts")
        return self

    def spirv_unify_aliased_resource(self):
        """Unify access of multiple aliased resources into access of one single resource"""
        self.add_pass("spirv-unify-aliased-resource")
        return self

    def spirv_update_vce(self):
        """Deduce and attach minimal (version, capabilities, extensions) requirements to spirv.module ops

        Operation pass that deduces and attaches the minimal version/
        capabilities/extensions requirements for spirv.module ops.
        For each spirv.module op, this pass requires a `spirv.target_env` attribute
        on it or an enclosing module-like op to drive the deduction. The reason is
        that an op can be enabled by multiple extensions/capabilities. So we need
        to know which one to pick. `spirv.target_env` gives the hard limit as for
        what the target environment can support; this pass deduces what are
        actually needed for a specific spirv.module op.

        """
        self.add_pass("spirv-update-vce")
        return self

    def spirv_webgpu_prepare(self):
        """Prepare SPIR-V to target WebGPU by expanding unsupported ops and replacing with supported ones"""
        self.add_pass("spirv-webgpu-prepare")
        return self

    def sroa(self):
        """Scalar Replacement of Aggregates

        Scalar Replacement of Aggregates. Replaces allocations of aggregates into
        independant allocations of its elements.

        Allocators must implement `DestructurableAllocationOpInterface` to provide
        the list of memory slots for which destructuring should be attempted.

        This pass will only be applied if all accessors of the aggregate implement
        the `DestructurableAccessorOpInterface`. If the accessors provide a view
        into the struct, users of the view must ensure it is used in a type-safe
        manner and within bounds by implementing `TypeSafeOpInterface`.

        """
        self.add_pass("sroa")
        return self

    def strip_debuginfo(self):
        """Strip debug info from all operations

        This pass strips the IR of any location information, by replacing all
        operation locations with [`unknown`](Dialects/Builtin.md/#unknownloc).

        """
        self.add_pass("strip-debuginfo")
        return self

    def symbol_dce(self):
        """Eliminate dead symbols

        This pass deletes all symbols that are found to be unreachable. This is done
        by computing the set of operations that are known to be live, propagating
        that liveness to other symbols, and then deleting all symbols that are not
        within this live set. Live symbols are those that have a
        [visibility](SymbolsAndSymbolTables.md/#symbol-visibility) that extends
        beyond the IR, e.g. `public`, or those that are referenced by live symbols
        or other non-Symbol operations.

        For example, consider the following input:

        ```mlir
        func.func private @dead_private_function()
        func.func private @live_private_function()

        // Note: The `public` isn't necessary here, as this is the default.
        func.func public @public_function() {
          "foo.return"() {uses = [@live_private_function]} : () -> ()
        }
        ```

        A known live function, `public_function`, contains a reference to an
        otherwise non-live function `live_private_function`. After running
        `symbol-dce`, only these two symbols should remain, as the final symbol
        `dead_private_function` is not visible outside of the current IR and there
        are no links to known-live operations. After running, we get the expected:

        ```mlir
        func.func private @live_private_function()

        func.func public @public_function() {
          "foo.return"() {uses = [@live_private_function]} : () -> ()
        }
        ```

        See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more
        information on `Symbols`.

        """
        self.add_pass("symbol-dce")
        return self

    def symbol_privatize(self, exclude: list[str] = None):
        """Mark symbols private

        This pass marks all top-level symbols of the operation run as `private`
        except if listed in `exclude` pass option.

        Args:
            exclude: Comma separated list of symbols that should not be marked private
        """
        if exclude is not None and isinstance(exclude, (list, tuple)):
            exclude = ",".join(map(str, exclude))
        self.add_pass("symbol-privatize", exclude=exclude)
        return self

    def tensor_bufferize(self):
        """Bufferize the `tensor` dialect"""
        self.add_pass("tensor-bufferize")
        return self

    def test_scf_parallel_loop_collapsing(
        self,
        collapsed_indices_0: list[int] = None,
        collapsed_indices_1: list[int] = None,
        collapsed_indices_2: list[int] = None,
    ):
        """Test parallel loops collapsing transformation

          This pass is purely for testing the scf::collapseParallelLoops
          transformation. The transformation does not have opinions on how a
          parallel loop should be collapsed, so this pass is structured for the
          common case on GPUs of collapsing to a 3d parallel loop. 3 lists can be
          provided to collapsed-indices-{0,1,2} to represent how the loop should be
          collapsed and must reference evrey iterator in the original parallel loop.

        ```mlir
        # Before:
        scf.parallel (%arg0, %arg1)
                     = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
          "test.sink"(%5, %3) : (index, index) -> ()
          scf.yield
        }

        # After:
        scf.parallel (%arg0) = (%c0) to (%c4) step (%c1) {
          %0 = arith.remsi %arg0, %c2 : index
          %1 = arith.divsi %arg0, %c2 : index
          %2 = arith.muli %0, %c7 : index
          %3 = arith.addi %2, %c3 : index
          %4 = arith.muli %1, %c7 : index
          %5 = arith.addi %4, %c3 : index
          "test.sink"(%5, %3) : (index, index) -> ()
        }
        ```

        Args:
            collapsed-indices-0: Which loop indices to combine 0th loop index
            collapsed-indices-1: Which loop indices to combine into the position 1 loop index
            collapsed-indices-2: Which loop indices to combine into the position 2 loop index
        """
        if collapsed_indices_0 is not None and isinstance(
            collapsed_indices_0, (list, tuple)
        ):
            collapsed_indices_0 = ",".join(map(str, collapsed_indices_0))
        if collapsed_indices_1 is not None and isinstance(
            collapsed_indices_1, (list, tuple)
        ):
            collapsed_indices_1 = ",".join(map(str, collapsed_indices_1))
        if collapsed_indices_2 is not None and isinstance(
            collapsed_indices_2, (list, tuple)
        ):
            collapsed_indices_2 = ",".join(map(str, collapsed_indices_2))
        self.add_pass(
            "test-scf-parallel-loop-collapsing",
            collapsed_indices_0=collapsed_indices_0,
            collapsed_indices_1=collapsed_indices_1,
            collapsed_indices_2=collapsed_indices_2,
        )
        return self

    def topological_sort(self):
        """Sort regions without SSA dominance in topological order

        Recursively sorts all nested regions without SSA dominance in topological
        order. The main purpose is readability, as well as potentially processing of
        certain transformations and analyses. The function sorts the operations in
        all nested regions such that, as much as possible, all users appear after
        their producers.

        This sort is stable. If the block is already topologically sorted, the IR
        is not changed. Operations that form a cycle are moved to the end of the
        regions in a stable order.

        """
        self.add_pass("topological-sort")
        return self

    def tosa_infer_shapes(self):
        """Propagate shapes across TOSA operations

        Pass that uses operand types and propagates shapes to TOSA operations.
        This includes legalizing rankless and dynamic shapes towards static.

        """
        self.add_pass("tosa-infer-shapes")
        return self

    def tosa_layerwise_constant_fold(self):
        """Fold layerwise operations on constant tensors

        Pass that enables folding of full-layer operations on constant tensors.

        """
        self.add_pass("tosa-layerwise-constant-fold")
        return self

    def tosa_make_broadcastable(self):
        """TOSA rank Reshape to enable Broadcasting

        Pass that enables broadcast by making all input arrays have the same
        number of dimensions. Insert RESHAPE operations to prepend dimensions
        of size one until the number of dimensions is equal. Implements
        approach similar to step 1 of Numpy 4-step broadcasting:
        https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting

        """
        self.add_pass("tosa-make-broadcastable")
        return self

    def tosa_optional_decompositions(self):
        """Applies Tosa operations optional decompositions

        Pass to apply the Tosa operations decompositions
        exposed as populate functions in include/mlir/Dialect/Tosa/Transforms/Passes.h

        """
        self.add_pass("tosa-optional-decompositions")
        return self

    def tosa_to_arith(
        self, include_apply_rescale: bool = None, use_32_bit: bool = None
    ):
        """Lower TOSA to the Arith dialect

        Pass that converts TOSA operations to the equivalent operations using the
        operations in the Arith dialect. The ApplyScale operator is optionally
        included as it is often preserved until the final invocation.

        Args:
            include-apply-rescale: Whether to include the lowering for tosa.apply_rescale to arith
            use-32-bit: Whether to prioritze lowering to 32-bit operations
        """
        self.add_pass(
            "tosa-to-arith",
            include_apply_rescale=include_apply_rescale,
            use_32_bit=use_32_bit,
        )
        return self

    def tosa_to_scf(self):
        """Lower TOSA to the SCF dialect

        Pass that converts TOSA's control flow operations to the equivalent SCF
        operations.

        """
        self.add_pass("tosa-to-scf")
        return self

    def tosa_to_tensor(self):
        """Lower TOSA to the Tensor dialect

        Pass that converts TOSA operations to the equivalent operations using the
        operations in the Tensor dialect.

        """
        self.add_pass("tosa-to-tensor")
        return self

    def tosa_validate(
        self,
        profile: str = None,
        strict_op_spec_alignment: bool = None,
        level: str = None,
    ):
        """Validates TOSA dialect

        This pass validates if input TOSA operations match the specification for given
        criteria, e.g. TOSA profile.

        Args:
            profile: Validate if operations match for the given profile
            strict-op-spec-alignment: Verify if the properties of certain operations align the spec requirement
            level: Validate if operator parameters are within specfication for the given level
        """
        self.add_pass(
            "tosa-validate",
            profile=profile,
            strict_op_spec_alignment=strict_op_spec_alignment,
            level=level,
        )
        return self

    def transform_dialect_check_uses(self):
        """warn about potential use-after-free in the transform dialect

        This pass analyzes operations from the transform dialect and its extensions
        and warns if a transform IR value may be used by an operation after it was
        "freed" by some other operation, as described by side effects on the
        `TransformMappingResource`. This statically detects situations that lead to
        errors when interpreting the Transform IR.

        The pass is capable of handling branching control flow and reports all
        _potential_ use-after-free situations, e.g., a may-use-after-free is
        reported if at least one of the control flow paths between the definition of
        a value and its use contains an operation with a "free" effect on the
        `TransformMappingResource`. It does not currently perform an SCCP-style data
        flow analysis to prove that some branches are not taken, however, SCCP and
        other control flow simplifications can be performed on the transform IR
        prior to this pass provided that transform ops implement the relevant
        control flow interfaces.

        """
        self.add_pass("transform-dialect-check-uses")
        return self

    def transform_infer_effects(self):
        """infer transform side effects for symbols

        This pass analyzes the definitions of transform dialect callable symbol
        operations, such as `transform.named_sequence`, and annotates the symbol
        arguments with attributes indicating the side effects that the nested
        operations have on them.

        """
        self.add_pass("transform-infer-effects")
        return self

    def vector_bufferize(self):
        """Bufferize Vector dialect ops"""
        self.add_pass("vector-bufferize")
        return self

    def view_op_graph(
        self,
        max_label_len: int = None,
        print_attrs: bool = None,
        print_control_flow_edges: bool = None,
        print_data_flow_edges: bool = None,
        print_result_types: bool = None,
    ):
        """Print Graphviz visualization of an operation

        This pass prints a Graphviz graph of a module.

        - Operations are represented as nodes;
        - Uses (data flow) as edges;
        - Control flow as dashed edges;
        - Regions/blocks as subgraphs.

        By default, only data flow edges are printed.

        Note: See https://www.graphviz.org/doc/info/lang.html for more information
        about the Graphviz DOT language.

        Args:
            max-label-len: Limit attribute/type length to number of chars
            print-attrs: Print attributes of operations
            print-control-flow-edges: Print control flow edges
            print-data-flow-edges: Print data flow edges
            print-result-types: Print result types of operations
        """
        self.add_pass(
            "view-op-graph",
            max_label_len=max_label_len,
            print_attrs=print_attrs,
            print_control_flow_edges=print_control_flow_edges,
            print_data_flow_edges=print_data_flow_edges,
            print_result_types=print_result_types,
        )
        return self


affine_data_copy_generate = Pipeline().affine_data_copy_generate
affine_expand_index_ops = Pipeline().affine_expand_index_ops
affine_loop_coalescing = Pipeline().affine_loop_coalescing
affine_loop_fusion = Pipeline().affine_loop_fusion
affine_loop_invariant_code_motion = Pipeline().affine_loop_invariant_code_motion
affine_loop_normalize = Pipeline().affine_loop_normalize
affine_loop_tile = Pipeline().affine_loop_tile
affine_loop_unroll = Pipeline().affine_loop_unroll
affine_loop_unroll_jam = Pipeline().affine_loop_unroll_jam
affine_parallelize = Pipeline().affine_parallelize
affine_pipeline_data_transfer = Pipeline().affine_pipeline_data_transfer
affine_scalrep = Pipeline().affine_scalrep
affine_simplify_structures = Pipeline().affine_simplify_structures
affine_super_vectorize = Pipeline().affine_super_vectorize
allocate_arm_sme_tiles = Pipeline().allocate_arm_sme_tiles
amdgpu_emulate_atomics = Pipeline().amdgpu_emulate_atomics
arith_bufferize = Pipeline().arith_bufferize
arith_emulate_unsupported_floats = Pipeline().arith_emulate_unsupported_floats
arith_emulate_wide_int = Pipeline().arith_emulate_wide_int
arith_expand = Pipeline().arith_expand
arith_int_narrowing = Pipeline().arith_int_narrowing
arith_unsigned_when_equivalent = Pipeline().arith_unsigned_when_equivalent
arm_neon_2d_to_intr = Pipeline().arm_neon_2d_to_intr
async_func_to_async_runtime = Pipeline().async_func_to_async_runtime
async_parallel_for = Pipeline().async_parallel_for
async_runtime_policy_based_ref_counting = (
    Pipeline().async_runtime_policy_based_ref_counting
)
async_runtime_ref_counting = Pipeline().async_runtime_ref_counting
async_runtime_ref_counting_opt = Pipeline().async_runtime_ref_counting_opt
async_to_async_runtime = Pipeline().async_to_async_runtime
buffer_deallocation = Pipeline().buffer_deallocation
buffer_hoisting = Pipeline().buffer_hoisting
buffer_loop_hoisting = Pipeline().buffer_loop_hoisting
buffer_results_to_out_params = Pipeline().buffer_results_to_out_params
bufferization_bufferize = Pipeline().bufferization_bufferize
canonicalize = Pipeline().canonicalize
control_flow_sink = Pipeline().control_flow_sink
convert_amdgpu_to_rocdl = Pipeline().convert_amdgpu_to_rocdl
convert_arith_to_llvm = Pipeline().convert_arith_to_llvm
convert_arith_to_spirv = Pipeline().convert_arith_to_spirv
convert_async_to_llvm = Pipeline().convert_async_to_llvm
convert_bufferization_to_memref = Pipeline().convert_bufferization_to_memref
convert_cf_to_llvm = Pipeline().convert_cf_to_llvm
convert_cf_to_spirv = Pipeline().convert_cf_to_spirv
convert_complex_to_libm = Pipeline().convert_complex_to_libm
convert_complex_to_llvm = Pipeline().convert_complex_to_llvm
convert_complex_to_spirv = Pipeline().convert_complex_to_spirv
convert_complex_to_standard = Pipeline().convert_complex_to_standard
convert_elementwise_to_linalg = Pipeline().convert_elementwise_to_linalg
convert_func_to_llvm = Pipeline().convert_func_to_llvm
convert_func_to_spirv = Pipeline().convert_func_to_spirv
convert_gpu_launch_to_vulkan_launch = Pipeline().convert_gpu_launch_to_vulkan_launch
convert_gpu_to_nvvm = Pipeline().convert_gpu_to_nvvm
convert_gpu_to_rocdl = Pipeline().convert_gpu_to_rocdl
convert_gpu_to_spirv = Pipeline().convert_gpu_to_spirv
convert_index_to_llvm = Pipeline().convert_index_to_llvm
convert_linalg_to_affine_loops = Pipeline().convert_linalg_to_affine_loops
convert_linalg_to_loops = Pipeline().convert_linalg_to_loops
convert_linalg_to_parallel_loops = Pipeline().convert_linalg_to_parallel_loops
convert_linalg_to_std = Pipeline().convert_linalg_to_std
convert_math_to_funcs = Pipeline().convert_math_to_funcs
convert_math_to_libm = Pipeline().convert_math_to_libm
convert_math_to_llvm = Pipeline().convert_math_to_llvm
convert_math_to_spirv = Pipeline().convert_math_to_spirv
convert_memref_to_spirv = Pipeline().convert_memref_to_spirv
convert_nvgpu_to_nvvm = Pipeline().convert_nvgpu_to_nvvm
convert_nvvm_to_llvm = Pipeline().convert_nvvm_to_llvm
convert_openacc_to_scf = Pipeline().convert_openacc_to_scf
convert_openmp_to_llvm = Pipeline().convert_openmp_to_llvm
convert_parallel_loops_to_gpu = Pipeline().convert_parallel_loops_to_gpu
convert_pdl_to_pdl_interp = Pipeline().convert_pdl_to_pdl_interp
convert_scf_to_cf = Pipeline().convert_scf_to_cf
convert_scf_to_openmp = Pipeline().convert_scf_to_openmp
convert_scf_to_spirv = Pipeline().convert_scf_to_spirv
convert_shape_constraints = Pipeline().convert_shape_constraints
convert_shape_to_std = Pipeline().convert_shape_to_std
convert_spirv_to_llvm = Pipeline().convert_spirv_to_llvm
convert_tensor_to_linalg = Pipeline().convert_tensor_to_linalg
convert_tensor_to_spirv = Pipeline().convert_tensor_to_spirv
convert_ub_to_llvm = Pipeline().convert_ub_to_llvm
convert_ub_to_spirv = Pipeline().convert_ub_to_spirv
convert_vector_to_arm_sme = Pipeline().convert_vector_to_arm_sme
convert_vector_to_gpu = Pipeline().convert_vector_to_gpu
convert_vector_to_llvm = Pipeline().convert_vector_to_llvm
convert_vector_to_scf = Pipeline().convert_vector_to_scf
convert_vector_to_spirv = Pipeline().convert_vector_to_spirv
cse = Pipeline().cse
decorate_spirv_composite_type_layout = Pipeline().decorate_spirv_composite_type_layout
drop_equivalent_buffer_results = Pipeline().drop_equivalent_buffer_results
duplicate_function_elimination = Pipeline().duplicate_function_elimination
eliminate_empty_tensors = Pipeline().eliminate_empty_tensors
empty_tensor_to_alloc_tensor = Pipeline().empty_tensor_to_alloc_tensor
enable_arm_streaming = Pipeline().enable_arm_streaming
ensure_debug_info_scope_on_llvm_func = Pipeline().ensure_debug_info_scope_on_llvm_func
expand_strided_metadata = Pipeline().expand_strided_metadata
finalize_memref_to_llvm = Pipeline().finalize_memref_to_llvm
finalizing_bufferize = Pipeline().finalizing_bufferize
fold_memref_alias_ops = Pipeline().fold_memref_alias_ops
fold_tensor_subset_ops = Pipeline().fold_tensor_subset_ops
func_bufferize = Pipeline().func_bufferize
generate_runtime_verification = Pipeline().generate_runtime_verification
gpu_async_region = Pipeline().gpu_async_region
gpu_kernel_outlining = Pipeline().gpu_kernel_outlining
gpu_launch_sink_index_computations = Pipeline().gpu_launch_sink_index_computations
gpu_map_parallel_loops = Pipeline().gpu_map_parallel_loops
gpu_to_llvm = Pipeline().gpu_to_llvm
inline = Pipeline().inline
int_range_optimizations = Pipeline().int_range_optimizations
launch_func_to_vulkan = Pipeline().launch_func_to_vulkan
linalg_bufferize = Pipeline().linalg_bufferize
linalg_fold_unit_extent_dims = Pipeline().linalg_fold_unit_extent_dims
linalg_fuse_elementwise_ops = Pipeline().linalg_fuse_elementwise_ops
linalg_generalize_named_ops = Pipeline().linalg_generalize_named_ops
linalg_inline_scalar_operands = Pipeline().linalg_inline_scalar_operands
linalg_named_op_conversion = Pipeline().linalg_named_op_conversion
llvm_legalize_for_export = Pipeline().llvm_legalize_for_export
llvm_optimize_for_nvvm_target = Pipeline().llvm_optimize_for_nvvm_target
llvm_request_c_wrappers = Pipeline().llvm_request_c_wrappers
llvm_type_consistency = Pipeline().llvm_type_consistency
loop_invariant_code_motion = Pipeline().loop_invariant_code_motion
lower_affine = Pipeline().lower_affine
lower_host_to_llvm = Pipeline().lower_host_to_llvm
lower_vector_mask = Pipeline().lower_vector_mask
map_memref_spirv_storage_class = Pipeline().map_memref_spirv_storage_class
math_uplift_to_fma = Pipeline().math_uplift_to_fma
mem2reg = Pipeline().mem2reg
memref_emulate_wide_int = Pipeline().memref_emulate_wide_int
memref_expand = Pipeline().memref_expand
normalize_memrefs = Pipeline().normalize_memrefs
nvgpu_optimize_shared_memory = Pipeline().nvgpu_optimize_shared_memory
one_shot_bufferize = Pipeline().one_shot_bufferize
opt_reduction_pass = Pipeline().opt_reduction_pass
outline_shape_computation = Pipeline().outline_shape_computation
post_sparsification_rewrite = Pipeline().post_sparsification_rewrite
pre_sparsification_rewrite = Pipeline().pre_sparsification_rewrite
print_ir = Pipeline().print_ir
print_op_stats = Pipeline().print_op_stats
promote_buffers_to_stack = Pipeline().promote_buffers_to_stack
reconcile_unrealized_casts = Pipeline().reconcile_unrealized_casts
reduction_tree = Pipeline().reduction_tree
remove_shape_constraints = Pipeline().remove_shape_constraints
resolve_ranked_shaped_type_result_dims = (
    Pipeline().resolve_ranked_shaped_type_result_dims
)
resolve_shaped_type_result_dims = Pipeline().resolve_shaped_type_result_dims
sccp = Pipeline().sccp
scf_bufferize = Pipeline().scf_bufferize
scf_for_loop_canonicalization = Pipeline().scf_for_loop_canonicalization
scf_for_loop_peeling = Pipeline().scf_for_loop_peeling
scf_for_loop_range_folding = Pipeline().scf_for_loop_range_folding
scf_for_loop_specialization = Pipeline().scf_for_loop_specialization
scf_for_to_while = Pipeline().scf_for_to_while
scf_parallel_loop_fusion = Pipeline().scf_parallel_loop_fusion
scf_parallel_loop_specialization = Pipeline().scf_parallel_loop_specialization
scf_parallel_loop_tiling = Pipeline().scf_parallel_loop_tiling
shape_bufferize = Pipeline().shape_bufferize
shape_to_shape_lowering = Pipeline().shape_to_shape_lowering
snapshot_op_locations = Pipeline().snapshot_op_locations
sparse_buffer_rewrite = Pipeline().sparse_buffer_rewrite
sparse_gpu_codegen = Pipeline().sparse_gpu_codegen
sparse_storage_specifier_to_llvm = Pipeline().sparse_storage_specifier_to_llvm
sparse_tensor_codegen = Pipeline().sparse_tensor_codegen
sparse_tensor_conversion = Pipeline().sparse_tensor_conversion
sparse_vectorization = Pipeline().sparse_vectorization
sparsification = Pipeline().sparsification
spirv_canonicalize_gl = Pipeline().spirv_canonicalize_gl
spirv_lower_abi_attrs = Pipeline().spirv_lower_abi_attrs
spirv_rewrite_inserts = Pipeline().spirv_rewrite_inserts
spirv_unify_aliased_resource = Pipeline().spirv_unify_aliased_resource
spirv_update_vce = Pipeline().spirv_update_vce
spirv_webgpu_prepare = Pipeline().spirv_webgpu_prepare
sroa = Pipeline().sroa
strip_debuginfo = Pipeline().strip_debuginfo
symbol_dce = Pipeline().symbol_dce
symbol_privatize = Pipeline().symbol_privatize
tensor_bufferize = Pipeline().tensor_bufferize
test_scf_parallel_loop_collapsing = Pipeline().test_scf_parallel_loop_collapsing
topological_sort = Pipeline().topological_sort
tosa_infer_shapes = Pipeline().tosa_infer_shapes
tosa_layerwise_constant_fold = Pipeline().tosa_layerwise_constant_fold
tosa_make_broadcastable = Pipeline().tosa_make_broadcastable
tosa_optional_decompositions = Pipeline().tosa_optional_decompositions
tosa_to_arith = Pipeline().tosa_to_arith
tosa_to_scf = Pipeline().tosa_to_scf
tosa_to_tensor = Pipeline().tosa_to_tensor
tosa_validate = Pipeline().tosa_validate
transform_dialect_check_uses = Pipeline().transform_dialect_check_uses
transform_infer_effects = Pipeline().transform_infer_effects
vector_bufferize = Pipeline().vector_bufferize
view_op_graph = Pipeline().view_op_graph
