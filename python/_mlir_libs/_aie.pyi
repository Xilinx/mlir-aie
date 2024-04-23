# generated with PYTHONPATH=python pybind11-stubgen --print-invalid-expressions-as-is aie._mlir_libs._aie

from __future__ import annotations

from .ir import Operation, DialectRegistry, Type

__all__ = [
    "ObjectFifoSubviewType",
    "ObjectFifoType",
    "aie_llvm_link",
    "generate_bcf",
    "generate_cdo",
    "generate_xaie",
    "npu_instgen",
    "register_dialect",
    "translate_aie_vec_to_cpp",
    "translate_mlir_to_llvmir",
]

def aie_llvm_link(modules: list[str]) -> str: ...
def generate_bcf(module: Operation, col: int, row: int) -> str: ...
def generate_cdo(
    module: Operation,
    work_dir_path: str,
    bigendian: bool = False,
    emit_unified: bool = False,
    cdo_debug: bool = False,
    aiesim: bool = False,
    xaie_debug: bool = False,
    partition_start_col: int = 1,
    enable_cores: bool = True,
) -> None: ...
def generate_xaie(module: Operation) -> str: ...
def npu_instgen(module: Operation) -> list: ...
def register_dialect(registry: DialectRegistry) -> None: ...
def translate_aie_vec_to_cpp(module: Operation, aieml: bool = False) -> str: ...
def translate_mlir_to_llvmir(module: Operation) -> str: ...

class ObjectFifoType:
    @staticmethod
    def get(type: Type) -> ObjectFifoType:
        """
        Create an ObjectFifoType type
        """

    @staticmethod
    def isinstance(other: Type) -> bool: ...

class ObjectFifoSubviewType:
    @staticmethod
    def get(type: Type) -> ObjectFifoSubviewType:
        """
        Create an ObjectFifoSubviewType type
        """

    @staticmethod
    def isinstance(other: Type) -> bool: ...
