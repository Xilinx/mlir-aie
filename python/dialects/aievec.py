from ._aievec_ops_gen import *

from .._mlir_libs._aie import *
from .._mlir_libs import get_dialect_registry

# Comes from _aie
register_dialect(get_dialect_registry())
