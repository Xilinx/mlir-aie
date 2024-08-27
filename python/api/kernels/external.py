"""
TODO: 
* docs
* types for inout_types
"""

from ... import ir
from ..resolvable import Resolvable


class ExternalKernel(Resolvable):
    def __init__(self, name: str, bin_name: str, inout_types: list) -> None:
        pass
