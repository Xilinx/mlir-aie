#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Llama 3.2 1B on the NPU as one full ELF: the decode (aie2_decode.py) and
the prefill (aie2_prefill_layer.py), each its own DeviceConfiguration, so a
host runs both on one hardware context. Each device's first command loads
its PDI, so the host may run a request's prefill passes and then its decode
in one queue, the switch a PDI load rather than a context switch.

Run (compiles the design, or reuses its JIT cache entry):
  python aie2_llama.py --n-layers 16 --t 131072 --full-elf-path llama.elf
"""

import argparse

from aie.iron import CompileTime, Program, jit
from aie.iron.device import NPU2
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie2_decode import decode_configuration
from aie2_prefill_layer import prefill_configuration
from llama_config import N_LAYERS as MODEL_LAYERS


# The sequential placer leaves the decode's 31 cores' flows unroutable along
# the shim row; the SA placer's placement routes.
@jit(aiecc_flags=["--placer=sa_placer", "--sa-seed=1"])
def build_llama(
    *,
    P_MAX: CompileTime[int] = 4,
    N_LAYERS: CompileTime[int] = 2,
    T: CompileTime[int] = 128,
    KSET: CompileTime[int] = 8,
):
    """The decode and the prefill of one build, the decode's command the ELF's
    entry.

    Args:
        `P_MAX`, `N_LAYERS`, `T`, `KSET`: aie2_decode.decode_configuration's;
            the prefill runs all `N_LAYERS` over the same `T`.

    Returns:
        The resolved program.
    """
    decode = decode_configuration(P_MAX=P_MAX, N_LAYERS=N_LAYERS, T=T, KSET=KSET)
    prefill = prefill_configuration(N_LAYERS=N_LAYERS, LAYER=0, W_LAYERS=N_LAYERS, T=T)
    return Program.compose(
        [decode, prefill], entry=decode.runtimes[0]
    ).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B decode and prefill")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    # Four passes use 31 of the decode's 32 ScratchpadParameters; a fifth
    # needs four more.
    p.add_argument("--p-max", type=int, choices=range(1, 5), default=4)
    p.add_argument(
        "--n-layers", type=int, choices=range(1, MODEL_LAYERS + 1), default=2
    )
    p.add_argument("--t", type=int, default=128)
    p.add_argument("--kset", type=int, choices=range(1, 9), default=8)
    opts = p.parse_args()
    run_design_cli(
        build_llama,
        opts,
        compile_kwargs=lambda o: dict(
            P_MAX=o.p_max, N_LAYERS=o.n_layers, T=o.t, KSET=o.kset
        ),
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
