# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Concrete end-to-end check: the oracle's data movement plus the matmul kernel contract compute A @ B.

Each core's operand layouts come from its matmul kernel's own ``KernelContract``: the
``TensorLayout`` codecs say how the A, B and C buffers hold the logical tiles. The core runs
``C_tile = sum_kk A_tile(kk) @ B_tile(kk)`` for each output tile, as whole_array's ``core_fn``
does. With ``b_col_maj`` (``c_col_maj``) the host holds B (C) transposed, as whole_array's taps
assume.
"""

import numpy as np
from aie.iron import Program
from aie.iron.kernels.linalg import MatrixKernel
from aie.iron.runtime.endpoint import RuntimeEndpoint

from sim import HOST, Simulation


class MatmulCheck:
    """Runs whole_array's data movement on random integers and compares C with numpy."""

    def __init__(
        self,
        program: Program,
        M: int,
        K: int,
        N: int,
        b_col_maj: bool = False,
        c_col_maj: bool = False,
        seed: int = 0,
    ):
        self.sim = Simulation(program)
        self.M, self.K, self.N = M, K, N
        self.c_col_maj = c_col_maj
        rng = np.random.default_rng(seed)
        self.A = rng.integers(-8, 8, (M, K), dtype=np.int64)
        self.B = rng.integers(-8, 8, (K, N), dtype=np.int64)
        self.host_B = (self.B.T if b_col_maj else self.B).reshape(-1)
        self.errors: list[str] = []

    def host_values(self, ids: np.ndarray) -> np.ndarray:
        """The A or B values the oracle's element ids name."""
        buf, idx = (ids >> 48) & 0xFF, ids & ((1 << 48) - 1)
        is_a = ((ids & HOST) != 0) & (buf == 0) & (idx < self.A.size)
        is_b = ((ids & HOST) != 0) & (buf == 1) & (idx < self.host_B.size)
        if not (is_a | is_b).all():
            bad = ids[~(is_a | is_b)][0]
            raise ValueError(f"core operand element from {self.sim.decode(bad)}")
        return np.where(
            is_a,
            self.A.flat[np.where(is_a, idx, 0)],
            self.host_B[np.where(is_b, idx, 0)],
        )

    def core_tiles(self, w: int) -> list[np.ndarray]:
        """Worker `w`'s output tiles, in its C buffer's layout."""
        worker = self.sim.workers[w]
        kernel = next(a for a in worker.flat_fn_args if isinstance(a, MatrixKernel))
        layout_a, layout_b, layout_c = kernel.contract.layouts
        (m, k), n = layout_a.shape, layout_b.shape[1]
        operands = {}
        for h in worker.fifos:
            if not h.is_prod:
                objs = self.sim.consumer_objects(h.object_fifo, worker, None)
                operands[h.object_fifo.name[0]] = self.host_values(objs)
        k_iters = self.K // k
        tiles = (self.M // m) * (self.N // n) // len(self.sim.workers)
        for name, objs in operands.items():
            if len(objs) != tiles * k_iters:
                self.errors.append(
                    f"worker {w} receives {len(objs)} {name} objects; it consumes {tiles * k_iters}"
                )
        out = []
        for tau in range(tiles):
            acc = np.zeros((m, n), np.int64)
            for kk in range(k_iters):
                j = tau * k_iters + kk
                if j >= min(len(operands["A"]), len(operands["B"])):
                    break
                a = layout_a.decode(operands["A"][j])[0]
                b = layout_b.decode(operands["B"][j])[0]
                acc += a @ b
            out.append(layout_c.encode(acc)[0])
        return out

    def run(self) -> list[str]:
        """Every way the computed C differs from `A @ B`; empty when it matches."""
        tiles = [self.core_tiles(w) for w in range(len(self.sim.workers))]
        C = np.zeros(self.M * self.N, np.int64)
        writes = np.zeros(self.M * self.N, np.int64)
        drains = [
            of
            for of in self.sim.fifos
            if any(isinstance(c.endpoint, RuntimeEndpoint) for c in of.consumers)
        ]
        for of in drains:
            for buf, idx, x in self.sim.drain_writes(of):
                match self.sim.decode(x):
                    case ("core", w, _, j, slot) if buf == 2 and 0 <= idx < C.size:
                        if j >= len(tiles[w]):
                            self.errors.append(
                                f"C[{idx}] reads worker {w}'s object {j}, which it never produces"
                            )
                            continue
                        C[idx] = tiles[w][j][slot]
                        writes[idx] += 1
                    case other:
                        self.errors.append(
                            f"drain writes buffer {buf}[{idx}] from {other}"
                        )
        if (writes != 1).any():
            twice, never = int((writes > 1).sum()), int((writes == 0).sum())
            self.errors.append(f"C elements written twice: {twice}, never: {never}")
        want = self.A @ self.B
        wrong = int((C != (want.T if self.c_col_maj else want).reshape(-1)).sum())
        if wrong:
            self.errors.append(f"{wrong} of {C.size} C elements differ from A @ B")
        return self.errors
