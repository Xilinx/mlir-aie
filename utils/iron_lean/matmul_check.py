# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Concrete end-to-end check: the oracle's data movement plus the matmul kernel contract compute A @ B.

The kernel contract is the one ``mm_stream_dims`` documents: the core's A, B and C
buffers hold ``(r, s)``, ``(s, t)`` and ``(r, t)`` blocks in row-major block order,
and the core runs ``C_tile = sum_kk A_tile(kk) @ B_tile(kk)`` for each output tile,
as whole_array's ``core_fn`` does.
"""

import numpy as np
from aie.iron import Program
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
        m: int,
        k: int,
        n: int,
        mac_dims: tuple[int, int, int],
        seed: int = 0,
    ):
        self.sim = Simulation(program)
        self.M, self.K, self.N, self.m, self.k, self.n = M, K, N, m, k, n
        self.r, self.s, self.t = mac_dims
        rng = np.random.default_rng(seed)
        self.A = rng.integers(-8, 8, (M, K), dtype=np.int64)
        self.B = rng.integers(-8, 8, (K, N), dtype=np.int64)
        self.errors: list[str] = []

    def host_values(self, ids: np.ndarray) -> np.ndarray:
        """The A or B values the oracle's element ids name."""
        buf, idx = (ids >> 48) & 0xFF, ids & ((1 << 48) - 1)
        is_a = ((ids & HOST) != 0) & (buf == 0) & (idx < self.A.size)
        is_b = ((ids & HOST) != 0) & (buf == 1) & (idx < self.B.size)
        if not (is_a | is_b).all():
            bad = ids[~(is_a | is_b)][0]
            raise ValueError(f"core operand element from {self.sim.decode(bad)}")
        return np.where(
            is_a,
            self.A.flat[np.where(is_a, idx, 0)],
            self.B.flat[np.where(is_b, idx, 0)],
        )

    def core_tiles(self, w: int) -> list[np.ndarray]:
        """Worker `w`'s output tiles, in its C buffer's blocked layout."""
        m, k, n, r, s, t = self.m, self.k, self.n, self.r, self.s, self.t
        worker = self.sim.workers[w]
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
                a = operands["A"][j].reshape(m // r, k // s, r, s).transpose(0, 2, 1, 3)
                b = operands["B"][j].reshape(k // s, n // t, s, t).transpose(0, 2, 1, 3)
                acc += a.reshape(m, k) @ b.reshape(k, n)
            out.append(
                acc.reshape(m // r, r, n // t, t).transpose(0, 2, 1, 3).reshape(-1)
            )
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
        wrong = int((C != (self.A @ self.B).reshape(-1)).sum())
        if wrong:
            self.errors.append(f"{wrong} of {C.size} C elements differ from A @ B")
        return self.errors
