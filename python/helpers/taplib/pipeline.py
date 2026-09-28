# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compose and check data-movement layouts across L3, memtile and core hops.

A design moves a tensor through up to four DMA walks before a core sees it:

1. **shim**: the runtime tap reads the host tensor onto the stream
   (``fill``/``drain``; there is no ``dims_to_stream`` on a shim).
2. **memtile_in**: the memtile consumer writes the stream into its object
   (``dims_from_stream`` on the fifo the shim feeds).
3. **memtile_out**: a sub-fifo's memtile producer reads its segment of that
   object back onto the stream (``dims_to_stream`` on ``split``/``forward``).
4. **core_in**: the core consumer writes the stream into the core's object
   (``dims_from_stream`` on the sub-fifo).

Each walk is a :class:`Hop`: an object shape, a ``(size, stride)`` list
(``None`` for a linear walk), a segment offset and the element width. The
:class:`Pipeline` composes them element by element with NumPy, so
:meth:`Pipeline.compose` tells you, for every element of every core object in
arrival order, which host element it is; and :meth:`Pipeline.check` reports
every hop that a tile's DMA could not execute (dimension count, wrap and
stride widths, the 32-bit address-generation granule) and every hop whose walk
does not cover its object exactly. Nothing here talks to the compiler: the
hop descriptors are what you passed to ``ObjectFifo``, spelled once, and the
result is what the verifier would only discover on hardware. A memtile_out
hop may pad (``pad_dimensions``, or a :class:`PaddedLayout`); padded
positions arrive as host index ``-1``.

Limits are the AIE2 family's (NPU1 and NPU2): see :data:`LIMITS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .layout import Layout, PaddedLayout
from .tap import TensorAccessPattern

__all__ = ["Hop", "Pipeline", "LIMITS", "KINDS"]

KINDS = ("shim", "memtile_in", "memtile_out", "core_in")

#: Per tile type: addressing dimensions a BD supports, the largest wrap
#: (size) an addressing dimension can hold, the largest step (stride), and
#: for the shim the largest outermost tap dimension: it becomes the queue
#: repeat count (8 bits, so 256 executions) and, when it steps, also the BD
#: iteration wrap (6 bits, so 64). Address generation is in 32-bit words on
#: every tile.
LIMITS = {
    "shim": dict(
        max_dims=3, max_wrap=1023, max_stride=1 << 20, max_repeat=256, max_iter=64
    ),
    "memtile": dict(
        max_dims=4, max_wrap=1023, max_stride=1 << 17, max_repeat=None, max_iter=None
    ),
    "core": dict(
        max_dims=3, max_wrap=255, max_stride=1 << 13, max_repeat=None, max_iter=None
    ),
}
_TILE_OF_KIND = {
    "shim": "shim",
    "memtile_in": "memtile",
    "memtile_out": "memtile",
    "core_in": "core",
}
GRANULE_BYTES = 4


@dataclass(frozen=True)
class Hop:
    """One DMA walk over one object.

    Args:
        kind (str): One of :data:`KINDS`.
        shape (Sequence[int]): The object the walk indexes: the host tensor for a
            shim hop, the memtile pool object for memtile hops, the core's
            object for a core hop.
        dims (Layout | PaddedLayout | Sequence[tuple[int, int]] | None): the
            walk, as ``dims_to_stream``/``dims_from_stream`` take it: a
            ``Layout`` (a ``PaddedLayout`` also sets ``pad``) or ``(size,
            stride)`` pairs, outermost first, in elements; ``None`` walks the
            object (or the segment) linearly. A shim hop may instead be given a
            :class:`Layout` or :class:`TensorAccessPattern` via
            :meth:`Hop.shim`.
        offset (int): Element offset of the walk's origin within the object
            (a split's segment start, or a tap's offset).
        length (int | None): For a linear walk, how many elements; defaults to
            the whole object after ``offset`` (a segment's length for a split).
        elem_bytes (int): Element width, for the granule rule.
        name (str): Label used in messages.
        pad (Sequence[tuple[int, int]] | None): ``(before, after)`` constant
            elements the walk emits around each pass over each dimension
            (``pad_dimensions``); memtile_out only, one pair per entry of
            ``dims``. Padded positions carry host index ``-1`` in
            :meth:`Pipeline.compose`.
    """

    kind: str
    shape: tuple
    dims: tuple | None = None
    offset: int = 0
    length: int | None = None
    elem_bytes: int = 4
    name: str = ""
    pad: tuple | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        if isinstance(self.dims, PaddedLayout):
            object.__setattr__(self, "pad", self.dims.pad_dims())
            object.__setattr__(self, "dims", self.dims.stream_dims())
        elif isinstance(self.dims, Layout):
            object.__setattr__(self, "dims", self.dims.stream_dims())
        if self.dims is not None:
            object.__setattr__(
                self, "dims", tuple((int(s), int(t)) for s, t in self.dims)
            )
        if self.pad is not None:
            object.__setattr__(
                self, "pad", tuple((int(b), int(a)) for b, a in self.pad)
            )

    @classmethod
    def shim(cls, tap, elem_bytes: int = 4, name: str = "shim") -> Hop:
        """Build a shim hop from a runtime tap (:class:`Layout` or :class:`TensorAccessPattern`)."""
        if isinstance(tap, Layout):
            tap = tap.tap()
        if not isinstance(tap, TensorAccessPattern):
            raise TypeError("shim() takes a Layout or a TensorAccessPattern")
        return cls(
            "shim",
            tuple(tap.tensor_dims),
            tuple(zip(tap.sizes, tap.strides)),
            int(tap.offset),
            None,
            elem_bytes,
            name,
        )

    # ------------------------------------------------------------ geometry

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def label(self) -> str:
        return self.name or self.kind

    def layout(self) -> Layout:
        """Return the walk as a :class:`Layout` over the object."""
        if self.dims is None:
            n = self.numel - self.offset if self.length is None else self.length
            return Layout(self.shape, self.offset, [n], [1])
        sizes = [s for s, _ in self.dims]
        strides = [t for _, t in self.dims]
        return Layout(self.shape, self.offset, sizes, strides)

    def order(self) -> np.ndarray:
        """Flat object indices in the order the walk touches them.

        A padded walk (``pad``) emits ``-1`` at every padded position.
        """
        lay = self.layout()
        idx = np.zeros((), dtype=np.int64) + lay.offset
        for size, stride in zip(lay.sizes, lay.strides):
            idx = idx[..., None] + np.arange(size, dtype=np.int64) * stride
        if self.pad and len(self.pad) == idx.ndim:
            idx = np.pad(idx, self.pad, constant_values=-1)
        return idx.reshape(-1)

    @property
    def emitted(self) -> int:
        """Elements the walk puts on the stream, padding included."""
        return len(self.order())

    # --------------------------------------------------------------- checks

    def issues(self) -> list[str]:
        """Why a DMA on this hop's tile could not execute this walk, if anything."""
        out: list[str] = []
        lim = LIMITS[_TILE_OF_KIND[self.kind]]
        lay = self.layout()
        sizes, strides = list(lay.sizes), list(lay.strides)
        # The shim's outermost tap slot is the queue repeat.
        if self.kind == "shim" and len(sizes) == 4:
            rep, rep_stride = sizes[0], strides[0]
            sizes, strides = sizes[1:], strides[1:]
            if lim["max_repeat"] and rep > lim["max_repeat"]:
                out.append(f"{self.label}: repeat {rep} exceeds {lim['max_repeat']}")
            elif rep_stride and lim["max_iter"] and rep > lim["max_iter"]:
                out.append(
                    f"{self.label}: stepped repeat {rep} exceeds the iteration wrap {lim['max_iter']}"
                )
        real = [(s, t) for s, t in zip(sizes, strides) if s != 1]
        if len(real) > lim["max_dims"]:
            out.append(
                f"{self.label}: {len(real)} addressing dimensions, tile supports {lim['max_dims']}"
            )
        for i, (s, t) in enumerate(real):
            if s > lim["max_wrap"] and not (
                i == len(real) - 1 and t == 1 and len(real) == 1
            ):
                out.append(f"{self.label}: size {s} exceeds wrap {lim['max_wrap']}")
            if t > lim["max_stride"]:
                out.append(f"{self.label}: stride {t} exceeds {lim['max_stride']}")
        # Address generation works in 32-bit words.
        if real:
            inner_size, inner_stride = real[-1]
            if inner_stride == 1:
                if (inner_size * self.elem_bytes) % GRANULE_BYTES:
                    out.append(
                        f"{self.label}: innermost extent {inner_size} x {self.elem_bytes} B is not a whole {GRANULE_BYTES}-byte granule"
                    )
            elif (inner_stride * self.elem_bytes) % GRANULE_BYTES:
                out.append(
                    f"{self.label}: innermost stride {inner_stride} x {self.elem_bytes} B is not a whole {GRANULE_BYTES}-byte granule (a sub-word transpose)"
                )
            for s, t in real[:-1]:
                if (t * self.elem_bytes) % GRANULE_BYTES:
                    out.append(
                        f"{self.label}: stride {t} x {self.elem_bytes} B is not a whole granule"
                    )
        if (self.offset * self.elem_bytes) % GRANULE_BYTES:
            out.append(
                f"{self.label}: offset {self.offset} x {self.elem_bytes} B is not a whole granule"
            )
        if self.pad:
            if self.kind != "memtile_out":
                out.append(
                    f"{self.label}: padding is only available on a memtile_out hop"
                )
            if self.dims is None or len(self.pad) != len(self.dims):
                out.append(
                    f"{self.label}: padding has {len(self.pad)} entries for {0 if self.dims is None else len(self.dims)} dims"
                )
            else:
                before, after = self.pad[-1]
                for what, count in (("before", before), ("after", after)):
                    if (count * self.elem_bytes) % GRANULE_BYTES:
                        out.append(
                            f"{self.label}: innermost padding {what} {count} x {self.elem_bytes} B is not a whole {GRANULE_BYTES}-byte granule"
                        )
        last = lay.offset + sum((s - 1) * t for s, t in zip(lay.sizes, lay.strides))
        if last >= self.numel:
            out.append(
                f"{self.label}: walk reaches element {last} of an object of {self.numel}"
            )
        return out


@dataclass
class Pipeline:
    """A chain of hops from the host tensor to a core's object.

    Build it hop by hop with :meth:`shim`, :meth:`memtile_in`,
    :meth:`memtile_out` and :meth:`core_in`, in stream order. Several shim
    taps (issued one after another) and several memtile-out segments (a
    ``split``) are allowed; a segment carries its own core hop.
    """

    hops: list = field(default_factory=list)

    def add(self, hop: Hop) -> Pipeline:
        self.hops.append(hop)
        return self

    def shim(self, tap, elem_bytes: int = 4, name: str = "shim") -> Pipeline:
        return self.add(Hop.shim(tap, elem_bytes, name))

    def memtile_in(
        self, shape, dims=None, elem_bytes: int = 4, name: str = "memtile_in"
    ) -> Pipeline:
        return self.add(Hop("memtile_in", shape, dims, 0, None, elem_bytes, name))

    def memtile_out(
        self,
        shape,
        dims=None,
        offset: int = 0,
        length: int | None = None,
        elem_bytes: int = 4,
        name: str = "memtile_out",
        pad=None,
    ) -> Pipeline:
        """Add a memtile ``dims_to_stream`` walk; ``dims`` may be a :class:`PaddedLayout`."""
        return self.add(
            Hop("memtile_out", shape, dims, offset, length, elem_bytes, name, pad)
        )

    def core_in(
        self, shape, dims=None, elem_bytes: int = 4, name: str = "core_in"
    ) -> Pipeline:
        return self.add(Hop("core_in", shape, dims, 0, None, elem_bytes, name))

    # --------------------------------------------------------------- checks

    def check(self) -> list[str]:
        """Every per-hop legality issue plus every coverage mismatch between hops."""
        out: list[str] = []
        for h in self.hops:
            out += h.issues()
        # Coverage: a fill walk must be a bijection onto its object, and a
        # drain walk must be a bijection onto its segment.
        for h in self.hops:
            if h.kind in ("memtile_in", "core_in"):
                o = h.order()
                if len(o) != h.numel or len(np.unique(o)) != h.numel:
                    out.append(
                        f"{h.label}: dims_from_stream walk visits {len(o)} elements "
                        f"({len(np.unique(o))} distinct) of an object of {h.numel}"
                    )
            elif h.kind == "memtile_out":
                o = h.order()
                o = o[o >= 0]  # padded positions read nothing
                seg = h.numel - h.offset if h.length is None else h.length
                if len(np.unique(o)) != len(o):
                    out.append(
                        f"{h.label}: dims_to_stream walk revisits elements of its segment"
                    )
                if len(o) != seg:
                    out.append(
                        f"{h.label}: dims_to_stream walk emits {len(o)} elements but the segment holds {seg}"
                    )
        # Stream lengths must chunk evenly from hop to hop.
        stream = sum(len(h.order()) for h in self.hops if h.kind == "shim")
        mem_in = [h for h in self.hops if h.kind == "memtile_in"]
        if mem_in and stream % mem_in[0].numel:
            out.append(
                f"shim taps put {stream} elements on the stream, not a multiple of the memtile object ({mem_in[0].numel})"
            )
        return out

    # -------------------------------------------------------------- compose

    def compose(self) -> list[np.ndarray]:
        """Host flat index of every element of every core object, in arrival order.

        Returns one array per core object received, shaped like the core
        object. With no core hop, the memtile objects are returned; with no
        memtile hops, the shim stream chunked by nothing (one array).

        Raises:
            ValueError: If a walk does not cover its object exactly or the
                stream does not chunk evenly (see :meth:`check`).
        """
        shim = [h for h in self.hops if h.kind == "shim"]
        if not shim:
            raise ValueError("a Pipeline needs at least one shim hop")
        stream = np.concatenate([h.order() for h in shim])
        mem_in = [h for h in self.hops if h.kind == "memtile_in"]
        mem_out = [h for h in self.hops if h.kind == "memtile_out"]
        cores = [h for h in self.hops if h.kind == "core_in"]
        if not mem_in and not mem_out:
            if not cores:
                return [stream]
            return self._chunk_into(stream, cores[0], "core_in")
        if len(mem_in) > 1:
            raise ValueError("one memtile_in hop per pipeline")
        if mem_in:
            objs = self._chunk_into(stream, mem_in[0], "memtile_in")
        else:
            # A linear fill of the memtile object.
            shape = mem_out[0].shape
            objs = self._chunk_into(stream, Hop("memtile_in", shape), "memtile_in")
        if not mem_out:
            return objs
        if cores and len(cores) not in (1, len(mem_out)):
            raise ValueError("give one core_in hop, or one per memtile_out segment")
        results: list[np.ndarray] = []
        for obj in objs:
            flat = obj.reshape(-1)
            for i, drain in enumerate(mem_out):
                order = drain.order()
                out_stream = np.where(order >= 0, flat[np.maximum(order, 0)], -1)
                if not cores:
                    results.append(out_stream)
                    continue
                core = cores[i] if len(cores) > 1 else cores[0]
                results += self._chunk_into(out_stream, core, "core_in")
        return results

    @staticmethod
    def _chunk_into(stream: np.ndarray, hop: Hop, what: str) -> list[np.ndarray]:
        n = hop.numel
        if len(stream) % n:
            raise ValueError(
                f"{hop.label}: stream of {len(stream)} elements does not chunk into objects of {n}"
            )
        order = hop.order()
        if len(order) != n or len(np.unique(order)) != n:
            raise ValueError(
                f"{hop.label}: {what} walk visits {len(order)} elements ({len(np.unique(order))} distinct) of an object of {n}"
            )
        out = []
        for start in range(0, len(stream), n):
            obj = np.empty(n, dtype=np.int64)
            obj[order] = stream[start : start + n]
            out.append(obj.reshape(hop.shape))
        return out

    def host_coords(
        self, host_shape: Sequence[int], objects: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Turn composed host flat indices into host coordinates (last axis)."""
        return [np.stack(np.unravel_index(o, host_shape), axis=-1) for o in objects]
