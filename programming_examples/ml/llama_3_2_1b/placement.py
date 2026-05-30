"""Llama 3.2 1B — physical tile placement on AIE2P NPU2 (decode + prefill).

Mirrors the algorithm/placement split from
miniature-fishstick/npu2/placement.py: this file is the ONLY place tile
coordinates appear. Re-targeting the design = editing the dicts here.

    llama_spec.py             - algorithm (what each op computes)
    llama_capacity_analysis.py - feasibility math
    llama_placement.py        - physical tile assignment (this file)
    aie2_llama_iron.py        - placement-level IRON skeleton + real-build stub
    aie2_llama_layer.py       - runnable one-layer dataflow design (vs iron_mock)

Reads `llama_spec` and validates (on import) that every op kind in a
transformer layer has a placement home, that tile coords are in range,
and that no two functions collide on a tile.

================================================================================
AIE2P tile array: 8 columns (0-7) x 6 rows (0-5)
  row 0      : shim   (DRAM interface; 8 DMA columns)
  row 1      : memtile (L2; weight + KV staging, double-buffer prefetch)
  rows 2-5   : compute (4 rows x 8 cols = 32 compute tiles)
================================================================================

DECODE OVERLAY LAYOUT (the primary v1 overlay — see docs/capacity_findings.md
"Tile mapping" + "Overlay strategy"):

        c0      c1      c2      c3      c4      c5      c6      c7
  r5   attn1.0 attn1.1 attn1.2 attn1.3 silu    sample  ----    ----     compute
  r4   attn0.0 attn0.1 attn0.2 attn0.3 rope    rmsnorm ----    ----     compute
  r3   proj    proj    proj    proj    proj    proj    proj    proj     compute
  r2   proj    proj    proj    proj    proj    proj    proj    proj     compute
  r1   MEMTILE  MEMTILE  MEMTILE  MEMTILE  MEMTILE  MEMTILE  MEMTILE  MEMTILE   L2
  r0   SHIM     SHIM     SHIM     SHIM     SHIM     SHIM     SHIM     SHIM      DRAM

  proj   (16 CTs, rows 2-3 x all 8 cols): projection block. The single
         GEMM kernel reused for q/k/v/o/gate/up/down/lm_head via BD
         patching of the weight base address. Spans all 8 columns so
         all 8 shim DMA channels fan out weights in parallel (the
         8-col bandwidth requirement, Tier-1 decision #5).
  attn0.N / attn1.N (8 CTs, rows 4-5 x cols 0-3): FlowKV attention,
         4 vertical pairs. attn0 (row 4) = CT0 (Q@Kᵀ over a chunk +
         row-max/exp/denominator); attn1 (row 5) = CT1 (softmax-
         weighted @V accumulation). Each pair handles 2 KV heads.
         CT0->CT1 intermediate state (F, C, l) flows through the
         shared memory of vertically-adjacent tiles — which is why the
         pairs are vertical, not horizontal.
  rope     (c4,r4): RoPE rotation-pair permute + cos/sin multiply.
  rmsnorm  (c5,r4): RMSNorm reduction + scale, fused with residual add.
  silu     (c4,r5): SiLU (linear_approx) + gate*up elementwise multiply.
  sample   (c5,r5): temperature + top-k + multinomial + EOS detect.
  embedding: NO compute tile — pure shim DMA (one row of the 263 MiB
         table per token, broadcast into rmsnorm). See PLACEMENT["embedding"].
  spare  (cols 6-7, rows 4-5): 4 tiles. Headroom for v2 (speculative-
         decode draft heads, lm_head vocab-split reduction, etc.).

  Total: 28 of 32 compute tiles used, 4 spare.

--------------------------------------------------------------------------------
DEVIATION from docs/capacity_findings.md first-draft tile mapping:
  The doc's first draft described projection as "8 cols x 2 rows" AND
  attention as "2 cols x 4 rows". Those over-subscribe the column axis —
  the 8-col projection needs ALL 8 columns, leaving none for a 2-col
  attention block in the same rows. Resolved here by stacking: projection
  occupies rows 2-3 (all 8 cols, for the 8-col DMA fan-out), attention
  occupies rows 4-5 cols 0-3 as 4 VERTICAL pairs. This is strictly better
  for FlowKV: the CT0->CT1 handoff uses vertical row-adjacency shared
  memory rather than a cross-column stream. Projection and attention run
  at different times within a layer, so projection's 8-col DMA fan-out
  and attention's compute don't contend for the shim channels.
  (capacity_findings.md "Tile mapping" updated to match.)
--------------------------------------------------------------------------------

PREFILL OVERLAY LAYOUT (compute-bound regime — projection-heavy):

        c0      c1      c2      c3      c4      c5      c6      c7
  r5   attn0.0 attn0.1 attn1.0 attn1.1 rope    rmsnorm silu    sample   compute
  r4   proj    proj    proj    proj    proj    proj    proj    proj     compute
  r3   proj    proj    proj    proj    proj    proj    proj    proj     compute
  r2   proj    proj    proj    proj    proj    proj    proj    proj     compute
  r1/r0: memtile / shim (same as decode)

  Prefill is compute-bound and projection is ~97% of its MACs (32 GMACs/
  layer vs ~1 for attention at M=512). So projection gets 24 CTs (rows 2-4,
  all cols) — ATB-tiled GEMM, weights amortized over M tokens. Attention
  shrinks to 4 CTs (2 horizontal pairs in row 5) since it's ~3% of the work;
  glue (rope/rmsnorm/silu/sample) fills the rest of row 5. 32 CTs used, 0
  spare — a compute-bound regime should use the whole array.

  Why separate from decode: the 50-TOPS / 26-ms compute ceiling assumes the
  FULL array. Running projection on decode's 16 tiles would ~halve prefill
  throughput. 24 tiles -> 75% of the array; full 32-tile overlap needs
  time-multiplexed kernels (projection-phase + attention-phase share tiles)
  — a v2 lever. The 3 bin-and-pad variants (M in {128,512,2048}) share this
  layout; only compile-time M + ATB tile sizes differ.
--------------------------------------------------------------------------------
"""

from dataclasses import dataclass

import llama_spec as spec


# ---------------------------------------------------------------------------
# Tile — use the real IRON class if mlir-aie is installed, else a local
# dataclass so the placement is loadable for static checking outside IRON.
# ---------------------------------------------------------------------------
try:
    from aie.iron.device import Tile  # type: ignore
except ImportError:
    @dataclass(frozen=True)
    class Tile:  # type: ignore[no-redef]
        col: int
        row: int

        def __repr__(self) -> str:
            return f"Tile({self.col},{self.row})"


# Compute-row span and column count for AIE2P.
N_COLS = 8
COMPUTE_ROWS = (2, 3, 4, 5)
SHIM_ROW = 0
MEMTILE_ROW = 1


# ---------------------------------------------------------------------------
# Shim + memtile rows are shared by both overlays.
# ---------------------------------------------------------------------------
_SHIM = {
    "weights": [Tile(c, SHIM_ROW) for c in range(N_COLS)],  # 8-col fan-out
    "io":      Tile(0, SHIM_ROW),       # token-id(s) in, (token_id, eos) out
}
# NOTE: tiles span all 8 columns, but the per-column weight-DMA *byte
# allocation* is a separate (DMA-scheduling) decision. If the Phase-0
# per-column bandwidth sweep shows middle cols are faster than edges,
# size each column's BD proportional to its measured bandwidth rather
# than splitting equally — equal split is gated by the slowest column
# (decode is DRAM-bound). See docs/capacity_findings.md "First-hardware
# experiments". This does NOT change tile placement, only BD sizing.
_MEMTILE = [Tile(c, MEMTILE_ROW) for c in range(N_COLS)]


# ---------------------------------------------------------------------------
# DECODE_PLACEMENT — DRAM-bandwidth-bound regime. Projection spread across
# all 8 columns (8-col DMA fan-out is the binding lever); attention as 4
# vertical CT0->CT1 pairs. 28 CTs used, 4 spare. Keyed by functional unit;
# each llama_spec layer-op maps to a unit via OP_UNIT below.
# ---------------------------------------------------------------------------
DECODE_PLACEMENT: dict = {
    # Projection: 16 CTs, rows 2-3 x all 8 cols. One GEMM/GEMV kernel
    # serves q/k/v/o/gate/up/down/lm_head, weight base BD-patched per call.
    "projection": [Tile(c, r) for r in (2, 3) for c in range(N_COLS)],

    # FlowKV attention: 4 vertical pairs in cols 0-3, rows 4 (CT0) / 5 (CT1).
    # 2 KV heads per pair (4 pairs x 2 = 8 KV heads).
    "attention": {
        f"pair{p}": {"qk": Tile(p, 4), "sv": Tile(p, 5)}  # CT0 row 4, CT1 row 5
        for p in range(4)
    },

    # Glue (1 CT each).
    "rope":    Tile(4, 4),
    "rmsnorm": Tile(5, 4),   # RMSNorm + residual add (fused)
    "silu":    Tile(4, 5),   # SiLU + gate*up multiply (fused)
    "sample":  Tile(5, 5),   # temperature + top-k + multinomial + EOS

    "embedding": None,       # pure DMA — no compute tile
    "spare": [Tile(c, r) for r in (4, 5) for c in (6, 7)],  # 4 CTs, v2 headroom
    "shim": _SHIM,
    "memtile": _MEMTILE,
}


# ---------------------------------------------------------------------------
# PREFILL_PLACEMENT — compute-bound regime. Projection is ~97% of prefill
# MACs (32 GMACs/layer vs ~1 for attention at M=512), so maximize projection
# tiles: 24 CTs (rows 2-4 x all cols). Attention shrinks to 4 CTs (2
# horizontal CT0->CT1 pairs, 4 KV heads each) since it's a small fraction of
# the work. Glue fills row 5. 32 CTs used, 0 spare — prefill uses the whole
# array for compute (the point of a compute-bound regime).
#
# Why a separate partition from decode: the 50-TOPS / 26-ms compute ceiling
# assumes the FULL array. Running prefill's dominant op (projection) on only
# decode's 16 tiles would halve effective throughput. 24 tiles gets projection
# to 75% of the array; full overlap (32) needs time-multiplexed kernels — a v2
# lever (see docs Bring-up Phase 6).
#
# Bin-and-pad: 3 compiled prefill overlays for M in {128, 512, 2048} share
# this SAME tile layout — only the compile-time M (and ATB tile sizes) differ.
# ---------------------------------------------------------------------------
PREFILL_PLACEMENT: dict = {
    # Projection: 24 CTs, rows 2-4 x all 8 cols. ATB-tiled GEMM (small T_MA,
    # larger T_MC); weights amortized over M tokens so DMA is not the binding
    # cost here — compute is.
    "projection": [Tile(c, r) for r in (2, 3, 4) for c in range(N_COLS)],

    # FlowKV attention: 2 horizontal pairs in row 5 (c0->c1, c2->c3), 4 KV
    # heads each. Flash-attention chunked along M (the M=512 score matrix is
    # 32 MiB, exceeds L2). Horizontal neighbors share memory for the CT0->CT1
    # F/C/l handoff. Small tile count is fine: attention is ~3% of prefill MACs.
    "attention": {
        "pair0": {"qk": Tile(0, 5), "sv": Tile(1, 5)},
        "pair1": {"qk": Tile(2, 5), "sv": Tile(3, 5)},
    },

    # Glue (1 CT each, row 5).
    "rope":    Tile(4, 5),
    "rmsnorm": Tile(5, 5),
    "silu":    Tile(6, 5),
    "sample":  Tile(7, 5),   # fires once, on the final prefill position

    "embedding": None,       # pure DMA
    "spare": [],             # none — compute-bound regime uses the whole array
    "shim": _SHIM,
    "memtile": _MEMTILE,
}

# The decode overlay is the primary one; PLACEMENT aliases it for callers that
# don't care which overlay (e.g. generic validation).
PLACEMENT = DECODE_PLACEMENT


# ---------------------------------------------------------------------------
# OP -> UNIT: which functional unit runs each llama_spec layer-op kind.
# Used by _validate to confirm every op has a placement home.
# ---------------------------------------------------------------------------
OP_UNIT = {
    "Linear":      "projection",   # q/k/v/o/gate/up/down (+ lm_head, post-layer)
    "MatMul":      "attention",    # attn_qk, attn_sv  (FlowKV pairs)
    "Softmax":     "attention",    # folded into FlowKV CT0
    "RMSNorm":     "rmsnorm",
    "RoPE":        "rope",
    "SiLUMul":     "silu",
    "ResidualAdd": "rmsnorm",      # fused at the residual write-back
}


# ---------------------------------------------------------------------------
# FLOWS — on-chip data-movement edges the IRON design wires up: each FlowKV
# pair streams F,C,l from CT0 (Q@K + softmax numerator) to CT1 (@V accum)
# via shared memory of adjacent tiles (vertical in decode, horizontal in
# prefill — both are cardinal-neighbor adjacencies on AIE2P).
# ---------------------------------------------------------------------------
def flows(placement: dict) -> tuple:
    return tuple(
        (pair["qk"], pair["sv"]) for pair in placement["attention"].values()
    )


DECODE_FLOWS = flows(DECODE_PLACEMENT)
PREFILL_FLOWS = flows(PREFILL_PLACEMENT)


# ---------------------------------------------------------------------------
# Static checks — run on import to catch errors before IRON sees the file.
# ---------------------------------------------------------------------------
def _all_compute_tiles(placement: dict):
    """Flatten a placement to (unit, subname, Tile) for compute-row tiles."""
    out = []
    for unit, val in placement.items():
        if unit in ("shim", "memtile", "embedding"):
            continue
        if isinstance(val, Tile):
            out.append((unit, None, val))
        elif isinstance(val, list):
            for t in val:
                out.append((unit, None, t))
        elif isinstance(val, dict):  # attention pairs
            for sub, pair in val.items():
                for role, t in pair.items():
                    out.append((unit, f"{sub}/{role}", t))
    return out


def _validate(placement: dict, name: str, expect_used: int) -> None:
    compute = _all_compute_tiles(placement)

    # (a) all compute tiles in rows 2-5, cols 0-7
    for unit, sub, t in compute:
        assert 0 <= t.col < N_COLS, f"[{name}] {unit}/{sub}: col {t.col} out of range"
        assert t.row in COMPUTE_ROWS, f"[{name}] {unit}/{sub}: row {t.row} not a compute row"

    # (b) no two functions collide on a compute tile
    seen: dict = {}
    for unit, sub, t in compute:
        key = (t.col, t.row)
        if key in seen:
            raise AssertionError(f"[{name}] tile collision at {t}: {unit}/{sub} vs {seen[key]}")
        seen[key] = f"{unit}/{sub}"

    # (c) tile budget
    used = [x for x in compute if x[0] != "spare"]
    spare = [x for x in compute if x[0] == "spare"]
    assert len(used) == expect_used, f"[{name}] expected {expect_used} used CTs, got {len(used)}"
    assert len(compute) == 32, f"[{name}] expected 32 compute tiles total, got {len(compute)}"
    assert len(used) + len(spare) == 32, f"[{name}] used+spare != 32"

    # (d) every layer-op kind has a placement home present in this placement
    op_kinds = {type(op).__name__ for op in spec.LAYER_OPS}
    missing = op_kinds - set(OP_UNIT)
    assert not missing, f"layer-op kinds with no OP_UNIT mapping: {missing}"
    for kind, unit in OP_UNIT.items():
        assert unit in placement, f"[{name}] OP_UNIT maps {kind}->{unit}, not in placement"

    # (e) shim/memtile rows
    for t in placement["shim"]["weights"] + [placement["shim"]["io"]]:
        assert t.row == SHIM_ROW, f"[{name}] shim tile {t} not on row {SHIM_ROW}"
    for t in placement["memtile"]:
        assert t.row == MEMTILE_ROW, f"[{name}] memtile {t} not on row {MEMTILE_ROW}"


_validate(DECODE_PLACEMENT,  "decode",  expect_used=28)
_validate(PREFILL_PLACEMENT, "prefill", expect_used=32)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
def render_diagram(placement: dict, name: str):
    label = {}
    for unit, sub, t in _all_compute_tiles(placement):
        if unit == "projection":   label[(t.col, t.row)] = "proj"
        elif unit == "attention":  label[(t.col, t.row)] = sub.replace("pair", "att").replace("/qk", "0").replace("/sv", "1")
        elif unit == "spare":      label[(t.col, t.row)] = "----"
        else:                      label[(t.col, t.row)] = unit[:6]
    print(f"  === {name} overlay ===")
    print("        " + "".join(f"c{c:<6d}" for c in range(N_COLS)))
    for r in (5, 4, 3, 2):
        cells = "".join(f"{label.get((c, r), ''):<7s}" for c in range(N_COLS))
        print(f"  r{r}   {cells}")
    print(f"  r1   {'MEMTILE ' * N_COLS}  (L2 weight/KV staging)")
    print(f"  r0   {'SHIM    ' * N_COLS}  (DRAM, 8-col fan-out)")
    used = sum(1 for u, _, _ in _all_compute_tiles(placement) if u != "spare")
    print(f"  tiles used: {used}/32   spare: {32 - used}   "
          f"proj: {len(placement['projection'])}   attn pairs: {len(placement['attention'])}")


if __name__ == "__main__":
    render_diagram(DECODE_PLACEMENT, "decode")
    print()
    render_diagram(PREFILL_PLACEMENT, "prefill")
    print(f"\n  FlowKV flows: decode={len(DECODE_FLOWS)} pairs, prefill={len(PREFILL_FLOWS)} pairs")
    print(f"\n  layer-op -> functional unit (shared by both overlays):")
    for op in spec.LAYER_OPS:
        kind = type(op).__name__
        print(f"    {op.name:24s} {kind:12s} -> {OP_UNIT[kind]}")
