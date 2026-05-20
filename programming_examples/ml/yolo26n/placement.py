"""yolo26n-cls on NPU2 Strix Point — physical tile placement.

Algorithm/mapping split per the mobilenet-py pattern: this file is the ONLY
place tile coordinates appear. Re-targeting the design = editing this dict.

Layout: 8 columns × 6 rows.
  row 0 : shim (DRAM interface)
  row 1 : memtile (1 per column)
  rows 2-5 : compute (4 per col × 8 = 32 tiles)
Adjacent rows in the same column share memory (we exploit for fused pairs).

Compute tile budget: all 32 of 32 used (no spares). Earlier drafts had
4 spare slots on row 2; m8 grew from 5 to 8 tiles (split inner pairs +
dedicated cv3/cv2), m9 grew from 5 to 7 (split attn_core, unfused
proj/ffn/cv2), and m10 fused down to 1 tile — net +4 tiles used.
See tile_layout.py for the ASCII grid (still showing the earlier draft
layout; the current PLACEMENT[block] dict below is the source of truth).

Op-to-tile fusion (more than 1 op per tile in multi-conv blocks):
  c3k2_small  (3 tiles): cv1 / m.0_inner_pair / cv2
  c3k2_heavy  (5 tiles): cv1 / m.0_split / inner_pair_0 / inner_pair_1 / cv3+cv2
  psa         (5 tiles): cv1 / qkv / attn_core(qk+softmax+pe+sv) / ffn / proj+cv2
  head        (2 tiles): conv+avgpool / gemm

The IRON design files load PLACEMENT and `yolo_spec.NETWORK` together; the
spec says what each block computes, this file says where.

To swap in the real IRON Tile class once we integrate, change the local
import below to `from aie.iron.device import Tile`. Until then this file
is loadable without the aie package installed, which lets us static-check
the placement against the spec.

--- DESIGN RULES (learned the hard way) ------------------------------------

Rule: cons-side acquire ordering can deadlock the whole upstream chain.

When a worker's body acquires multiple fifos and one of those acquires
uses a sliding-window pattern (e.g. `acquire(N)` with N>1 then
`release(1)` per iteration), the cons body emits a single `AcquireGE,N`
as its FIRST lock op. Until that AcquireGE fires, every OTHER fifo the
cons would later drain stays full.

If any of those "other" fifos is filled by a producer that is itself
part of the same chain, you get a circular back-pressure deadlock:
  cons waits for N rows on fifo A → upstream producer for A is blocked
  because its own output fifo B (which cons would drain later) is full
  → cons never gets the N rows it needs on A.

Concrete example (m8 stage 4, debugged 2026-05-19):
  `passthrough_drain_top` at (4,4) acquires inner_0_out first (peek-3
  sliding) then top_fifo (acq-1) per emit. With top_fifo depth=4, cv1
  stalls after 4 rows, which back-pressures m_0_split → pair0_cv1 →
  pair0_cv2 → only 2 rows of inner_0_out delivered. Cons waits for 3
  → deadlock. Bumping top_fifo depth from 4 to 8 fixes it (cv1 has
  enough slack to keep the chain flowing).

Mitigations, in order of preference:
  (1) Restructure cons body to interleave acquires across all fifos
      (smallest acquire first, or per-row), so no fifo back-pressures
      while waiting on another.
  (2) Bump the "later-drained" fifos' depth to ≥ (max upstream chain
      latency in rows) for the sliding-acquire window.
  (3) Avoid sliding-window cons on the chain-output fifo entirely
      (use acquire(1)/release(1) on that fifo if the algorithm allows).

The fix was found by a min-depth scan over the chain-output fifo
combined with dummy-mode consumer probes (sliding peek-3 vs peek-2 vs
discrete batch); the trigger is `AcquireGE,N` with N≥3 on the chain
fifo's cons_lock regardless of transport or buffer depth.

Rule: declared-but-unused via_DMA ObjectFifo silently hangs the runtime.

If you create an ObjectFifo with via_DMA=True but never call .prod()
or .cons() on it (i.e. no worker consumes/produces it AND no rt.fill /
rt.drain references it), IRON still allocates a DMA channel, memtile
staging, and a shim BD for it. The shim DMA configuration is built but
never fires, leaving the runtime waiting forever on the next dma_await
of a real fifo that shares the same channel pool.

Concrete example (m9 stage 4, debugged 2026-05-19):
  Stage-3 builder allocated `packed_chunk_fifo` with via_DMA=True for
  qkv_pack's chunked output. Stage 4 replaced the pack worker with
  attn_qk + scores_fifo (also via_DMA) but left `packed_chunk_fifo`
  declared at the top of the stage>=3 block. The orphan fifo's DMA
  channel collided with scores_fifo and the host rt.drain on scores
  timed out (ERT_CMD_STATE_TIMEOUT) — with no compile-time warning.

Mitigation: gate each via_DMA ObjectFifo's *declaration* on a stage
check that exactly matches the stage(s) in which it is actually
produced/consumed. Don't declare-and-orphan based on stage>=N when
stage N+1 replaces the worker. Symptom is a 100% reproducible timeout
that disappears the moment the orphan decl is gated out.

Rule: kernel scalar args that index into a multi-dim L1 buffer must
match the buffer's actual stride dim, not a same-named smaller dim.

If your kernel does `buf[s * N + ...]` and you accidentally pass the
inner-loop bound (e.g. in_w=16) as N instead of the buffer's true row
stride (e.g. 256), the kernel writes to wrong cells silently — no
compile-time check catches it because the C signature is `int8_t*`.

Concrete example (m9 stage 4, debugged 2026-05-19):
  qk_pack writes `qk_frame[s * N + (yi*in_w + x)]`. The kernel was
  called with N=in_w=16 instead of N_tokens=256; row yi=1's writes
  for x=0..15 landed in cells [s*16 + 16 .. s*16 + 31] which alias
  s=1's range, clobbering the previous head's data. Downstream qk_row
  read all-zero/garbage scores → NPU output bit-exactly wrong.
  Bisected via a write-pattern debug kernel that proved chunk DMA
  worked, then an init-value pattern that proved kernel writes weren't
  reaching the buffer at the expected offsets.

Mitigation: name your stride dims unambiguously in the IRON-side call.
Don't reuse `in_w` or `chunk_dim` as the kernel's `N` arg when N is
really the buffer's column count.

Rule: 1KB+ stack arrays in a kernel silently hang AIE2P at runtime.

The default per-tile stack on AIE2P is small (~1KB). A scalar fp32
array like `float exps[256]` declared inside a kernel function is
1024 B on the stack — enough to clobber the return address and cause
a silent runtime hang (no compile-time warning, kernel never returns,
shim DMA times out).

Concrete example (m9 stage 5, debugged 2026-05-19):
  yolo_m9_softmax_row.cc kept `float exps[N=256]` as scratch between
  passes 2 (build exps + sum) and 3 (normalize). HW ran the trivial
  variant fine and the "max scan + LUT lookup" variant fine, but the
  full kernel hung. Reworking softmax as a 2-pass algorithm that
  re-fetches exp(shifted) from the LUT on the normalize pass (no
  stored exps array) immediately unblocked it. The 2-pass form is
  ~50% more compute but uses zero scratch.

Mitigations, in order of preference:
  (1) Restructure the algorithm to avoid the stack array (re-fetch /
      re-compute / fold passes).
  (2) Move the scratch to an IRON-side `Buffer(...)` allocated on the
      tile L1, passed as a kernel argument. Buffer is mutable and
      uses L1 SRAM, not the call stack.
  (3) Bump the per-tile stack size via the IRON API (Worker /
      Program configuration; check aie.iron for stack-size kwargs).
      Useful when neither (1) nor (2) is structurally clean.

Symptom is identical to a DMA back-pressure deadlock (XRT timeout
with no compile error), so this rule is the first thing to check
when a numerically-heavy scalar kernel hangs on HW.
"""

from dataclasses import dataclass

import yolo_spec

# ---------------------------------------------------------------------------
# Tile — use the real IRON class if mlir-aie is installed, else a local
# dataclass so the spec is loadable for static checking outside the IRON env.
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


# ---------------------------------------------------------------------------
# PLACEMENT — keyed by block name from yolo_spec.NETWORK
# ---------------------------------------------------------------------------
PLACEMENT: dict = {
    # ---- Stem + downsamplers (variant A, 1 tile each) ----
    "m0": Tile(0, 2),
    "m1": Tile(1, 2),
    "m3": Tile(2, 3),
    "m5": Tile(2, 4),
    "m7": Tile(4, 3),
    # ---- c3k2 small (3 tiles each) ----
    "m2": {
        "cv1": Tile(0, 3),  # m2-A
        "m_0_inner": Tile(0, 4),  # m2-B  : m.0/cv1 + m.0/cv2 fused (both 3x3)
        "cv2": Tile(0, 5),  # m2-C
    },
    "m4": {
        "cv1": Tile(1, 3),  # m4-A
        "m_0_inner": Tile(1, 4),  # m4-B
        "cv2": Tile(1, 5),  # m4-C
    },
    # ---- c3k2 heavy (5 tiles each) ----
    "m6": {
        "cv1": Tile(2, 5),  # m6-A
        "m_0_split": Tile(3, 3),  # m6-B  : m.0/cv1 + m.0/cv2 fused (parallel 1x1s)
        # inner_pair_0 at (3,2) leaves (4,4) free for m8/cv2 in the chain.
        # split_a (3,3)→(3,2) becomes shared memory (south access).
        "inner_pair_0": Tile(3, 2),  # m6-C
        "inner_pair_1": Tile(3, 5),  # m6-D
        # cv3_cv2 at (3,4) is memory-adjacent to both split_b producer (3,3)
        # and inner_1_out producer (3,5) → those become shared-mem hops,
        # keeping cv3_cv2's S2MM channel count at 2 (top + bot from (2,5)
        # are the only DMA inputs). m6 keeps cv3+cv2 fused on one tile.
        "cv3_cv2": Tile(3, 4),  # m6-E
    },
    "m8": {
        "cv1": Tile(4, 5),  # m8-A
        "m_0_split": Tile(5, 3),  # m8-B
        # m8 also SPLITS each inner pair into separate cv1 + cv2 tiles
        # because the 3x3 weights are too big for L1 even chunked — two
        # weight streams + the activation fifos on one tile blow the
        # per-tile 16-block BD budget. Splitting halves the BD load.
        "inner_pair_0_cv1": Tile(5, 2),  # m8-C0
        "inner_pair_0_cv2": Tile(6, 2),  # m8-C1
        # pair1 originally planned for (5,5)/(4,2) but m8_stage moved
        # to (6,3)/(6,4) during the stage-4 deadlock debug (commit f1c1344).
        # Keep the entries in sync with the actual build so the chain-mode
        # collision detector matches reality.
        "inner_pair_1_cv1": Tile(6, 3),  # m8-D0 (was (5,5))
        "inner_pair_1_cv2": Tile(6, 4),  # m8-D1 (was (4,2))
        # cv3 and cv2 are also split (streamed cv2 weights add a 3rd DMA
        # input that would push a fused cv3_cv2 over its 2-S2MM budget).
        "cv3": Tile(5, 4),  # m8-E0
        "cv2": Tile(4, 4),  # m8-E1
    },
    # ---- PSA (7 tiles, chain-compat) ----
    # Original 5-tile design assumed monolithic attn_core; staged build
    # split attn_core across (attn_core + sv_tile) and unfused proj/ffn/cv2
    # → 7 tiles. Layout below avoids m8's effective footprint (which
    # includes (5,3), (6,3), (6,4) per m8_stage even though
    # placement entry says (5,5), (4,2)) so m0..m9 chain coexists.
    "m9": {
        "cv1": Tile(7, 5),  # col 7 stack: cv1 → qkv → attn_core → sv
        "qkv": Tile(7, 4),  #              shared-mem along the column
        "attn_core": Tile(7, 3),
        "sv": Tile(7, 2),
        # cv2 at (6,5) is shared-mem with BOTH cv1 (7,5)→east and ffn
        # (5,5)→west, so its two activation inputs (top + ffn_block_out)
        # don't burn S2MM channels — only cv2 wts stream uses a channel.
        # proj sits at (4,2), the last remaining free tile.
        "cv2": Tile(6, 5),
        "ffn": Tile(5, 5),  # shared-mem east of cv2 for ffn_block_out
        "proj": Tile(4, 2),  # far from sv/cv1; attn_pre_proj + b via DMA
    },
    # ---- Head (1 tile, fused conv+pool+gemm+softmax) ----
    # Single tile because chain budget only leaves (2,2) free after m9
    # claims 7. m10's pipeline is sequential per-sample anyway so the
    # fused worker has the same wall time as the 2-tile version.
    "m10": {
        "fused": Tile(2, 2),
    },
    # ---- Shim DMA endpoints (row 0) ----
    # SHIM_OUT_COL env override: pin drain column for routing-cost
    # experiments. Default 7 (far corner). Use 0 to test same-col drain.
    "shim": {
        "input": Tile(0, 0),
        "output": Tile(int(__import__("os").environ.get("SHIM_OUT_COL", "7")), 0),
        # Weight DMA columns — first draft spreads loads across all 8 shims;
        # final mapping should bias toward columns near the consuming blocks
        # to shorten the L3 -> L2 -> L1 weight path.
        "wts": [Tile(c, 0) for c in range(8)],
    },
    # ---- Memtiles (row 1) — first-draft minimal mapping; refined per block ----
    "memtile": {
        # One memtile per column is available. Most blocks pull weights from
        # the memtile directly above their shim; PSA + head share col 6/7.
        # Specific block->memtile bindings TBD in the IRON design pass.
        "available": [Tile(c, 1) for c in range(8)],
    },
}


# ---------------------------------------------------------------------------
# Cascades — cross-column partial-sum streams (lifted from tile_layout.CASCADES).
# Used by the IRON design for cascade_flow() connections.
# ---------------------------------------------------------------------------
CASCADES: tuple = (
    (Tile(2, 5), Tile(3, 3)),  # m6: cv1 -> m_0_split
    (Tile(3, 5), Tile(4, 4)),  # m6: inner_pair_1 -> cv3_cv2
    (Tile(4, 5), Tile(5, 3)),  # m8: cv1 -> m_0_split
    (Tile(5, 5), Tile(6, 2)),  # m8: inner_pair_1 -> cv3_cv2
    (Tile(6, 4), Tile(6, 5)),  # m9: qkv -> attn_core (in-column)
    (Tile(6, 5), Tile(7, 2)),  # m9: attn_core -> proj_cv2
)


# ---------------------------------------------------------------------------
# Static checks — run on import to catch typos before IRON ever sees the file.
# ---------------------------------------------------------------------------
def _all_compute_tiles() -> list:
    """Flatten PLACEMENT to a list of (block_name, sub_name, Tile) for compute tiles."""
    out = []
    for block_name, val in PLACEMENT.items():
        if block_name in ("shim", "memtile"):
            continue
        if isinstance(val, Tile):
            out.append((block_name, None, val))
        elif isinstance(val, dict):
            for sub, t in val.items():
                out.append((block_name, sub, t))
    return out


def _validate() -> None:
    # (a) every spec block has a placement entry
    spec_blocks = {b.name for b in yolo_spec.NETWORK}
    placed_blocks = {k for k in PLACEMENT.keys() if k not in ("shim", "memtile")}
    missing = spec_blocks - placed_blocks
    extra = placed_blocks - spec_blocks
    assert not missing, f"PLACEMENT missing blocks: {missing}"
    assert not extra, f"PLACEMENT has unknown blocks: {extra}"

    # (b) tile coords in range; rows 2-5 only for compute
    for blk, sub, t in _all_compute_tiles():
        assert 0 <= t.col <= 7, f"{blk}/{sub}: col {t.col} out of range"
        assert 2 <= t.row <= 5, f"{blk}/{sub}: row {t.row} not a compute row"

    # (c) no two compute tile assignments collide
    seen: dict = {}
    for blk, sub, t in _all_compute_tiles():
        key = (t.col, t.row)
        if key in seen:
            other = seen[key]
            raise AssertionError(
                f"tile collision: {blk}/{sub} and {other[0]}/{other[1]} both on {t}"
            )
        seen[key] = (blk, sub)

    # (d) shim/memtile rows
    for t in PLACEMENT["shim"]["wts"] + [
        PLACEMENT["shim"]["input"],
        PLACEMENT["shim"]["output"],
    ]:
        assert t.row == 0, f"shim tile {t} not on row 0"
    for t in PLACEMENT["memtile"]["available"]:
        assert t.row == 1, f"memtile {t} not on row 1"


_validate()


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    compute = _all_compute_tiles()
    print(
        f"placement: {len(compute)} compute tiles assigned across {len(set(c.col for _,_,c in compute))} columns"
    )
    spare = {(c, r) for c in range(8) for r in range(2, 6)} - {
        (t.col, t.row) for _, _, t in compute
    }
    print(f"  spare compute tiles ({len(spare)}/32): {sorted(spare)}")
    print(f"  cascades: {len(CASCADES)}")
    for blk in yolo_spec.BLOCK_NAMES:
        val = PLACEMENT[blk]
        if isinstance(val, Tile):
            print(f"  {blk:5s} {val}")
        else:
            tiles = ", ".join(f"{k}={v}" for k, v in val.items())
            print(f"  {blk:5s} {{ {tiles} }}")
