"""
Tile-mapping diagram: yolo26n-cls XINT8 on Ryzen AI NPU2 Strix Point.

Models the 8-column × 6-row tile array (rows 0=shim, 1=memtile, 2–5=compute)
and assigns each yolo26n-cls block (model.0–model.10) to a set of compute tiles,
based on:
  - Per-block weight + peak-activation sizing (see BLOCKS table below)
  - MobileNet V3 reference variant sizing (A=1 tile, B=3 tiles, C=5 tiles)
  - mlir-aie placement convention (tile(col, row), column-major packing,
    cascade_flow between adjacent put/get tiles)

Run to print the diagram and per-block stats:
    python tile_layout.py
"""

from dataclasses import dataclass


@dataclass
class Block:
    name: str  # e.g. "m.0", "m.9 attn"
    sub: str  # short tag for diagram cell, e.g. "m9-A1", "spare"
    variant: str  # "A" / "B" / "C" / "head" / "PSA"
    n_tiles: int
    wt_bytes: int  # INT8 weight+bias bytes for whole block
    gmacs: float  # total GMACs per inference (batch=4)
    peak_act_bytes: int  # peak live in+out activation (INT8 bytes)
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-block sizing (batch=4, imgsz=512) — weights + peak live activation
# ---------------------------------------------------------------------------
BLOCKS = [
    Block(
        "m.0",
        "m0",
        "A",
        1,
        448,
        0.113,
        7 * (1 << 20),
        "InitConv 3→16, stride 2; act-dominated",
    ),
    Block("m.1", "m1", "A", 1, 4_530, 0.302, 6 * (1 << 20), "Conv 16→32, stride 2"),
    Block(
        "m.2",
        "m2",
        "B",
        3,
        6_370,
        0.419,
        7 * (1 << 20),
        "C3k2: cv1 + 2 inner Convs + cv2",
    ),
    Block("m.3", "m3", "A", 1, 36_060, 0.604, 5 * (1 << 20), "Conv 64→64, stride 2"),
    Block(
        "m.4",
        "m4",
        "B",
        3,
        25_230,
        0.419,
        int(3.5 * (1 << 20)),
        "C3k2: cv1 + 2 inner Convs + cv2",
    ),
    Block(
        "m.5",
        "m5",
        "A",
        1,
        144_120,
        0.604,
        int(2.5 * (1 << 20)),
        "Conv 128→128, stride 2",
    ),
    Block(
        "m.6",
        "m6",
        "C",
        5,
        84_500,
        0.352,
        int(1.25 * (1 << 20)),
        "C3k2 heavy: cv1 + 6 inner Convs + cv2",
    ),
    Block(
        "m.7", "m7", "A", 1, 288_250, 0.302, 768 * (1 << 10), "Conv 128→256, stride 2"
    ),
    Block(
        "m.8",
        "m8",
        "C",
        5,
        337_000,
        0.352,
        640 * (1 << 10),
        "C3k2 heavy: cv1 + 6 inner Convs + cv2",
    ),
    Block(
        "m.9",
        "m9 PSA",
        "PSA",
        5,
        242_500,
        0.303,
        576 * (1 << 10),
        "PSA: cv1, qkv, attn (Q@K-soft-V), pe, proj, ffn, cv2",
    ),
    Block(
        "m.10",
        "m10",
        "head",
        2,
        323_750,
        0.336,
        int(1.5 * (1 << 20)),
        "Head Conv 256→1280 + GAP + Gemm 1280→2",
    ),
]

# ---------------------------------------------------------------------------
# Tile array layout: tile(col, row), col in [0..7], row in [0..5]
#   row 0 = shim (DRAM interface)
#   row 1 = memory tile (1 per column)
#   rows 2–5 = compute (4 per column × 8 cols = 32)
# Naming style: "<block>-<sub>" e.g. "m6-A", "m6-B", ...
# "----" = spare; "MEM"/"SHIM" indicate non-compute rows.
# ---------------------------------------------------------------------------
ROW_LABELS = {
    5: "compute (top)",
    4: "compute",
    3: "compute",
    2: "compute (bot)",
    1: "memory tile",
    0: "shim (DRAM)",
}

# Per-column tile names. Position [col][row-2] for compute rows.
# Layout chosen to keep data flow short: early blocks at low columns, deep
# blocks at high columns, PSA at columns 5-6, head at column 7.
LAYOUT = {
    # col : { row : cell }
    0: {5: "m2-C", 4: "m2-B", 3: "m2-A", 2: "m0"},
    1: {5: "m4-C", 4: "m4-B", 3: "m4-A", 2: "m1"},
    2: {5: "m6-A", 4: "m5", 3: "m3", 2: "----"},
    3: {5: "m6-D", 4: "m6-C", 3: "m6-B", 2: "----"},
    4: {5: "m8-A", 4: "m6-E", 3: "m7", 2: "----"},
    5: {5: "m8-D", 4: "m8-C", 3: "m8-B", 2: "----"},
    6: {5: "m9-att", 4: "m9-qkv", 3: "m9-cv1", 2: "m8-E"},
    7: {5: "m10-2", 4: "m10-1", 3: "m9-ffn", 2: "m9-prj"},
}

# Cascade flows: tile(col_a, row_a) → tile(col_b, row_b)
# Used for cross-column partial-sum streaming (per MobileNet bn13/bn14 pattern).
CASCADES = [
    # Within-column northbound cascades are implicit. Cross-column listed here.
    ((2, 5), (3, 3)),  # m6-A → m6-B   (east jump to col 3 for the rest of m.6)
    ((3, 5), (4, 4)),  # m6-D → m6-E   (east jump to col 4 to finish m.6)
    ((4, 5), (5, 3)),  # m8-A → m8-B   (east jump to col 5 to continue m.8)
    ((5, 5), (6, 2)),  # m8-D → m8-E   (east jump to col 6 for last m.8 tile)
    # PSA internal: qkv → attn core (Q@K → softmax → @V) → projection
    ((6, 4), (6, 5)),  # m9-qkv → m9-att (in column)
    ((6, 5), (7, 2)),  # m9-att → m9-prj
]


def render_diagram():
    """Print a 4-row × 8-col ASCII diagram of the tile array."""
    cell_w = 11
    sep = "+" + ("-" * (cell_w + 1) + "+") * 8

    def fmt_row(row):
        parts = []
        for col in range(8):
            cell = LAYOUT.get(col, {}).get(row, "")
            parts.append(f" {cell:<{cell_w}}")
        return "|" + "|".join(parts) + "|"

    print("                                       NPU2 STRIX POINT  (8 cols × 6 rows)")
    print()
    header = "      " + "".join(f"  col {c:<8}" for c in range(8))
    print(header)
    print("      " + sep)
    for row in range(5, -1, -1):
        if row >= 2:
            print(f" r{row}   " + fmt_row(row) + f"   {ROW_LABELS[row]}")
        else:
            label_cell = "MemTile" if row == 1 else "ShimTile"
            print(
                f" r{row}   |"
                + "|".join(f" {label_cell:<{cell_w}}" for _ in range(8))
                + f"|   {ROW_LABELS[row]}"
            )
        print("      " + sep)


def render_block_stats():
    print()
    print(" Per-block plan (tiles, weights, MACs, peak live act, variant)")
    print(" " + "-" * 100)
    hdr = f" {'block':6s} {'variant':7s} {'tiles':>5s} {'wt+b':>10s} {'GMACs':>7s} {'peak act':>10s}  notes"
    print(hdr)
    print(" " + "-" * 100)
    total_tiles = 0
    for b in BLOCKS:
        total_tiles += b.n_tiles
        wt_kb = b.wt_bytes / 1024
        act = b.peak_act_bytes
        act_str = (
            f"{act / (1<<20):.2f} MB" if act >= (1 << 20) else f"{act / (1<<10):.1f} KB"
        )
        print(
            f" {b.name:6s} {b.variant:7s} {b.n_tiles:>5d} {wt_kb:>8.1f} KB {b.gmacs:>7.3f} {act_str:>10s}  {b.notes}"
        )
    print(" " + "-" * 100)
    print(f" Total compute tiles used: {total_tiles} / 32   ({32 - total_tiles} spare)")


def render_dataflow():
    print()
    print(" Data flow (depth-first, mostly column-major):")
    print()
    flow = [
        "  DRAM ─► shim(c0) ─► memtile(c0) ─► m.0 (col0,r2)",
        "  ─► m.1 (col1,r2) ─► m.2 (col0 r3-5)",
        "  ─► m.3 (col2,r3) ─► m.4 (col1 r3-5)",
        "  ─► m.5 (col2,r4) ─► m.6 (col2 r5, casc→ col3 r3-5, casc→ col4 r4) ─► m.7 (col4,r3)",
        "  ─► m.8 (col4 r5, casc→ col5 r3-5, casc→ col6 r2)",
        "  ─► m.9 PSA (col6 r3-5 + col7 r2-3)",
        "    │  cv1 → qkv → attn core (Q@K, softmax, @V) → pe+proj → ffn → cv2",
        "  ─► m.10 (col7 r4-5: Conv 256→1280 → GAP → Gemm 1280→2)",
        "  ─► memtile(c7) ─► shim(c7) ─► DRAM (logits)",
    ]
    for line in flow:
        print(line)
    print()
    print(" Cascade flows (cross-column partial-sum streams):")
    for a, b in CASCADES:
        print(f"   tile({a[0]},{a[1]}) ──cascade──► tile({b[0]},{b[1]})")


def render_caveats():
    print()
    print(" Open design questions this layout does NOT yet resolve:")
    print("  1. Bias add — MobileNet kernels have NO bias support; 40/42 of our")
    print("     ops have INT8 biases. Need to fork a Conv kernel (cleanest) or add")
    print("     a separate bias-add kernel between Conv and the SRS requant.")
    print("  2. PSA tile budget (5 tiles) may be tight — 7 Convs + 2 MatMul +")
    print("     Softmax. May need to fuse cv1+qkv into 1 tile, or extend to 6 tiles.")
    print("  3. SiLU/HardSigmoid handling — graph uses HardSigmoid+Mul pairs;")
    print("     to compress, write a fused INT8 SiLU LUT kernel.")
    print("  4. m.10 GAP reduction layout — global average over 16×16 spatial")
    print("     should fuse into the tile holding the final Conv output.")


if __name__ == "__main__":
    render_diagram()
    render_block_stats()
    render_dataflow()
    render_caveats()
