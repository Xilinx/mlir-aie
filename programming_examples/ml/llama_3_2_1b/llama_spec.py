"""Llama 3.2 1B logical network specification for AIE2P NPU2 (Strix Point).

Pure Python, no AIE imports. Mirrors the structure of
miniature-fishstick/npu2/yolo_spec.py so the same algorithm/mapping split
applies:

    llama_spec.py             - algorithm (this file)
    llama_capacity_analysis.py - feasibility math against AIE2P envelope
    llama_placement.py        - physical tile assignment (decode + prefill)
    aie2_llama_iron.py        - placement-level IRON skeleton + real-build stub
    aie2_llama_layer.py       - runnable one-layer dataflow design (vs iron_mock)

Config source: IRON/iron/applications/llama_3.2_1b/llama_inference_harness.py
(vocab=128256, emb=2048, layers=16, n_heads=32, n_kv_groups=8, head_dim=64,
hidden=8192, rope_base=500000, ctx=131072). Llama 3.2 1B ties input embedding
to lm_head (one matrix serves both).

The 16 transformer layers are structurally identical, so we describe ONE
layer here plus the embedding/lm_head pre- and post-processing. Shapes are
parameterized by M (sequence length) so the same spec covers prefill
(M = prompt_len) and decode (M = 1).

Op naming follows the standard HuggingFace Llama module path so weights map
cleanly: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
input_layernorm, post_attention_layernorm.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Model config (Llama 3.2 1B)
# ---------------------------------------------------------------------------
VOCAB_SIZE      = 128_256
EMB_DIM         = 2_048          # hidden_size
N_LAYERS        = 16
N_HEADS         = 32
N_KV_GROUPS     = 8              # num_key_value_heads (GQA: 32/8 = 4-way share)
HEAD_DIM        = EMB_DIM // N_HEADS  # 64
HIDDEN_DIM      = 8_192          # FFN intermediate_size
ROPE_BASE       = 500_000.0
MAX_CONTEXT     = 131_072
TIE_EMBED_LMHEAD = True          # Llama 3.2 1B specifically

Q_DIM  = N_HEADS    * HEAD_DIM   # 2048
KV_DIM = N_KV_GROUPS * HEAD_DIM  # 512


# ---------------------------------------------------------------------------
# Op descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Linear:
    """A weighted linear projection (Q/K/V/O/gate/up/down/lm_head).

    Treated as (M, K) -> (M, N) GEMM. Weight shape is (N, K) in HF convention.
    AIE2P target: INT8 x INT8 -> INT32 accum -> INT8 with right-shift requant.
    """

    name: str
    in_dim: int
    out_dim: int
    weight_name: str = ""          # HF safetensors key, for the loader
    bias: bool = False             # Llama: no bias on any linear
    activation: Optional[str] = None  # set on FFN gate output if absorbed


@dataclass(frozen=True)
class MatMul:
    """Activation-by-activation matmul (Q @ Kᵀ and Softmax @ V).

    No weights; both operands are runtime tensors. This is the YOLO m9 PSA
    pattern -- same dual-scale-signature wiring problem.
    """

    name: str
    a_shape: tuple   # symbolic shape, may contain "M" placeholder
    b_shape: tuple
    out_shape: tuple


@dataclass(frozen=True)
class RMSNorm:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps). One weight vector of length C.

    Compute: reduction over C, divide, scale. Tiny weight (C bytes at INT8).
    Numerically sensitive -- typical to keep in bf16/fp32 even if weights INT8.
    """

    name: str
    dim: int
    weight_name: str = ""
    eps: float = 1e-5


@dataclass(frozen=True)
class RoPE:
    """Rotary position embedding applied to Q and K. No weights -- angle LUT
    is a constant table of size (max_ctx, head_dim/2) precomputed."""

    name: str
    n_heads: int        # 32 for Q, 8 for K (post-GQA)
    head_dim: int       # 64
    seq_dim: str = "M"  # sequence length, parameterized


@dataclass(frozen=True)
class Softmax:
    """Numerically-stable softmax along the last (key) dim of attention scores.

    Standard practice on AIE: stays in bf16 or fp32; INT8->bf16 on input
    (via dequant from acc32 of Q@Kᵀ), bf16 internal, bf16->INT8 on output
    (with per-row max/sum tracked for stability).
    """

    name: str
    shape: tuple        # (B, H, M, M) for full; (B, H, 1, M) for decode


@dataclass(frozen=True)
class SiLUMul:
    """SwiGLU activation: silu(gate) * up, elementwise. No weights."""

    name: str
    shape: tuple


@dataclass(frozen=True)
class ResidualAdd:
    """x = x + residual; elementwise INT8 add (or upcast to INT16 to avoid
    saturation if both inputs are full-range INT8)."""

    name: str
    shape: tuple


@dataclass(frozen=True)
class Embedding:
    """Token-id -> hidden vector lookup. Single row read per token; bandwidth
    is per-token negligible, but the table is huge (vocab x emb)."""

    name: str
    vocab: int
    dim: int
    weight_name: str = ""


# ---------------------------------------------------------------------------
# A single Llama transformer layer (one of 16; all identical).
#
# Shapes use "M" as a placeholder for sequence length.
#   prefill: M = prompt_len
#   decode:  M = 1   (KV cache of length L provides past keys/values)
#
# For attention shapes we also track "L" = current KV cache length (decode
# only); for prefill, L = M (causal mask handles the triangle).
# ---------------------------------------------------------------------------
LAYER_OPS: tuple = (
    # --- Attention block ---------------------------------------------------
    RMSNorm("input_layernorm", EMB_DIM,
            weight_name="model.layers.{i}.input_layernorm.weight"),

    # QKV projections. K, V are smaller than Q due to GQA.
    Linear("q_proj", EMB_DIM, Q_DIM,
           weight_name="model.layers.{i}.self_attn.q_proj.weight"),
    Linear("k_proj", EMB_DIM, KV_DIM,
           weight_name="model.layers.{i}.self_attn.k_proj.weight"),
    Linear("v_proj", EMB_DIM, KV_DIM,
           weight_name="model.layers.{i}.self_attn.v_proj.weight"),

    # RoPE on Q and K only (V passes through).
    RoPE("rope_q", n_heads=N_HEADS,    head_dim=HEAD_DIM),
    RoPE("rope_k", n_heads=N_KV_GROUPS, head_dim=HEAD_DIM),

    # KV cache concat is implicit (not a tile op; it's a memory move).
    # After concat: K, V have shape ("L+M", N_KV_GROUPS, HEAD_DIM).

    # Q @ Kᵀ -- the YOLO m9/attn/qk analogue. GQA: each KV head serves 4 Q heads.
    # Shapes: Q=(B, N_HEADS, M, HEAD_DIM), K=(B, N_KV_GROUPS, L+M, HEAD_DIM).
    # In practice the kernel either repeats K (B, N_HEADS, L+M, HEAD_DIM) or
    # broadcasts the GQA inside the tile.
    MatMul("attn_qk",
           a_shape=("B", N_HEADS, "M", HEAD_DIM),       # Q
           b_shape=("B", N_KV_GROUPS, "L+M", HEAD_DIM), # K (GQA-broadcast)
           out_shape=("B", N_HEADS, "M", "L+M")),

    # Numerically-stable softmax along L+M dim. Includes causal mask for
    # prefill (decode is always [1, L+1] so no masking).
    Softmax("attn_softmax", shape=("B", N_HEADS, "M", "L+M")),

    # Softmax @ V -- the YOLO m9/attn/sv analogue.
    MatMul("attn_sv",
           a_shape=("B", N_HEADS, "M", "L+M"),
           b_shape=("B", N_KV_GROUPS, "L+M", HEAD_DIM),
           out_shape=("B", N_HEADS, "M", HEAD_DIM)),

    # Output projection: re-mix heads back to emb_dim.
    Linear("o_proj", Q_DIM, EMB_DIM,
           weight_name="model.layers.{i}.self_attn.o_proj.weight"),

    ResidualAdd("attn_residual", shape=("B", "M", EMB_DIM)),

    # --- FFN (SwiGLU) ------------------------------------------------------
    RMSNorm("post_attention_layernorm", EMB_DIM,
            weight_name="model.layers.{i}.post_attention_layernorm.weight"),

    # gate and up share input; can be packed into one (K=emb, N=2*hidden) GEMM.
    Linear("gate_proj", EMB_DIM, HIDDEN_DIM,
           weight_name="model.layers.{i}.mlp.gate_proj.weight"),
    Linear("up_proj",   EMB_DIM, HIDDEN_DIM,
           weight_name="model.layers.{i}.mlp.up_proj.weight"),
    SiLUMul("silu_mul", shape=("B", "M", HIDDEN_DIM)),
    Linear("down_proj", HIDDEN_DIM, EMB_DIM,
           weight_name="model.layers.{i}.mlp.down_proj.weight"),

    ResidualAdd("ffn_residual", shape=("B", "M", EMB_DIM)),
)


# ---------------------------------------------------------------------------
# Network-level pre/post that wraps the 16-layer stack.
# ---------------------------------------------------------------------------
PRE_OPS: tuple = (
    # Embedding lookup: token_id -> (B, M, EMB_DIM). Trivial bandwidth per
    # token (one row of the huge table), but the table itself dominates the
    # non-layer parameter count.
    Embedding("embed_tokens", vocab=VOCAB_SIZE, dim=EMB_DIM,
              weight_name="model.embed_tokens.weight"),
)

POST_OPS: tuple = (
    # Final RMSNorm before the output head.
    RMSNorm("model.norm", EMB_DIM, weight_name="model.norm.weight"),
    # lm_head: project to vocab logits. TIED to embed_tokens in Llama 3.2 1B,
    # so no new weight bytes -- but the matmul is huge (M x EMB x VOCAB).
    Linear("lm_head", EMB_DIM, VOCAB_SIZE,
           weight_name="model.embed_tokens.weight"),  # same tensor, transposed
)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
def linear_ops(ops=LAYER_OPS):
    return [o for o in ops if isinstance(o, Linear)]


def matmul_ops(ops=LAYER_OPS):
    return [o for o in ops if isinstance(o, MatMul)]


def layer_weight_params() -> int:
    """Sum of all Linear weight params in one layer. Llama 3.2 1B = 60.8M
    (4M+1M+1M+4M Q/K/V/O attn + 16M+16M+16M gate/up/down FFN)."""
    return sum(o.in_dim * o.out_dim for o in linear_ops(LAYER_OPS))


def rmsnorm_weight_params() -> int:
    """Sum of RMSNorm weight params in one layer (2 norms x EMB_DIM)."""
    return sum(o.dim for o in LAYER_OPS if isinstance(o, RMSNorm))


def network_weight_params() -> int:
    """All weight params in the model. Embedding shared with lm_head per
    TIE_EMBED_LMHEAD; counted once."""
    per_layer = layer_weight_params() + rmsnorm_weight_params()
    embed = VOCAB_SIZE * EMB_DIM
    final_norm = EMB_DIM
    if TIE_EMBED_LMHEAD:
        return N_LAYERS * per_layer + embed + final_norm
    return N_LAYERS * per_layer + 2 * embed + final_norm


# ---------------------------------------------------------------------------
# Self-check (run on import; cheap)
# ---------------------------------------------------------------------------
def _self_check():
    # Linear counts: 4 attn + 3 ffn = 7 weighted linears per layer
    assert len(linear_ops(LAYER_OPS)) == 7, len(linear_ops(LAYER_OPS))
    # MatMul counts: Q@K and S@V
    assert len(matmul_ops(LAYER_OPS)) == 2, len(matmul_ops(LAYER_OPS))
    # Per-layer param count: 4M + 1M + 1M + 4M + 16M + 16M + 16M = 58M
    expected_layer = (
        EMB_DIM * Q_DIM           # q_proj
        + EMB_DIM * KV_DIM        # k_proj
        + EMB_DIM * KV_DIM        # v_proj
        + Q_DIM * EMB_DIM         # o_proj
        + EMB_DIM * HIDDEN_DIM    # gate_proj
        + EMB_DIM * HIDDEN_DIM    # up_proj
        + HIDDEN_DIM * EMB_DIM    # down_proj
    )
    assert layer_weight_params() == expected_layer, (
        layer_weight_params(), expected_layer)
    # 60_817_408 expected (4M + 1M + 1M + 4M + 16M + 16M + 16M, exactly)
    assert expected_layer == 60_817_408, expected_layer


_self_check()


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Llama 3.2 1B spec: {N_LAYERS} identical transformer layers")
    print(f"  emb={EMB_DIM}  hidden={HIDDEN_DIM}  vocab={VOCAB_SIZE}")
    print(f"  n_heads={N_HEADS} (Q)  n_kv_groups={N_KV_GROUPS} (K,V)  head_dim={HEAD_DIM}")
    print(f"  tie_embed_lmhead={TIE_EMBED_LMHEAD}")
    print()
    print(f"Per-layer ops ({len(LAYER_OPS)}):")
    for op in LAYER_OPS:
        kind = type(op).__name__
        extra = ""
        if isinstance(op, Linear):
            params = op.in_dim * op.out_dim
            extra = f"  ({op.in_dim} -> {op.out_dim}, {params/1e6:.1f}M params)"
        elif isinstance(op, MatMul):
            extra = f"  a={op.a_shape}  b={op.b_shape}"
        print(f"  {kind:13s} {op.name:30s}{extra}")
    print()
    layer_p = layer_weight_params()
    norm_p = rmsnorm_weight_params()
    net_p = network_weight_params()
    print(f"Per-layer Linear params:   {layer_p/1e6:8.2f} M  ({layer_p} bytes INT8)")
    print(f"Per-layer RMSNorm params:  {norm_p/1e6:8.2f} M  ({norm_p} bytes INT8)")
    print(f"All {N_LAYERS} layers:             {N_LAYERS*(layer_p+norm_p)/1e6:8.2f} M")
    print(f"Embed (tied with lm_head): {VOCAB_SIZE*EMB_DIM/1e6:8.2f} M")
    print(f"Network total:             {net_p/1e6:8.2f} M  ({net_p/(1<<20):.1f} MB INT8)")
