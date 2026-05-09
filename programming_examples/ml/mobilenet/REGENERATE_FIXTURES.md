# Regenerating brevitas reference fixtures

The IRON mobilenet design (`aie2_mobilenet_iron.py`) consumes the same
brevitas-quantized fixtures the original placed-API design did:

| File | Source | Consumer |
|---|---|---|
| `data/scale_factors_final.json` | brevitas calibration | IRON + placed-API + numpy reference |
| `data/bnN_chain.txt` (and `*_1/2/3/put/get_*`) | brevitas-quantized weights, OIYXI8O8 reordered | IRON + placed-API |
| `data/before_ifm_mem_fmt_1x1.txt` | quantized input tensor | IRON + placed-API + numpy reference |
| `data/golden_output.txt` | brevitas inference output | end-to-end correctness check |

These fixtures are produced by `gen_golden.py` (top level, ~2630 lines, uses
PyTorch + brevitas + a calibration image). Same script, same output, regardless
of which design downstream consumes the files.

## Top-level fixtures (full mobilenet)

```bash
cd programming_examples/ml/mobilenet
pip install -r requirements_gen_golden.txt    # torch + torchvision + brevitas + onnx
python3 gen_golden.py
# writes data/scale_factors_final.json + data/bn*.txt + data/golden_output.txt
```

After running this, both `aie2_mobilenet.py` (placed-API) and
`aie2_mobilenet_iron.py` (IRON) read from `data/`.

## Per-block fixtures (used by per-block hardware lit tests)

The per-block fixtures under `bottleneck_A/data/`, `bottleneck_B/data/`,
`bottleneck_C/data/` are calibrated SEPARATELY (each per-bn brevitas run picks
its own scales for its block in isolation). They are produced by:

```bash
cd bottleneck_A && python3 gen_golden_bnN.py    # one per bn = bn0,1,2,3,6,7,8_9
cd bottleneck_B && python3 gen_golden.py        # bn10..bn12 chain
cd bottleneck_C && python3 gen_golden.py        # bn13..bn14 chain
```

These produce `input_bnN_single.txt`, `bnN_single.txt`, `golden_output_bnN_single.txt`
(per-bn) or `before_ifm_mem_fmt_1x1.txt` + `golden_output.txt` (per-chain).

## IRON consumer summary

  - Full mobilenet:  `aie2_mobilenet_iron.py` reads `data/`.
  - Per-block tests: `aie2_iron_per_block.py` (with `--data-dir` + `--scales-json`)
                     reads `bottleneck_A/data/` (see `run_per_block_e2e.lit`).
  - Per-chain tests: `aie2_iron_chain.py` (with `--data-dir` + `--scales-json`)
                     reads `bottleneck_B/data/` or `bottleneck_C/data/`
                     (see `run_chain_e2e.lit`).
  - Numpy reference: `mobilenet_numpy.py` reads `data/`. The verification
                     script `test_numpy_per_bn.py` reads `bottleneck_*/data/`
                     directly with their own scale_factors.json.

So **IRON has full parity** with the original on fixture regeneration: nothing
is IRON-specific; the existing `gen_golden*.py` scripts produce everything
both implementations need.
