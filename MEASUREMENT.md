# Measuring the field-level set-register-field helper

## What this op is, for measurement purposes

`RegisterField.h`/`.cpp` is a compile-time IR-generation helper, not a new
runtime op. It changes how two existing lowerings (`AIELowerCoreReset`,
`AIELowerDmaChannelReset`) *build* their `npu.maskwrite32`s; it does not
change what gets executed on the NPU. For both shipped callers the emitted
IR is byte-identical before and after the refactor (confirmed: rebuilt
`aie-opt` and re-ran both `lower-core-reset` and `lower-dma-channel-reset`
FileCheck tests, which check exact address/value/mask constants and op
order -- both pass unchanged). So there is nothing new to observe on
hardware from these two callers alone; the device-test question is really
"does the helper's shift/mask derivation stay correct for a caller this
header does not have yet."

## Device test

### What already covers today's callers

`test/npu-xrt/local_reset/{core,dma,core_reset_op,dma_channel_reset_op}`
already on-board test the exact `maskwrite32` pairs `createMaskWriteField`
emits for `CoreReset`/`DmaChannelReset`, each as a correct-with/drift-without
pair (e.g. `dma_channel_reset_op`: reset-then-good-BD collects
`[100..107]`; no reset collects the bad BD's `[900..907]` instead -- wrong
data, not a hang). Because the refactor is IR-byte-identical, these four
tests remain valid, currently-passing on-board evidence that the helper's
shift/mask math lands the reset bit correctly and preserves the sibling
field (`ENABLE`, `CONTROLLER_ID`/`FOT_MODE`/...) on real aie2p silicon. This
is regression coverage, not new coverage, and I did not re-add it.

### What is not covered yet, honestly

Neither shipped caller threads a caller-controlled, multi-bit value through
a shared field -- both only ever pass a fixed 0 or 1 into a 1-bit field. The
class of bug this header exists to prevent (a value that is in-range but
lands on the wrong bits because lsb and mask disagree, or a non-contiguous
mask) is therefore not reachable through any op that exists today, on
device or off. I am not inventing a synthetic op purely to manufacture that
coverage -- that was the implementer's call in the original commit and I
agree with it (a test-only op is exactly the kind of speculative surface
the project's earn-from-instance rule warns against).

### The device test to add once a real consumer lands

`aiex.set_bd_length` (named as the next consumer in the original commit) is
the natural trigger. Model it directly on `dma_channel_reset_op`:

- **Correct-with:** set a BD's length to a value that is not a clean
  power-of-two (e.g. 0x64 = 100, so a wrong shift amount produces a visibly
  different, non-zero, wrong byte count instead of coincidentally landing on
  the right answer) via `aiex.set_bd_length`, run the transfer, and check the
  collected byte count and content match exactly.
- **Drift-without:** temporarily reintroduce the exact bug this refactor
  closes -- restore a caller-suppliable `lsb` that can disagree with `mask`
  (or hand-write the pre-helper raw shift/mask) -- and show the same test
  now moves the wrong number of bytes (the red team's worked example: 100
  requested, 0 delivered, because `(100 << 0) & 0x1FFFF800 == 0` when lsb
  should have been 11). This reproduces the hazard on real hardware, not
  just in the CppTest.
- Gate: exact equality between requested and observed length/content. No
  tolerance -- register field values are exact integers, there is no
  meaningful "close enough."

## Quiescing before any on-device run

The NPU is single-tenant and shared. Before either the existing local_reset
on-board tests or the future set_bd_length test: announce on the shared
channel, `fuser` the XRT device node to confirm nothing else has it open,
stop `npu-asr`/`vox` if running, and do not auto-restart them after. Serialize
-- no concurrent device access, including from a second terminal running an
unrelated NPU task. This op does not need RAPL/energy quiescing (it makes no
energy claim, see below), just exclusive device access for a clean run.

## Performance test -- honest answer: there isn't a speedup to gate

This is a correctness/ergonomics refactor with no performance effect, by
construction: for existing callers the emitted `npu.maskwrite32` sequence is
byte-identical, and for a future caller the helper emits exactly the
instruction a hand-coded lowering would have emitted -- same opcode, same
operand count, same bytes moved. There is no mechanism by which this change
could move tok/s or ms/token in the resident-decode pipeline, and I am not
going to invent one.

The only numeric gate that means anything here is a **non-regression**
gate, not a speedup gate:

- **Metric:** compiled instruction sequence for the two refactored lowering
  paths (core reset pulse, DMA channel reset pulse) -- opcode, operand
  count, and byte size.
- **How measured:** `aie-opt --aie-lower-core-reset` /
  `--aie-lower-dma-channel-reset` output diffed against the pre-refactor
  lowering (this is what the two FileCheck tests already check, statically,
  with no device time required). Optionally, on-board wall-clock of the
  reset-pulse phase in the existing `local_reset` tests, before vs after.
- **Pass threshold:** bit-identical IR (hard requirement, already met by
  both FileCheck tests). If wall-clock is measured on-device as a
  belt-and-suspenders check, delta must fall within +/-2% of the
  pre-refactor baseline (noise floor for two `maskwrite32`s) -- expected
  result is ~0% since the IR did not change.

Do not report this op as a latency or throughput win anywhere. It buys
correctness margin on future single-bitfield lowerings, nothing else.
