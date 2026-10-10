<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# The deadlock model: plan

This is the plan for building the model [DeadlockModel.md](DeadlockModel.md)
specifies, milestone by milestone, with the decisions behind it and where the
work stands. Status as of 2026-10-10: M0 is done and M1 is partway (the
engine and `--aie-check-deadlock` work; front-end coverage and the aiecc gate
remain).

## Goal

For every design, the compiler decides one of three things, and is right:

- **Accept.** No run the design's asserts allow can deadlock.
- **Error.** Every allowed run deadlocks. The diagnostic gives a schedule that
  reaches the deadlock: which agent waits on what, at which point in its
  program.
- **Accept with a runtime guard.** Whether it deadlocks depends on runtime
  scalars. The compiler emits a `cf.assert` in the runtime sequence that holds
  exactly for the safe values, so the host refuses an unsafe dispatch with a
  message instead of hanging the NPU.

"Right" means right with respect to a stated model of the hardware. Whether
the model matches the hardware is checked separately, on the NPU, from M0 on.

Two consumers come first:

- **Shim channel sharing between flows live at the same time.** Today ends
  share a channel only when one is done before the other starts.
- **The router.** Its deadlock rules work on which agents wait on which, not
  on program order, so it refuses some safe designs and lists the assumptions
  it made (`allow-deadlock-prone` exists for those cases).

Later consumers: hub merge order, out-of-order merge, and the objectFIFO
transfer-budget check (`verifyTransferBudgets`), which becomes one query of
the model.

## What exists to build on

| Piece | Where | What it gives | What it leaves to the model |
|---|---|---|---|
| `StreamVolumeAnalysis` | `AIEStreamDependencyAnalysis.{h,cpp}` | bytes a stream sends, bytes a receiver takes before it waits, read off BDs, repeat counts, lock inits, runtime transfers | order between a runtime-issued channel's transfers (`maySendAfter`) |
| `StreamWaitGraph` | same | who waits on whom: locks, streams, host waits | program order within an agent |
| `StreamDeadlockAnalysis`, `StreamConflicts` | same | `canBlock`, `unavoidable`, `holdCycle`: what the router asks, with the assumptions it made listed in `assumptions()` | exact answers where it assumes |
| Router integration | `AIECreatePathFindFlows.cpp` (`planArbiters`, `tooFewArbiters`, the routing-check hook) | arbiter and link hazards, which stay the router's | |
| `ShimTransferSpans` | `AIEShimSharing.{h,cpp}` | when each shim end is in flight, by runtime sequence order and waits | everything but shim ends |
| Runtime asserts | `cf.assert` from the dynamic DMA lowering and IRON's `require` | ranges and divisibility on runtime scalars | |
| Presburger | upstream `mlir/Analysis/Presburger` | emptiness of linear integer constraints with congruences | |
| Reference model and fuzzer | `test/create-packet-flows/nightly/aiemodel/` | a Python model of the router's rules, a fuzzer, shrinking | program order (added in M0) |
| Hardware verdicts | `nightly/hw_verdicts.py`, `Inputs/hw_verdicts/` | designs that passed or hung on npu2, checked against the rules; "cautious" marks passes the rules cannot explain (timing, buffering, hidden programs) | |

The cautious verdicts are exactly what this model should turn into proofs.

## The model in brief

[DeadlockModel.md](DeadlockModel.md) is the specification; this is the shape
the plan relies on.

- **Agents**, each a program run in order: cores, DMA channels (a BD chain or
  the queue of runtime tasks pushed to it) and the host (the runtime
  sequence).
- **Resources**: locks, streams (whose capacity comes from the receiver's
  program plus what the fabric buffers), task queues four deep, and host
  completion tokens.
- **One run decides it.** Where no two agents race for one resource (one
  acquirer per lock side, one sender per receiver, no merge whose order is
  open), the system is a Kahn process network with bounded channels, so
  whether it deadlocks does not depend on the schedule. The model checks that
  condition and names where it does not hold.
- **A dispatch ends** when every host wait is satisfied. At that point every
  channel must be empty, no task queued, and every pool the sequence filled
  drained to the state the next dispatch starts from: "dispatches quiesce",
  checked rather than assumed (D8).
- **Counts, not tokens.** Agents move counts of words and lock tokens; long
  loops are crossed by finding a repeating state.
- **Two front ends, one engine** (D3). The pool level, after split and before
  allocation, lets allocation decide and prove in one pass. The physical
  level, after DMA lowering, serves the router, a final verification and
  hand-written explicit-DMA designs. Running both on one design checks the
  lowering.

## Symbolic counts and guards (M2)

Trip counts and sizes that come from runtime scalars (`DispatchTime`
parameters, scratchpad values, RTP values the sequence writes from them)
become affine expressions in those scalars. The runtime asserts become the
precondition: ranges and congruences ("a multiple of 64") are Presburger
constraints.

Each place the run could block becomes a linear condition, so "the asserts
hold and some schedule deadlocks" is a Presburger set:

- empty: accept;
- everything the asserts allow: error, with a value and the schedule;
- otherwise: emit its complement as a `cf.assert` before the first transfer
  that depends on it.

Products of two runtime scalars (`n * m`) leave Presburger arithmetic. Their
guard is checked at dispatch with the concrete values (D4).

## Outside the deterministic subset

The model must say which of these a design hits, never guess:

1. **Token operations under data-dependent control.** An `scf.if` on data, or
   an early exit, around an acquire, release or stream access (D2).
2. **Kernels that touch streams, cascades or locks.** A core with no
   stream-port end and no cascade has kernels that can only touch memory, so
   it is fully modeled from its own code. A core with them needs a contract:
   tokens per call. Without one, only the obligations whose wait chains pass
   through that core are undecided (D5).
3. **Merges whose order is open.** Fan-in routes, two cores acquiring one
   lock, out-of-order BD selection. M5 adds the forced-order rule (the
   runtime sequence or the design's structure orders the senders); anything
   else is undecided.
4. **Runtime reconfiguration inside a sequence** (`aiex.configure`,
   `load_pdi`). Each configure point ends one run, which must quiesce (D8),
   and starts the next from the new configuration's initial state. Control
   packets that rewrite BDs or switches mid-run stay out of scope until a
   design needs them.

Trace flows fan into a shim channel but carry no data any agent waits on, so
their arrival order is no open merge.

An undecided obligation is an error naming the construct (D9), so the
designer can change it, add the annotation (D2) or contract (D5), or set the
escape flag while the milestone that decides it is pending.

## Milestones

Each one is usable on its own and lands with lit tests, fuzzing and hardware
checks. Because the model gates every design from M1 (D9), the order puts
what shrinks undecided errors first: symbolic counts (IRON's RTP-driven
loops) and kernel contracts come right after M1, and the features built on
the model (live-flow sharing, merges, the router) after those.

**M0. Semantics and oracle.**
- Write the model's semantics down: every event, every blocking rule, the
  queue depth, BD repeat, lock flavors, what await and free promise, packet
  header words, `keep_pkt_header`.
- Split `router_properties.py` into a package (D7), unchanged in behavior,
  then add a reference model that executes small designs by exploring every
  interleaving. It is slow and simple, and it is the oracle the C++ engine is
  fuzzed against.
- Hardware cases for program order: a shim channel head-of-line hang and the
  buffering in front of a receiver.
- Survey the in-tree corpus: how many designs fall in the deterministic
  subset with constant counts, and what keeps the rest out.

**M1. Concrete engine, physical front end, the gate.**
- C++ engine over constant counts; physical front end for cores, BD chains
  and the runtime sequence (design below).
- A verification pass in aiecc's pipeline decides every design (D9): accept,
  error with a schedule, or undecided with the reason, which is an error too.
  One explicit escape flag turns undecided errors into warnings; a proven
  deadlock always errors. Every use of the flag in tree names the milestone
  that removes it.
- The "dispatches quiesce" check (D8).
- Fuzzed against the M0 oracle; every hardware verdict agrees.
- `StreamDeadlockAnalysis::canBlock` asks the engine where the design is in
  the subset, so the router's assumptions shrink to the cases outside it.

**M2. Symbolic counts, Presburger, guards.**
- Parameters from runtime scalars; asserts as preconditions; guard emission;
  the dispatch-time check for conditions outside Presburger (D4).
- IRON's GEMV (`n = rtp[0]`) as the first real case.

**M3. Kernel contracts, branch annotations, partial answers.**
- The contract format, the stream-port/cascade rule, and scoping undecided
  obligations to the cores they pass through (D5).
- The branch token-effect annotation (D2), in the IR and in IRON.
- Contracts for the in-tree streaming kernels (cascade matmul, stream-port
  kernels).

**M4. Pool-level front end and live-flow shim sharing.**
- Allocation shares a shim channel between live flows when the engine proves
  no head-of-line wait. `ShimTransferSpans` gains this as its second tier, and
  the placer and allocation keep agreeing.
- Pool-level and physical-level models of one design must agree (a test over
  the whole objectFIFO lit corpus).
- Hardware: one MM2S channel feeding two consumers by packet id, both live.

**M5. Merges.**
- The forced-order rule; hub merge order proven or refused; this also
  settles the deferred out-of-order merge's head-of-line question.

**M6. The router on exact waits.**
- Packet flows the engine proves never live together share arbiters freely.
- `unavoidable()` reports only real deadlocks, each with its schedule.
- `allow-deadlock-prone` is removed once no in-tree design depends on it.

**M7. Retire what the engine replaces.**
- `verifyTransferBudgets` becomes a query; `StreamWaitGraph` and
  `StreamDeadlockAnalysis` are deleted once the engine covers their uses (D6).
- The escape flag is removed once no in-tree design uses it.

## M1 design

### Where it runs

`--aie-check-deadlock` runs on `aie.device` after the objectFIFO lowering and
before routing: BD chains, locks, cores and `aie.flow` / `aie.packet_flow`
exist, and the runtime sequence still holds `npu.dma_memcpy_nd` and DMA
tasks. aiecc will run it at the start of its routing pipeline
(`getRoutingPipeline`, before `aie-create-pathfinder-flows`).

Options:
- `allow-undecided` (the D9 escape flag): an undecided design warns instead
  of failing. A proven deadlock or an unquiesced dispatch always fails.
- `buffering`: words a path holds in front of its receiver, for the second
  run (default 8, measured on npu2).
- `shim-buffering`: words a shim MM2S channel reads ahead of its stream
  before it reports its task complete (default 1580, measured on npu2).

### Files

- `AIEDeadlockModel.{h,cpp}`: class `DeadlockModel`.
  - The constructor is the physical front end. It reads locks and their
    initial values, BD chains (`aie.dma_start` chains, and `aie.dma` once
    covered), flows and packet flows (sender channel and packet id to
    receivers, and whether each receiver keeps headers), cores and the
    runtime sequence, and records why a design is outside the subset
    (`outside()`). Cores and the runtime sequence go through an interpreter
    of `scf` / `arith` on integers that do not come from memory, since
    lowered objectFIFO cores compute lock amounts from loop-carried values.
    A loop that never exits is found by its carried state repeating.
  - `run(buffering, shimBuffering)` is one run in any order (the subset is a
    Kahn network). Agents move in macro steps: a core runs to its next
    blocking acquire; a stream moves as many words as both ends can take. A
    state that repeats ends the run. It returns the deadlock (each blocked
    agent and what it waits for), or what is still in flight at the end.
- `AIECheckDeadlock.cpp`: the pass. Two runs: the least buffering (0) decides
  accept; when it deadlocks, the run with the measured buffering tells a sure
  deadlock (it also deadlocks) from one that depends on the fabric
  (undecided). Quiesce is checked at the end of the measured-buffering run,
  since more buffering only leaves more in flight. Leftover completion tokens
  count only on channels some sequence waits on.

### Outside the subset in M1 (undecided, named)

- a receiver more than one channel sends to (fan-in): M5;
- a lock more than one agent acquires;
- a lock the host sets;
- counts, sizes or bounds it cannot know (runtime scalars): M2;
- tokens under data-dependent control: D2's annotation, M3;
- a channel a flow names that the design does not program;
- core stream ports and cascades: M3's contracts;
- host register writes (`write32`, `maskwrite32`, `blockwrite`, ...).

### Tests

- `test/deadlock-model/check_deadlock.test`: the pass's diagnostics on hand
  cases and the hardware verdicts.
- `test/deadlock-model/engine_vs_oracle.py`: small generated designs
  (`aiemodel/subset.py`); wherever the pass decides, the oracle must agree.
- The npu-xrt corpus survey: every design the oracle accepts, the engine
  accepts.

## Status

### M0 (done)

- The router's model split into `nightly/aiemodel/` (D7), the nightly
  reports identical before and after.
- `docs/DeadlockModel.md` (the semantics) and the oracle
  (`aiemodel/program.py`, `test/deadlock-model/oracle.py`).
- Hardware, npu2 (`test/npu-xrt/deadlock_model_head_of_line`):
  - Head-of-line blocking on a shared shim channel is real.
  - A packet stream from the shim to a core buffers 8 words in front of the
    receiving BD (8 extra words pass, 9 hang), the same at rows 2 and 5, so
    it is not per hop.
  - A shim MM2S task reports its completion token while up to about 1580 of
    its words are still in the shim. Within one runtime sequence, a sending
    shim end awaited before another starts is therefore not done, so shim
    sharing keeps sending ends in flight to the end of their sequence.

Survey of `test/npu-xrt` (79 hand-written designs, lowered with
`--aie-place-tiles --aie-objectFifo-stateful-transform`, classified by the
oracle's front end): 35 inside the oracle's coverage (none called a
deadlock), 44 outside. Of those, front-end coverage (`aie.dma`, `cf.br`
cores, `npu.sync`, trace configuration) and reconfiguration
(`aiex.configure`, `load_pdi`) are M1's; host register writes and
control-packet ends need the escape flag; runtime counts are M2's.

Engine requirements the survey found:
- Lowered objectFIFO cores compute lock amounts from loop-carried values
  (`maxsi(1 - held, 0)` through `iter_args`).
- Packet headers come from `aie.dma_bd_packet` as well as `dma_bd`'s
  `packet`.
- Locks and buffers can be declared inside DMA regions.
- `hw_verdicts/kh_on_exact` passes on hardware but leaves one word in the
  stream; the model reports it unquiesced, the D8 check working as intended.

### M1 (in progress)

Done:
- `--aie-check-deadlock` with the engine, the interpreter, the two runs and
  the quiesce check.
- Against the oracle on generated designs: 340 compared, 0 disagree.
- On the npu-xrt corpus: 26 of 79 accepted, none called a deadlock.

Next:
1. Front-end coverage, by how many npu-xrt designs each keeps undecided:
   `aie.dma` (14); `aiex.configure` (6), whose regions run other devices'
   sequences, each checked by its own pass; cores with several blocks (5),
   interpreted as a CFG with a repeated block and arguments meaning forever;
   `load_pdi` (4), where the leading `load_pdi @empty; load_pdi @self`
   reset is the initial state; trace configuration in the sequence (3);
   `npu.sync` (1), a host wait on its channel.
2. Several `aie.dma_start`s on one channel queue in their chain's order; the
   front end keeps one per channel today (`dmabd_task_queue`).
3. The aiecc gate (D9): measure the effect on the npu-xrt and python lit
   corpora first, then switch it on with the escape flag where needed, each
   use naming its milestone.

## Risks

- **Model fidelity** is the real risk: anything the model gets wrong about the
  hardware makes it wrong, not merely cautious. Mitigation: the semantics
  document, the hardware verdicts, and the rule that every new modeled
  construct comes with a hardware case.
- **Cost.** Loop acceleration has to work on real designs (MobileNet, Llama
  decode packs) within compile-time budgets. Measure on those from M1.
- **Scope creep.** The deterministic subset, concrete counts and the physical
  front end (M1) must be finished and useful before anything symbolic.
- **The gate from M1 (D9).** Until M2 and M3 land, most dynamic designs are
  undecided, so the escape flag will be widespread; M2 has to follow M1
  closely.
- **Deliberate hangs on hardware** can wedge the driver. Cases expected to
  hang run one at a time, on a machine nobody else is using.

## Decisions

- **D1. Runtime guards.** Accepted: "accept with a runtime guard" is an
  outcome.
- **D2. Token operations under data-dependent control.** An annotation states
  the branch's token effect; the model checks each arm's code against it and
  then trusts it. Without one, an error naming the branch.
- **D3. One engine, two front ends.** Both the pool level (allocation decides
  and proves in one pass) and the physical level (the router, hand-written
  designs), compared on one design as a lowering check.
- **D4. Guards outside Presburger arithmetic.** Checked at dispatch. The
  exact condition is emitted as host code the C++ TXN builder evaluates with
  concrete values; where no closed form exists, the builder runs the engine
  on the concrete counts. Measure the dispatch cost.
- **D5. Kernels that stream without a contract.** An error naming the
  kernel, the obligation and the contract it needs, raised only where an
  obligation passes through its core. Once the model decides every design,
  that is nearly every streaming kernel (M3).
- **D6. The existing wait graph.** Replaced once the engine covers
  `canBlock`, `unavoidable` and the drain waits, keeping one model. The
  engine must therefore meet the router's compile-time budget on real
  designs.
- **D7. Where the reference model lives.** A small Python package under
  `test/create-packet-flows/nightly/`: design builder, target, routing rules
  and the program executor, split out with behavior unchanged before the
  executor was added.
- **D8. The "dispatches quiesce" contract.** Checked. When every host wait
  is satisfied, every channel must be empty, no task queued, and every pool
  the sequence filled drained. A design that leaves something in flight
  errors, naming it; one that quiesces only for some runtime values gets a
  guard.
- **D9. When the model gates every design.** From M1, as an error, with one
  explicit escape flag for undecided designs (never for proven deadlocks).
  Milestones are ordered to shrink its use.
- **Shim sharing of sending ends.** Restricted: a sending (MM2S) shim end
  stays in flight to the end of its runtime sequence, because its completion
  token fires before its words leave the shim.
- **Shim sharing of receiving ends.** Kept, documented: sharing an S2MM end
  assumes its senders don't send outside their turn; M1's engine reports
  fan-in as undecided until M5 proves the order.
