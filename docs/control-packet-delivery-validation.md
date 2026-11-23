<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Control Packet Delivery Validation

## Explicit Interval Contract

The November 22, 2026 implementation replaces activation recovery, start-gating
proofs, resource-activity tracking, and stream-history checking with producer
assertions. AIERT emits control-packet setup before activation. Hardware
barriers complete each phase. The direct transaction path retains its per-tile
write32/blockwrite order.

All packet optimizers use the shared packet interval iterator. `unobserved`
permits ordinary-storage optimization until an action, barrier, or other
operation. `independent` permits different destinations to execute in any order
until the next hardware barrier or other runtime operation. Each destination
retains its actions and their multiplicity. The producer supplies quiescence
between configurations and a control overlay that permits concurrent delivery.
Packet passes do not analyze arbitrary application behavior or repair routes.

The compiler retains dead-write elimination, address ordering, coalescing,
prefix multicast, parallel delivery, and allocator-owned BD chaining. It removes
substring multicast and the separate authored-packet transport path. The
allocator still checks BD ownership, queue capacity, and completion tokens.
Teardown still drains pending lock acquisitions.

The source comparison against `c789eec4c90` removes 2,896 net implementation
and declaration lines. It removes 4,019 net test lines. The implementation
contains 1,826 fewer lines than the rebased delivery implementation at
`6caec84da77`. Interval-contract tests replace tests for the deleted analyses.
Tests for packet encoding, DMA capacity, masked writes, source locations, and
pending-lock teardown remain.

### Llama Measurements

The fresh run `simplify-verified-20261122-ctrlpkt-teardown` uses the pinned
frontend and audited kernel objects described below. The compiler regenerates
decode and prefill images. All six 32-token generations match the saved reference.
The run measures 18.46 tokens/s. Compile and load takes 69.53 seconds, compared
with 95.07 seconds at the preceding resource-state checkpoint. Decode and prefill
compiler times are 15.53 and 46.52 seconds.

Nine-round saved-image repeats execute the candidate and reference consecutively.
They verify unchanged image hashes and reject any compiler invocation. The
candidate images come from `simplify-destinations-20261122-ctrlpkt-teardown`.
The final fresh build produces byte-identical images.

| Image Pair | Tokens/s | First-Token Time (ms) |
| --- | ---: | ---: |
| Resource-state reference `c789eec4c90` | 18.468 | 140.630 |
| Explicit intervals | 18.417 | 137.772 |

The candidate measures 0.27% lower decode throughput and 2.0% lower first-token
time. These measurements do not establish exact throughput equality.

| Image | Multicast Packets | Packet Words | Destination Words | Submissions |
| --- | ---: | ---: | ---: | ---: |
| Decode | 37,784 | 1,105,270 | 993,815 | 4,324 |
| Prefill | 182,696 | 2,064,756 | 3,031,723 | 8,340 |

The instruction verifier finds no protocol violations. Both images retain one
entry PDI load and no inter-application PDI reset. It checks complete lock
summaries and packet completion before core-status and lock-stall polls.
This verifier does not simulate stream routing or application execution.

The final build completes 20 focused compiler tests and five hardware cases.
The hardware cases cover parameter updates between two runs, the master
optimization opt-out, two-column multicast with serial and parallel delivery,
and a pending core lock acquisition at teardown. Two driver tests also cover
standalone control-packet output and full-ELF scalar arguments. This validation
does not run the full lit suite.

The decoder SHA256 is
`102438e016a4bd13540f0076fbae488cadaf8a748e68201bf11d52a41db34fcf`.
The prefill SHA256 is
`7c21dddc1fe029836630c16ba5d659d1e381514e208578d34a40fe1776520d0a`.

### Direct Transactions

Direct configuration output matches byte for byte for all 17 devices in each
saved Llama shape. Both compilers receive the same static device input without
runtime sequences or multicast declarations. Those declarations are outside
the direct writer's contract. The reference compiler uses the same AIERT direct
generator as `c789eec4c90`. The comparison includes payloads and source locations.
The ELF-loading fixture separately produces identical transaction output.

| Input | Output SHA256 |
| --- | --- |
| Llama decode static devices | `89083f77d5a51de74725a1f48b983e7d84b9d05a1489527aa86df92231d4fb5e` |
| Llama prefill static devices | `119dee753a36d9617bc65254144c9e32ab7c9c4fcb3fd4c221a4ae5344503d36` |
| ELF-loading fixture | `4954797a3db24204a815c821926cb01584d25098c3e706551445157c6ff12d11` |

The implementation does not change direct-write ordering. This comparison does
not measure a reordered direct-write variant.

The following sections record historical implementations and measurements.

## Resource-State Review

The six-finding review used `c1d71b45f13`. Commit `13a9979208a` addressed
findings 1 and 3. Nested packet execution already produced an explicit
diagnostic rather than silently losing context. The added loop-unrolling
regression checks nested parameter writes and their sync boundaries.
Findings 2, 4, and 5 required further changes.

Authored and generated packets use one resource-state analysis. Ordinary
writes retain their order and multiplicity when activity is unknown. An
explicit quiescence assertion permits ordinary-write optimization until
activation affects those resources. The analysis includes multicast
destinations, memory affinity, and switch dependencies. A sync remains a
boundary but does not assert array-wide inactivity. A hardware regression
executes one run, awaits completion, updates two core parameters, and executes
a second run. The host checks both results.

The multicast planner retains the legal prefix before an activation that
would block remaining control traffic. It also separates prefixes from
existing multicast packets whose routes share an arbiter.
The descriptor allocator warns about
unfinished runtime transfers at device markers, using its existing completion
records. It does not infer completion from distinct descriptor IDs. Llama
emits these warnings for transfers whose completion the allocator cannot
prove. Matching output does not establish completion for arbitrary overlap.

`--ctrlpkt-optimize=false` disables optional packet transformations, chaining,
and parallel delivery. It retains packet legalization, synchronization,
allocation checks, and mandatory delivery validation. The same two-run
parameter regression also executes with this option.

The fresh result `review2-verified-20261121-ctrlpkt-teardown` produces the
expected 192 token IDs across six generations. Five measured generations
have a median rate of 18.47 tokens/s, versus 18.80 in the preceding review
run. Compile and load takes 95.07 seconds. Decode compilation takes
18.04 seconds; prefill takes 66.46 seconds.

| Image | Multicast Packets | Packet Words | Destination Words | Submissions |
| --- | ---: | ---: | ---: | ---: |
| Decode | 33,718 | 1,162,150 | 994,207 | 3,960 |
| Prefill | 167,462 | 2,294,526 | 3,036,003 | 7,384 |

The inactivity checks retain more writes and add completion waits. This
measurement does not establish optimization parity with every delivery
baseline opportunity. The instruction checks require complete lock summaries,
packet completion before status polls, and no inter-application PDI reset.
Five hardware cases cover program multicast, unicast, and runtime parameter
updates. The full lit suite does not form part of this check.

The decoder image SHA256 is
`8bad5a3324eb9990ebf367d3b85ebafa18575cb3f881d42b72bd40bf3cd194ac`.
The prefill image SHA256 is
`29091c0d6c416b8a1e3951366c2f97c4ee0fe767b9587b3ab7a7cc883f493851`.

## Review Follow-up

The review compared `4f17842e3ac` with delivery. Findings 1, 2, and 3 also
applied to `c1d71b45f13`. Commit `eb5ba1cec78` already addressed finding 4
by checking multicast candidates against active streams before replacement.

- Activation scheduling applies only to the initial packet run immediately
	after a device marker. Later packets retain device context without gaining
	permission to reorder activation. Tests retain enable/disable order after
	an application write and after an explicit barrier.
- The gating proof includes cores with opaque calls, inline assembly, or
	external ELF programs as possible lock releasers within their memory
	affinity. An acquire-only contract does not restrict those releases.
	Tests cover direct, indirect, and LLVM calls and transitive starter lists.
- Delivery analysis and multicast planning retain stream history across
	ordinary runtime operations. Host effects can release armed gates and
	invalidate initial-lock proofs for subsequent starts. Tests cover writes,
	DMA setup, releases before and after enable, and overwritten initializers.
- Complete generated teardown emits a configuration-specific `drains`
	assertion after its completion polls. The assertion expires only declared
	static cores and non-shim DMA channels. Other configurations and uncovered
	channels remain live. Quiescence alone does not expire streams. Different
	PDI IDs or sizes do not prove a reset. Tests cover these boundaries and
	reject nested configuration or packet execution before analysis.
- Multicast fallback retains unicast when an earlier stream can block only
	the multicast tree. Tests include a non-wait write between the stream start
	and candidate packets, with prefix and substring matching.

The fresh result `review-findings-final-20261121-ctrlpkt-teardown` checks
six 32-token generations against the reference. Five measured generations
have a median rate of 18.80 tokens/s. Compile and load takes 91.08 seconds.
Decode compilation takes 17.62 seconds; prefill takes 63.88 seconds.

The decoder contains 35,826 multicast packets, 1,101,377 encoded packet
words, and 3,409 submissions. Prefill contains 178,769 multicast packets,
2,057,534 encoded packet words, and 5,537 submissions. Destination-word
counts remain 993,811 and 3,028,167. The instruction checks find no
inter-application PDI resets and require complete lock summaries and packet
completion before status polls. Six hardware cases cover multicast, unicast,
and teardown of a dropped core with a pending lock acquire.

The decoder image SHA256 is
`4e951ee2c8bdb1085f5b7d4dee2b5128953b6262f51ee0183f525ce66d77f8e3`.
The prefill image SHA256 is
`8f1d8b577fbecf9867a3993b8249553534f0040d0a68c9970ea0bbec8147fb02`.

The stream checker analyzes control-packet starts in materialized runtime
sequences. It does not simulate arbitrary application code or infer stream
activation from full-application PDI contents. Drain assertions require the
stated completion guarantee. These measurements cover the pinned workload,
not every prompt or external kernel.

## Scoped Prefix Restoration

The November 21, 2026 prefix measurements use the workload and audited
objects below. The compiler retains the supplied overlay, mandatory delivery
checks, and allocator-owned BD chains. Scoped compiler barriers permit
independent writes to move. Matching retains each destination's ordered
actions, including repeated writes. Explicit fences retain their boundaries.

Reset or complete generated teardown can establish a quiescent setup marker.
The analysis tracks each core activation and memory affinity. Incomplete
lock-drain summaries omit the teardown-based assertion. Multicast planning
inserts completion waits for route and affinity dependencies.

| Decoder | Packet Words | Multicast Packets | Submissions | Chain Links |
| --- | ---: | ---: | ---: | ---: |
| Delivery baseline | 1,149,738 | 34,176 | 8,698 | 684 |
| Review-fixed refactor | 1,680,077 | 0 | 3,408 | 9,288 |
| Scoped prefix | 1,103,483 | 35,533 | 3,409 | 11,017 |

The scoped decoder sends 4.0% fewer encoded words than delivery and 34.3%
fewer than the review-fixed refactor. It applies 993,811 destination words,
the same count as the review-fixed refactor. Prefill contains 177,709
multicast packets and sends 2,067,523 encoded words. The baseline prefill
has no completed teardown comparison.

Six generations produce the expected 192 token IDs. Five measured
generations have a median rate of 18.87 tokens/s, versus the prior
review-fixed measurement of 18.92. Compile and load takes 82.54 seconds.
Decode compilation takes 17.01 seconds; prefill takes 56.20 seconds.
The prior review-fixed profiles were 14.22 and 41.68 seconds. These
measurements establish restored traffic reduction, not a runtime speedup.

The encoded checks require complete lock summaries, packet completion
before status polls, one entry PDI load, and no inter-application reset.
Five payload-dependent hardware cases cover serial and parallel multicast
and unicast delivery. Focused compiler tests cover explicit scopes, unknown
effects, selected-copy dependencies, affinity completion, action multiplicity,
and stream hazards. The full lit suite does not form part of this check.

The result label is `scoped-prefix-final-20261121-ctrlpkt-teardown`.
The decoder image SHA256 is
`068d87b0f402498e36a125c5628ff8a437fb0e357460af3f34586fee8337d527`.
The prefill image SHA256 is
`dca636a740f52bd1cf938132c2c05bd2488c3af901013ddb1f90752dc82f6ffe`.

## Optional Substrings

`--ctrlpkt-multicast-substrings=true` enables additional internal matches.
The pass option is `aie-dedup-ctrl-packet-multicast="substrings=true"`.
Both options default to false. The heuristic retains prefix matches, examines
at most 64 unique anchors per destination group, and selects the legal
candidate with the largest encoded-word saving in each remaining run.
The score counts payload words and both packet headers for every removed
copy. Repeated writes may occur within a candidate. Dependency checks permit
insertion at the earliest candidate start or latest candidate end.

Completion barriers surround each additional match. The heuristic restricts
serial delivery to single-column runs because a column change can establish
a required completion boundary. Explicit scoped fences remain effective.
The same effect and route checks constrain both matching modes.

| Phase and Mode | Packet Words | Multicast Packets | Submissions |
| --- | ---: | ---: | ---: |
| Decode, prefix | 1,103,483 | 35,533 | 3,409 |
| Decode, substrings | 1,101,577 | 35,809 | 4,157 |
| Prefill, prefix | 2,067,523 | 177,709 | 5,665 |
| Prefill, substrings | 2,065,121 | 178,033 | 6,829 |

Substring mode reduces decoder packet words by 0.17% and prefill packet words
by 0.12%. Destination-word counts remain unchanged. The extra completion
boundaries add 748 decoder submissions and 1,164 prefill submissions.

The fresh substring run produces the expected 192 token IDs and measures
18.39 tokens/s. Compile and load takes 97.86 seconds. Decode compilation
takes 20.97 seconds; prefill takes 67.53 seconds. Saved-image repeats reject
changed image hashes and compiler invocations. They measure 18.34 tokens/s
for substrings and 18.86 for prefix-only mode. Each repeat checks six
32-token generations and excludes one warmup. The 2.7% throughput reduction
supports prefix-only mode as the default for this workload.

The protocol checks require complete lock summaries, packet completion
before status polls, and no inter-application PDI resets. The substring
regressions cover differing heads, weighted selection, repeated actions,
both insertion points, overwrites, explicit fences, and serial column waits.

The result label is `scoped-substrings-20261121-ctrlpkt-teardown`.
The decoder image SHA256 is
`af243c37a4559bd76d92664fb806a1153b731b31fcbe56e044bb6984603142a7`.
The prefill image SHA256 is
`3a23ccf0272ed47b8a208c792a0dd7eeb8b3647a8c1a306f38d649e748fdcd78`.

## Initial Measurement

The November 20, 2026 measurements use a Strix NPU, XRT 2.26.0, and
IRON revision `5c71537f1a65884fc36cc16ef3dd312f545484ba`.
The model is Llama 3.2 1B. Each generation uses 128 prompt tokens,
a 256-token prefill chunk, 32 generated tokens, and decode loop depth two.
The frontend and kernel sources use delivery revision `ede017b31ce9`.
The backend uses Peano `4f5df13df8199bc73bd955e0ef9e966303f8f23a`.

Every case checks the prompt and every token ID against the saved NPU
direct-write reference. A separate check compares the saved token arrays
and image hashes. The reference is not an independent CPU implementation.
Each reported median excludes one warmup and includes five generations.
IRON measures the 31 tokens after the first token. Compilation and time to
first token do not contribute to that rate.

| Implementation | Configuration Protocol | Tokens/s |
| --- | --- | ---: |
| Delivery `ede017b31ce9` | Control packets with PDI resets | 9.57 |
| Refactor `bc2001dd2b2` plus lock contract and test controls | Control packets with PDI resets | 9.43 |
| Same refactor | Control packets with teardown | 18.74 |
| Main `0f767004a478` plus lock contract and call-lifetime repair | Direct writes with PDI resets | 14.87 |
| Same repaired main | Direct writes with teardown | 15.79 |

The near doubling from 9.43 to 18.74 changes the protocol on the same
refactor compiler. It does not measure a refactor-only speedup.
The reset images contain 213 decode and 215 prefill inter-application PDI
loads. Each teardown image contains one entry overlay load and no later
PDI load. Each image contains 196 static teardown sites. The instruction
checks require packet completion before core-status and lock-stall polls.

The baseline builds use the same opt-in `aie.no_lock_acquires` contract.
The external-kernel audit covers the exact objects, LLVM rounding helper,
and transitive builtin dependencies. The compiler retains explicit core
lock acquisitions. Unannotated opaque calls retain unknown lock effects.
The audit digest is
`f807c0197124538ec4a205b3a1ebfe6239f0cb4a199629a175b18d84e5a7ab83`.

Pinned main cannot route the control-packet workload: one shim port requires
five packet rules but supports four. Its direct-write path exhausts runtime
BD reservations. The separately labeled main measurements include the
11-line call-lifetime repair from `a98ecdd44af`. The baseline tests retain
both failure records. They do not count either failure as hardware coverage.

Delivery teardown exceeded the first 600-second compilation limit.
Its completed decode profile records 484.95 seconds total, including
468.67 seconds in packet-to-DMA conversion. Its prefill compilation also
exceeded a separate 1,800-second limit. The complete delivery teardown
workload has no throughput result.

## Matched Decoder Comparison

The completed delivery decoder and the pre-review refactor decoder use the
same verified refactor prefill image. The harness regenerates the frontend
MLIR for both images and requires equality with the saved input before
loading either image. Both runs use teardown. Each run checks six complete
32-token generations and measures five after warmup.

| Decoder | Tokens/s | Packet Words | Multicast Packets | Packet BDs | Submissions and Waits | Chain Links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Delivery | 16.94 | 1,149,738 | 34,176 | 9,382 | 8,698 | 684 |
| Refactor | 18.90 | 1,680,077 | 0 | 12,696 | 3,408 | 9,288 |

The refactor decoder has 11.6% higher throughput in this comparison.
Its generic BD allocator forms longer chains and removes 5,290 submissions
and waits. The refactor sends more packet words because its compiler barriers
restrict multicast selection. The nine-case ablations identify chaining and
parallel submission as the largest individual effects. The packet-word
reduction in delivery does not compensate for its additional submissions.

Both decoder images contain 196 static teardown sites and one entry PDI
load. The encoded protocol checks accept both images. The shared prefill
keeps this a decoder comparison, not a complete delivery-versus-refactor
compilation benchmark. These counts describe static instructions, not
hardware events counted during generation.

## Optimization Attribution

The nine cases consist of both endpoints and seven cases that each disable
one optimization. All cases use teardown. Each case recompiles decode and
prefill. Runtime repeats reuse the verified images and reject changed bytes
or another compiler invocation.

| Disabled Optimization | Tokens/s |
| --- | ---: |
| All seven | 11.80 |
| Multicast | 18.74 |
| Early activation | 18.67 |
| Parallel shims | 16.86 |
| BD chaining | 16.66 |
| Adjacent packet merging | 18.75 |
| Dead-write elimination | 18.75 |
| Address order | 18.74 |
| None | 18.74 |

BD chaining removes 9,288 decode and 15,087 prefill submissions and waits.
The packet payloads remain unchanged in that comparison. Parallel shims
increase the maximum outstanding packet columns from one to eight without
changing packet or submission counts. These two optimizations have the
largest individual throughput effects in this workload.

The instruction comparison observes address ordering, dead-write
elimination, merging, chaining, parallel submission, and early activation.
The refactored Llama images emit zero multicast packets even when the overlay
contains multicast routes. Their timing does not measure multicast delivery.
The individual differences overlap and must not be added.
Compiler builds ran during some measurements; small timing differences
do not establish an optimization benefit.

## Multicast Coverage

The existing one-column and two-column hardware fixtures initialize each
core's bias to zero. Control packets update every bias to one. Each core
waits for input before reading its bias. The host checks every output word.
FileCheck requires the multicast payload in the expanded IR. A separate
ELF check confirms route ID 28 and the bias payload in the encoded packets.
The unicast control contains no multicast bias packet.

Five hardware cases cover unicast, one-column multicast, separate
span-one columns, and span-two multicast with serial or parallel shims.
Each lit case checks two dispatches. An additional runner checks 101
dispatches per case, with 100 measured dispatches after warmup.
Every dispatch compares 256 output words. Median dispatch latency ranges
from 75 to 85 microseconds. This includes entry configuration, computation,
and teardown; it is not isolated packet transit time.

Focused compiler checks cover packet normalization, multicast selection,
driver options, and external lock-contract analysis. The full lit suite
does not form part of this measurement.

The local evidence directory is
`/scratch/roesti/llama-reconfig-benchmark/teardown-matrix`.
The nine-case records use run ID `llama-9-attribution-20261120`.
`repeat_saved.py` repeats immutable images. `verify_protocol.py` checks
encoded transitions. `collect_optimization_evidence.py` compares packet
and DMA schedules. `compare_branches.py` checks saved token arrays and
comparison identities. These helpers and the model weights remain external
to the repository.

## Review Fixes

The initial test checkpoint is `ae5f1aad5cf`. The subsequent review changes
retain the regionless device marker and the generic descriptor allocator.

- Activation scheduling requires explicit device context. A raw enable
	followed by disable retains that order through the packet pipeline.
- The driver always runs delivery validation. It checks packet arrival
	order, streams across delivery barriers, and concurrent multicast routes.
	The arrival-order check compares only ingress groups whose relative order
	is uncertain. The driver regression requires the cross-barrier deadlock
	diagnostic.
- The gating proof rejects calls before the gating acquire. Direct,
	indirect, LLVM, and visible lock-releasing calls have regression coverage.
	An acquire-only contract does not establish that a call cannot release a
	lock. Calls after the gate remain eligible.
- Device markers must be direct children of a single-block runtime
	sequence. Nested markers fail verification before loop unrolling.
	Nested operations can inherit an outer context. The allocation regression
	checks those reservations through unrolling.

The review-fixed compiler recompiles both Llama images and measures
18.92 tokens/s across five generations after warmup. A separate check
compares all 192 token IDs and both image hashes. The encoded checks require
complete lock summaries and no inter-application PDI resets. All five
payload-dependent multicast hardware cases also complete with matching
output after the review changes.

The review-fixed decode compiler profile records 14.22 seconds, including
0.218 seconds in packet-to-DMA conversion. Prefill records 41.68 seconds,
including 0.698 seconds in that conversion. The comparison to delivery's
completed decoder concerns the whole conversion implementation; it does
not isolate one source-level loop.

The checker at that checkpoint assumed that intervening application
operations drain their streams. The follow-up above removes that assumption
for control-packet starts. The measurements do not prove arbitrary external
kernels, alternate prompts, or every configuration protocol. The external
kernel annotations apply only to the audited artifacts in this experiment.