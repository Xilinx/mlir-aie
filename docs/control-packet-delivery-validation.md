<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Control Packet Delivery Validation

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

The checker retains its contract that intervening application operations
drain their streams. The measurements do not prove arbitrary external
kernels, alternate prompts, or every configuration protocol. The external
kernel annotations apply only to the audited artifacts in this experiment.