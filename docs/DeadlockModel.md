# The deadlock model

**Status:** specification. Nothing in the compiler reads it yet. It is what
the exact deadlock engine and its Python oracle
(`test/create-packet-flows/nightly/aiemodel/`) implement, and what their
hardware checks hold the NPU to. Entries marked **to confirm** are stated as
the model will treat them until a hardware case settles them.

## What the model decides

For a design (one `aie.device`) and the values its runtime sequence may be
dispatched with, the model decides one of:

- **accept**: no run reachable under the runtime asserts deadlocks;
- **error**: every such run deadlocks, shown by a schedule that reaches the
  deadlock;
- **accept with a guard**: whether it deadlocks depends on the runtime
  values; the guard is the condition under which it does not, checked before
  the dispatch runs.

A design outside the subset the model decides (see
[Determinism](#determinism)) is reported as undecided, naming why.

The model is untimed. It says what can happen in some order, never how long
anything takes, so a design that works only because something is fast enough
is not accepted.

## Agents

An agent runs one program, in order, and blocks only at the events listed for
it below.

| Agent | Its program |
|---|---|
| A core | the `aie.core` body: `scf` control flow, `aie.use_lock`, objectFIFO acquire and release (before lowering), stream and cascade accesses, kernel calls |
| A DMA channel run by a BD chain | the chain an `aie.dma_start`, `aie.dma` or runtime `dma_start_bd_chain` starts on that channel, with its `next_bd` loop and repeat count |
| A DMA channel run by its task queue | the tasks the host pushes to it, in push order |
| The host | the runtime sequence, in order |

Each tile's task-complete tokens reach the host through its TileControl
port; a token is an event of the channel that issues it.

## State

- **Locks.** A value in [0, 63] per lock, from its `init`.
- **Streams.** A stream connects a sending port to one or more receiving
  ports through the switchboxes the routing gives it. What a stream holds is
  bounded by the fabric's buffering on its path (see
  [Buffering](#buffering)); everything beyond that waits at the sender.
- **Task queues.** At most four tasks queued per channel. Since #3734 the
  host polls the queue before pushing, so a push to a full queue waits; it
  does not drop.
- **Host tokens.** The completion tokens issued and not yet waited for.

## Events

### Locks

| Event | Blocks until | Then |
|---|---|---|
| `AcquireGreaterEqual v` | the value is at least `v` | subtract `v` |
| `Acquire v` (equal) | the value equals `v` | **to confirm:** whether the value changes |
| `Release v` | never | add `v`; **to confirm:** what a release past 63 does |
| host `set_lock` (`WorkerRuntimeBarrier.set`) | never | set the value |

### Streams

- A send of `n` bytes blocks until the stream takes them, word by word: it
  takes what its buffering has room for and what the receivers take.
- A receive of `n` bytes blocks until `n` bytes have arrived.
- A packet carries a 4-byte header. A receiver keeps it in its data, and so
  counts it, only where `keep_pkt_header` says so.
- A broadcast stream moves a word only when every receiver can take it.
- **To confirm:** whether a receiving DMA finishes a BD early at a packet's
  last word (TLAST) when the BD's length is longer.

### DMA channels

A BD is three steps, each completing before the next:

1. its acquire, if any (a lock event above);
2. its transfer: an MM2S BD sends its length in bytes on the channel's
   stream, with its packet header if it has one; an S2MM BD receives its
   length;
3. its release, if any.

A chain runs its BDs in `next_bd` order. A repeat count of `r` runs the chain
`r + 1` times. A chain whose last BD leads back into it never ends.

A queued task runs only when every task pushed to that channel before it has
finished. A task with `issue_token` sends its completion token when its last
BD's transfer is done; **to confirm:** for an S2MM, whether that is when the
last word is written to memory.

### The host

The runtime sequence runs in order:

| Op | Blocks until |
|---|---|
| `npu.dma_memcpy_nd`, `dma_start_task`, `npu.push_queue` | the channel's queue has room |
| `npu.dma_wait`, `dma_await_task` | the task's completion token has arrived |
| `dma_free_task` | never; the model checks the task is already finished, since its BD ids may be reused once it is freed |
| `npu.rtp_write`, `npu.write32` to a buffer | never; the value is visible to the core from then on |
| `cf.assert` | never; it is a precondition on the runtime values |
| `aiex.configure` | everything before it has quiesced (see [When a run ends](#when-a-run-ends)); the new configuration starts from its initial state |

A loop in the runtime sequence runs its body as many times as its bounds say.

### Cores

A core runs its body: its loops with their bounds, its `scf.if` with
conditions the model can evaluate, its lock and stream events. An IRON
`Worker` body loops forever.

A kernel call is no event, unless the kernel reads or writes streams or
cascades. A core with no stream-port end and no cascade cannot reach a stream
from a kernel. A kernel that can needs a contract stating its stream and
cascade traffic per call.

## Buffering

The switchboxes and stream ports on a path hold a few words. Within the
subset the model decides (see [Determinism](#determinism)), larger buffering
can only remove deadlocks, never add one, so the model uses:

- the least buffering a path can have, to accept;
- the most it can have, to report a deadlock.

Where the two disagree the design is undecided. **To confirm:** the words a
switchbox hop and a DMA port hold, for each device.

## Determinism

The model decides a design by running it once, in any order. That is sound
when the result does not depend on the order: no two agents race for one
resource. The subset the model decides is designs where:

- each lock is acquired by one agent while others only release it, or the
  acquirers are themselves ordered by other events;
- every `Acquire` (equal) waits on a lock no other agent can move past its
  value while it waits;
- each receiving port is fed by one sender at a time, or by senders the
  design orders (a fan-in route whose senders take turns the runtime sequence
  or the design forces);
- token events under data-dependent control carry an annotation stating
  their effect;
- every kernel that streams has a contract.

Within the subset, the system is a Kahn process network with bounded
channels, so whether a run deadlocks does not depend on the order agents
move in.

## When a run ends

A run ends when the runtime sequence has run to its end and every wait in it
has returned. It is safe if it gets there, and it has deadlocked if it
reaches a state where no agent can move first.

At its end, a safe run must also have quiesced, so the next dispatch starts
where this one assumed it would:

- every channel's queue is empty and no BD is mid-transfer;
- no stream holds data;
- every pool the sequence filled is drained to the state it started from.

A run that ends otherwise is reported, naming what is still in flight. Each
`aiex.configure` inside a sequence is the end of one run and the start of the
next.

## Outside the model

- A DMA channel a flow names but the design does not program (a program
  written elsewhere, or loaded later): it could do anything, so the design
  is undecided.
- Out-of-order BD selection by packet id.
- Control packets that rewrite BDs or switches during a run.
- Arbiter and link holds between packets: whether two packet streams can
  deadlock by sharing an arbiter is the router's check
  (`--aie-create-pathfinder-flows`), which asks this model who waits on whom.
- **To confirm:** whether a trace stream whose buffer is full can hold an
  arbiter a data stream needs.
