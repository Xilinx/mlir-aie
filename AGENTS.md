# Commenting style (applies to ALL code I write, every session)

Comments are expensive: they draw the reader's limited attention. Spend that
attention only on what the code cannot say for itself. Default to FEWER comments.

## A comment must earn its place
Write one ONLY to capture something not evident from the code itself:
- a *why* / rationale, a constraint, an invariant, a gotcha, a non-obvious
  consequence, or genuinely complex/surprising behavior.
Do NOT write a comment that restates what the adjacent code plainly does.

## Never narrate the change or the request
Code describes what IS, never its own history. The reader has no idea (and does
not care) what the code used to be, or that an edit happened. Do NOT:
- Turn my prompt / the change request into a comment.
- Document that code was previously wrong, broken, non-idiomatic, or slow.
- Explain that something "was changed", "now does X instead of Y", or "was
  refactored to follow <abstraction>". No changelog/diff narration in code —
  that belongs in the commit message, not the source.
- Reference a "baseline", "legacy", "old", "original", "previous", or "existing"
  path/behavior — or say a change "preserves"/"keeps"/"still" does something.
  These all frame the code against a prior version the reader can't see. Describe
  the code as it stands, with no before/after framing. (If two code paths
  genuinely coexist, name them for what they ARE, not which came first.)
- Justify following the codebase's normal conventions (that's just the default).

## Specific anti-patterns to avoid
- Paraphrasing the next line(s) of code.
- Enumerating where code is NOT (e.g. "parsing lives here, not in the driver").
- Long narrative blocks that duplicate the logic directly below them.
- Restating a well-named function/variable's obvious purpose.

## Default
When in doubt, delete the comment and let the code speak. Match the file's
existing comment density, but err toward less. Keep only high-signal comments.
