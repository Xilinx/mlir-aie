# Deliberate Changes from Base Monolithic `aiecc`

- Graph-oriented verbose output: The verbose output changed to print each graph edge getting executed ("(x/y) <edge>") vs. the previous manual print statements. This simplifies code and makes output consistent across different build steps, but does require altering the strings that the aiecc verbose output tests check against.
- Control packet DMA sequence no longer overrides the default NPU instruction sequence. Tests that require the special runtime sequence that streams the control packets, rather than the actual contents of the `aie.runtime_sequence` need to specify this in their aiecc invocation. In the previous implementation, the control packet streaming DMA sequence overrode the instruction sequence file.
- `--get/-g` selector allows selecting arbitrary edges by name.
- TODO: Repeater scripts.