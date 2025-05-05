# IPDPS 2025 (Double) Tutorial: Leveraging the IRON AI Engine API to program the Ryzen™ AI NPU

## Introduction

The NPU of AMD Ryzen™ AI devices includes an AI Engine array comprised of a set of VLIW vector processors, Direct Memory Access channels (DMAs) and adaptable interconnect. This tutorial is targeted at performance engineers who are looking to develop designs targeting the NPU with open source design tools. We provide a close-to-metal Python API: Interface Representation for hands-ON (IRON) AIE-array programming. IRON is an open access toolkit enabling performance engineers to build fast and efficient, often specialized, designs through a set of Python language bindings around the mlir-aie dialect. Participants will first get insight into the AI Engine compute and data movement capabilities. Through small design examples expressed in the IRON API and executed on an Ryzen™ AI device, participants will leverage AI Engine features for optimizing performance of increasingly complex designs. The labs will be done on Ryzen™ AI-enabled mini-PCs, giving participants the ability to execute their own designs on real hardware.

This tutorial will cover the following key topics:
1. NPU and AI Engine architecture introduction 
1. AIE core, array configuration, and host application code compilation
1. Data movement and communication abstraction layers
1. Tracing for performance monitoring
1. Putting it all together on larger examples: matrix multiplication, convolutions as building blocks for ML and computer vision examples 

## Agenda

Tutorial repeated on two dates.  
Dates: June 3rd and 4th, 2025 (exact times to follow)  
Location: Milano, Italy  

Prerequisites:
- Please bring your laptop so that you can SSH into our Ryzen™ AI-enabled miniPCs for the hands-on exercises.
- Knowledge in basic computer architecture, basic programming (Python), basic algorithms is required.
- Knowledge of vision pipelines and ML is not necessary but a plus.

### Contents and Timeline (tentative)

| Time | Topic | Presenter | Slides or Code |
|------|-------|-----------|----------------|
| 08:00am | Intro to spatial compute and explicit data movement | Kristof | [Programming Guide](../../programming_guide/) |
| 08:15am | "Hello World" from Ryzen™ AI | Andra | [AI Engine Basic Building Blocks](../../programming_guide/section-1/) |
| 08:35am | Exercise 1: Build and run your first program | All | [Passthrough](../../programming_examples/basic/passthrough_kernel/) |
| 08:50am | Data movement on Ryzen™ AI with objectFIFOs | Andra | [Data Movement](../../programming_guide/section-2/) |
| 09:10am | Exercise 2: Explore AIE DMA capabilities | All | [DMA Transpose](../../programming_examples/basic/dma_transpose/) |
| 09:20am | Your First Program | Kristof | [My First Program](../../programming_guide/section-3) |
| 09:50am | Exercise 3: Vector-scalar mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 10:00am | Coffee Break | | |
| 10:30am | Tracing and performance analysis | Gagan | [Timers](../../programming_guide/section-4/section-4a/) and [Tracing](../../programming_guide/section-4/section-4b/) |
| 10:50am | Exercise 4: Tracing vector-scalar mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:00am | Vectorizing on AIE | Kristof | [Kernel Vectorization](../../programming_guide/section-4/section-4c/) |
| 11:20am | Exercise 5: Tracing vectorized vector-scalar | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:30pm | Dataflow and larger designs | Gagan | [Example Vector Designs](../../programming_guide/section-5/) and [Large Example Designs](../../programming_guide/section-6/) |
| 11:40pm | Exercise 6: More examples | All | [Programming Examples](../../programming_examples/) |
| 11:50pm | Close Tutorial | All | |

## Organizers

*Kristof Denolf* is a Fellow in AMD's Research and Advanced Development group where he is working on energy-efficient computer vision and video processing applications to shape future AMD devices. He earned an M.Eng. in electronics from the Katholieke Hogeschool Brugge-Oostende (1998), now part of KULeuven, an M.Sc. in electronic system design from Leeds Beckett University (2000), and a Ph.D. from the Technical University Eindhoven (2007). He has over 25 years of combined research and industry experience at IMEC, Philips, Barco, Apple, Xilinx, and AMD. His main research interests are all aspects of the cost-efficient and dataflow-oriented design of video, vision, and graphics systems.

*Andra Bisca* is a Sr. Software Development Engineer in AMD's Research and Advanced Development group where she is working on MLIR compiler development for AMD's open-sourced compiler flow targeting AI Engines. She earned a Master's in Computer Science from École polytechnique fédérale de Lausanne in Switzerland (2022). Her main areas of industry experience and contribution are in the development of efficient data movement abstractions based in MLIR and up-leveled to Python for underlying spatial hardware architectures.

*Gagandeep Singh* is a Member of the Technical Staff (MTS) in AMD's Research and Advanced Development group, focusing on application acceleration, design space exploration, and performance modeling. Prior to joining AMD, he was a Postdoctoral Researcher at ETH Zürich in SAFARI Research Group. He received his Ph.D. from TU Eindhoven in collaboration with IBM Research Zürich in 2021. In 2017, Gagan received a joint M.Sc. degree with distinction in Integrated Circuit Design from TUM, Germany, and NTU, Singapore. Gagan was also an R&D Software Developer at Oracle, India. He has published numerous research papers in prestigious conferences and journals, including ISCA, MICRO, IEEE Micro, Genome Biology, and Bioinformatics. He is deeply passionate about AI, healthcare and life sciences, and computer architecture.
