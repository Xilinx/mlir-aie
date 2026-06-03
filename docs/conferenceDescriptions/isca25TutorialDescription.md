# ISCA 2025 Tutorial: Leveraging the IRON AI Engine API to program the Ryzen™ AI NPU

## Introduction

The NPU of AMD Ryzen™ AI devices includes an AI Engine array comprised of a set of VLIW vector processors, Direct Memory Access channels (DMAs) and adaptable interconnect. This tutorial is targeted at performance engineers who are looking to develop designs targeting the NPU with open source design tools. We provide a close-to-metal Python API: Interface Representation for hands-ON (IRON) AIE-array programming. IRON is an open access toolkit enabling performance engineers to build fast and efficient, often specialized, designs through a set of Python language bindings around the mlir-aie dialect. Participants will first get insight into the AI Engine compute and data movement capabilities. Through small design examples expressed in the IRON API and executed on an Ryzen™ AI device, participants will leverage AI Engine features for optimizing performance of increasingly complex designs. The labs will be done on Ryzen™ AI-enabled mini-PCs, giving participants the ability to execute their own designs on real hardware.

This tutorial will cover the following key topics:
1. NPU and AI Engine architecture introduction 
1. AIE core, array configuration, and host application code compilation
1. Data movement and communication abstraction layers
1. Tracing for performance monitoring
1. Putting it all together on larger examples: matrix multiplication, convolutions as building blocks for ML and computer vision examples 

## Agenda

Date: Sunday June 22rd 2025

Start time: 8 am

Location: Tokyo, Japan  

### Prerequisites:
- Please bring your laptop so that you can SSH into our Ryzen™ AI-enabled miniPCs for the hands-on exercises.
- Knowledge in basic computer architecture, basic programming (Python), basic algorithms is required.
- Knowledge of vision pipelines and ML is not necessary but a plus.

### Contents and Timeline (tentative)

| Time | Topic | Presenter | Slides or Code |
|------|-------|-----------|----------------|
| 08:00am | Intro to spatial compute and explicit data movement | Joe | [Programming Guide](../../programming_guide/) |
| 08:15am | "Hello World" from Ryzen™ AI | Joe | [AI Engine Basic Building Blocks](../../programming_guide/section-1/) |
| 08:35am | Exercise 1: Build and run your first program | All | [Passthrough](../../programming_examples/basic/passthrough_kernel/) |
| 08:50am | Data movement on Ryzen™ AI with objectFIFOs | Joe | [Data Movement](../../programming_guide/section-2/) |
| 09:10am | Exercise 2: Explore AIE DMA capabilities | All | [DMA Transpose](../../programming_examples/basic/dma_transpose/) |
| 09:20am | Your First Program | Joe | [My First Program](../../programming_guide/section-3) |
| 09:50am | Exercise 3: Vector-scalar mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 10:00am | Coffee Break | | |
| 10:30am | Tracing and performance analysis | Gagan | [Timers](../../programming_guide/section-4/section-4a/) and [Tracing](../../programming_guide/section-4/section-4b/) |
| 10:50am | Exercise 4: Tracing vector-scalar mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:00am | Vectorizing on AIE | Joe/Phil | [Kernel Vectorization](../../programming_guide/section-4/section-4c/) |
| 11:20am | Exercise 5: Tracing vectorized vector-scalar | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:30pm | Dataflow and larger designs | Gagan | [Example Vector Designs](../../programming_guide/section-5/) and [Large Example Designs](../../programming_guide/section-6/) |
| 11:40pm | Exercise 6: More examples | All | [Programming Examples](../../programming_examples/) |
| 11:50pm | Close Tutorial | All | |

## Organizers

*Joseph Melber* is a Senior Member of Technical Staff in AMD’s Research and Advanced Development group. At AMD, he is working on hardware architectures and compiler technologies for current and future AMD devices. He received a BS in electrical engineering from the University Buffalo, as well as MS and PhD degrees from the electrical and computer engineering department at Carnegie Mellon University. His research interests include runtime systems, compiler abstractions for data movement, and hardware prototypes for future adaptive heterogeneous computing architectures.

*Gagandeep Singh* is a Member of the Technical Staff (MTS) in AMD's Research and Advanced Development group, focusing on application acceleration, design space exploration, and performance modeling. Prior to joining AMD, he was a Postdoctoral Researcher at ETH Zürich in SAFARI Research Group. He received his Ph.D. from TU Eindhoven in collaboration with IBM Research Zürich in 2021. In 2017, Gagan received a joint M.Sc. degree with distinction in Integrated Circuit Design from TUM, Germany, and NTU, Singapore. Gagan was also an R&D Software Developer at Oracle, India. He has published numerous research papers in prestigious conferences and journals, including ISCA, MICRO, IEEE Micro, Genome Biology, and Bioinformatics. He is deeply passionate about AI, healthcare and life sciences, and computer architecture.

*Phil James-Roxby* is a Senior Fellow in AMD’s Research and Advanced Development group, working on compilers and runtimes to support current and future AMD devices, particularly in the domain of AI processing. In the past, he has been responsible for a number of software enablement activities for hardware devices, including SDNet and SDAccel at Xilinx, and the original development environment for the AI Engines. He holds a PhD from the University of Manchester on hardware acceleration of embedded machine learning applications, and his main research interest continues to be how to enable users to efficiently use diverse hardware in heterogeneous systems.
