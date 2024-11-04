# MICRO 2024 Tutorial: Leveraging the IRON AI Engine API to program the Ryzen™ AI NPU

## Introduction

The NPU of AMD Ryzen™ AI devices includes an AI Engine array comprised of a set of VLIW vector processors, data movement accelerators (DMAs) and adaptable interconnect. This tutorial is targeted at performance engineers who are looking to develop designs targeting the NPU with open source design tools. We provide a close-to-metal Python API: Interface Representation for hands-ON (IRON) AIE-array programming. IRON is an open access toolkit enabling performance engineers to build fast and efficient, often specialized, designs through a set of Python language bindings around the mlir-aie dialect. Participants will first get insight into the AI Engine compute and data movement capabilities. Through small design examples expressed in the IRON API and executed on an Ryzen™ AI device, participants will leverage AI Engine features for optimizing performance of increasingly complex designs. The labs will be done on Ryzen™ AI-enabled mini-PCs, giving participants the ability to execute their own designs on real hardware.

This tutorial will cover the following key topics:
1. AI Engine architecture introduction 
1. AIE core, array configuration, and host application code compilation
1. Data movement and communication abstraction layers
1. Tracing for performance monitoring
1. Putting it all together on larger examples: matrix multiplication, convolutions as building blocks for ML and computer vision examples 

## Agenda

Date: Sunday, November 3rd, 2024 (morning)  
Location: Austin, Texas, USA (with MICRO-57)  
Prerequisite: Please bring your laptop so that you can SSH into our Ryzen™ AI-enabled miniPCs for the hands-on exercises.

### Contents and Timeline (tentative)

| Time | Topic | Presenter | Slides or Code |
|------|-------|-----------|----------------|
| 08:00am | Intro to spatial compute and explicit data movement | Kristof | [Programming Guide](../../programming_guide/) |
| 08:15am | "Hello World" from Ryzen™ AI | Joe | [AI Engine Basic Building Blocks](../../programming_guide/section-1/) |
| 08:35am | Exercise 1: Build and run your first program | All | [Passthrough](../../programming_examples/basic/passthrough_kernel/) |
| 08:50am | Data movement on Ryzen™ AI with objectFIFOs | Joe | [Data Movement](../../programming_guide/section-2/) |
| 09:10am | Exercise 2: Explore AIE DMA capabilities | All | [DMA Transpose](../../programming_examples/basic/dma_transpose/) |
| 09:20am | Your First Program | Kristof | [My First Program](../../programming_guide/section-3) |
| 09:50am | Exercise 3: Vector-scalar mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 10:00am | Coffee Break | | |
| 10:30am | Tracing and performance analysis | Kristof | [Timers](../../programming_guide/section-4/section-4a/) and [Tracing](../../programming_guide/section-4/section-4b/) |
| 10:50am | Exercise 4: Tracing vector-scalar mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:00am | Vectorizing on AIE | Kristof | [Kernel Vectorization](../../programming_guide/section-4/section-4c/) |
| 11:20am | Exercise 5: Tracing vectorized vector-scalar | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:30pm | Dataflow and larger designs | Joe | [Example Vector Designs](../../programming_guide/section-5/) and [Large Example Designs](../../programming_guide/section-6/) |
| 11:40pm | Exercise 6: More examples | All | [Programming Examples](../../programming_examples/) |
| 11:50pm | Close Tutorial | All | |

## Organizers

*Joseph Melber* is a Senior Member of Technical Staff in AMD’s Research and Advanced Development group. At AMD, he is working on hardware architectures and compiler technologies for current and future AMD devices. He received a BS in electrical engineering from the University Buffalo, as well as MS and PhD degrees from the electrical and computer engineering department at Carnegie Mellon University. His research interests include runtime systems, compiler abstractions for data movement, and hardware prototypes for future adaptive heterogeneous computing architectures.

*Kristof Denolf* is a Fellow in AMD's Research and Advanced Development group where he is working on energy-efficient computer vision and video processing applications to shape future AMD devices. He earned an M.Eng. in electronics from the Katholieke Hogeschool Brugge-Oostende (1998), now part of KULeuven, an M.Sc. in electronic system design from Leeds Beckett University (2000), and a Ph.D. from the Technical University Eindhoven (2007). He has over 25 years of combined research and industry experience at IMEC, Philips, Barco, Apple, Xilinx, and AMD. His main research interests are all aspects of the cost-efficient and dataflow-oriented design of video, vision, and graphics systems.

*Andrew Schmidt* is a Senior Member of Technical Staff in the AMD University Program.  At AMD, he provides tutorials, training workshops and engages with universities across undergraduate and graduate curriculum, as well as research projects.  The AMD University Program offers researchers access to state-of-the-art hardware through various programs including the HPC Fund, HACC program, and donation program and offers professors and lecturers free software licenses and educational resources to support classroom teaching.  He has extensive background on adaptive computing acceleration with heterogenous architectures, previously working at the University of Southern California’s Information Sciences Institute where his focus was on reconfigurable computing, computer architecture, and hardware assurance.  He received his BS and MS in Computer Engineering from the University of Kansas and his PhD in Electrical Engineering from the University of North Carolina at Charlotte where his focus was on efficient utilization of heterogenous resources for High Performance Reconfigurable Computing.
