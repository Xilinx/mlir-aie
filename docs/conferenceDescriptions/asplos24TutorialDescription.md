# ASPLOS'24 Tutorial: Levering MLIR to Design for AI Engines on Ryzen™ AI

## Introduction

The AI Engine array in the NPU of the AMD Ryzen™ AI device includes a set of VLIW vector processors with adaptable interconnect. This tutorial targets performance engineers and tool developers looking for fast and completely open-source design tools to support their research. Participants will first get insight into the AI Engine compute and data movement capabilities. Through small design examples expressed in the MLIR-AIE python language bindings and executed on a Ryzen™ AI device, participants will leverage AI Engine features to optimize the performance of increasingly complex designs. The labs will be done on Ryzen™ AI-enabled miniPCs, giving participants the ability to execute their own designs on real hardware.


This tutorial will cover the following key topics:
1. AI Engine architecture introduction 
1. AIE core, array configuration, and host application code compilation
1. Data movement and communication abstraction layers
1. Tracing for performance monitoring
1. Putting it all together on larger examples: matrix multiplication, convolutions as building blocks for ML and computer vision examples 

## Agenda

Date: Saturday, April 27th, 2024 (morning)  
Location: Hilton La Jolla Torrey Pines, San Diego, California (with ASPLOS’24)  
Prerequisite: please bring your laptop so that you can SSH into our Ryzen™ AI-enabled miniPCs for the hands-on exercises.

### Contents and Timeline (tentative)

| Time | Topic | Presenter | Slides or Code |
|------|-------|-----------|----------------|
| 08:30am | Intro to spatial compute and explicit data movement | Kristof | [Programming Guide](../../programming_guide/) |
| 08:45am | "Hello World" from Ryzen™ AI | Joe | [AI Engine Basic Building Blocks](../../programming_guide/section-1/) |
| 09:00am | Data movement on Ryzen™ AI with objectFIFOs | Joe | [Data Movement](../../programming_guide/section-2/) |
| 09:30am | Your First Program | Kristof | [My First Program](../../programming_guide/section-3) |
| 09:50am | Exercise 1: Build and run your first program | All | [Passthrough](../../programming_examples/basic/passthrough_kernel/) |
| 10:00am | Break | | |
| 10:30am | Exercise 2: Vector-Scalar Mul | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 10:40am | Tracing and performance analysis | Jack | [Timers](../../programming_guide/section-4/section-4a/) and [Tracing](../../programming_guide/section-4/section-4b/) |
| 11:10am | Exercise 3: Tracing vector-scalar | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 11:30am | Vectorizing on AIE | Jack | [Kernel Vectorization](../../programming_guide/section-4/section-4c/) |
| 11:40am | Exercise 4: Vectorized vector-scalar | All | [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) |
| 12:00pm | Dataflow and larger designs | Joe | [Example Vector Designs](../../programming_guide/section-5/) and [Large Example Designs](../../programming_guide/section-6/) |
| 12:15pm | Exercises | All | [Programming Examples](../../programming_examples/) |
| 12:30pm | Close Tutorial | All | |


## Organizers

*Jack Lo* is a Senior Member of Technical Staff in AMD’s Research and Advanced Development group. At AMD, he is focused on developing tool frameworks and optimizing applications for current and future AMD devices, particularly in the area of adaptive computing and AI processing. 

*Joseph Melber* is a Senior Member of Technical Staff in AMD’s Research and Advanced Development group. At AMD, he is working on hardware architectures and compiler technologies for current and future AMD devices. He received a BS in electrical engineering from the University Buffalo, as well as MS and PhD degrees from the electrical and computer engineering department at Carnegie Mellon University. His research interests include runtime systems, compiler abstractions for data movement, and hardware prototypes for future adaptive heterogeneous computing architectures.

*Kristof Denolf* is a Fellow in AMD's Research and Advanced Development group where he is working on energy-efficient computer vision and video processing applications to shape future AMD devices. He earned an M.Eng. in electronics from the Katholieke Hogeschool Brugge-Oostende (1998), now part of KULeuven, an M.Sc. in electronic system design from Leeds Beckett University (2000), and a Ph.D. from the Technical University Eindhoven (2007). He has over 25 years of combined research and industry experience at IMEC, Philips, Barco, Apple, Xilinx, and AMD. His main research interests are all aspects of the cost-efficient and dataflow-oriented design of video, vision, and graphics systems.

*Phil James-Roxby* is a Senior Fellow in AMD’s Research and Advanced Development group, working on compilers and runtimes to support current and future AMD devices, particularly in the domain of AI processing.  In the past, he has been responsible for a number of software enablement activities for hardware devices, including SDNet and SDAccel at Xilinx, and the original development environment for the AI Engines.  He holds a PhD from the University of Manchester on hardware acceleration of embedded machine learning applications, and his main research interest continues to be how to enable users to efficiently use diverse hardware in heterogeneous systems.

*Samuel Bayliss* is a Fellow in the Research and Advanced Development group at AMD. His academic experience includes formative study at Imperial College London, for which he earned MEng and PhD degrees in 2006 and 2012, respectively. He is energized by his current work in advancing compiler tooling using MLIR, developing programming abstractions for parallel compute and evolving hardware architectures for efficient machine learning.