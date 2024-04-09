# ASPLOS'24 Tutorial: Levering MLIR to Design for AI Engines on Ryzen AI

## Introduction

The AI Engine array in the NPU of the AMD Ryzen AI device includes a set of VLIW vector processors with adaptable interconnect. This tutorial is targeted at performance engineers and tool developers who are looking for fast and completely open source design tools to support their research. Participants will first get insight into the AI Engine compute and data movement capabilities. Through small design examples expressed in the MLIR-AIE python language bindings and executed on an Ryzen AI device, participants will leverage AI Engine features for optimizing performance of increasingly complex designs. The labs will be done on Ryzen AI enabled miniPCs giving participants the ability to execute their own designs on real hardware.


This tutorial will cover the following key topics:
1. AI Engine architecture introduction 
1. AIE core, array configuration and host application code compilation
1. Data movement and communication abstraction layers
1. Tracing for performance monitoring
1. Putting it all together on larger examples: matrix multiplication, convolutions as building blocks for ML and computer vision examples 

## Agenda

Date: Saturday April 27th 2024 (morning)  
Location: Hilton La Jolla Torrey Pines, San Diego, California (with ASPLOS’24)  
Prerequisite: please bring your laptop, so that you can ssh into our Ryzen AI enabled miniPCs for the hands-on excersizes.

### Contents and Timeline (tentative)

| Time | Topic | Presenter | Slides or Code |
|------|-------|-----------|----------------|
| 08:30am | Intro to spatial compute and explicit data movement | Kristof | tbd |
| 08:45am | "Hello World" from Ryzen AI | Jack | tbd |
| 09:00am | Data movement on Ryzen AI with objectFIFOs | Joe | tbd |
| 09:30am | Exersise 1: Build and run your first program | All | tbd |
| 09:45am | Exersise 2: Vector-scalar | All |tbd |
| 10:00am | Break | | |
| 11:00am | Tracing and performance analysis | Jack | tbd |
| 11:10am | Exercise 3: Tracing vector-scalar | All | tbd |
| 11:30am | Vectorizing on AIE | Kristof | tbd |
| 11:40am | Exercise 4: Vectorized vector-scalar | All | tbd |
| 12:00pm | Dataflow and larger designs | Joe | tbd |
| 12:15pm | Exercises | All | |
| 12:30pm | Close Tutorial | All | |


## Organizers

*Jack Lo* is a Senior Member of Technical Staff in AMD’s Research and Advanced Development group. At AMD, he is focused on developing tool frameworks and optimizing applications for current and future AMD devices, particularly in the area of adaptive computing and AI processing. 

*Joseph Melber* is a Senior Member of Technical Staff in AMD’s Research and Advanced Development group. At AMD, he is working on hardware architectures and compiler technologies for current and future AMD devices. He received a BS in electrical engineering from the University Buffalo, as well as MS and PhD degrees from the electrical and computer engineering department at Carnegie Mellon University. His research interests include runtime systems, compiler abstractions for data movement, and hardware prototypes for future adaptive heterogeneous computing architectures.

*Kristof Denolf* is a Fellow at AMD's Research and Advanced Development group where he is working on energy efficient computer vision and video processing applications to shape future AMD devices. He earned a M.Eng. in electronics from the Katholieke Hogeschool Brugge-Oostende (1998), now part of KULeuven, a M.Sc. in electronic system design from Leeds Beckett University (2000) and a Ph.D. from the Technical University Eindhoven (2007). He has over 25 years of combined research and industry experience at IMEC, Philips, Barco, Apple, Xilinx and AMD. His main research interest are all aspects of the cost-efficient and dataflow oriented design of video, vision and graphics systems.
