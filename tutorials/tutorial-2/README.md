# <ins>Tutorial 2 - singel kernel compilation and simulation</ins>


2. After building and installing `mlir-aie`, run make to compile the first design.
    ```
    > make
    ```
    This will run the kernel compilation tool (xchesscc) on the kernel code for a single AI Engine which will be covered in more detail in tutorial 2. It then run the python script which executes the generated `mlir-aie` tools for compiling our design from the `aie.mlir` source.
    
3. Take a look at `aie.mlir` to see how we mapped externally compiled AIE kernel objects. What name is the external function set to? <img src="../images/answer1.jpg" title="extern_kernel" height=25>. Note that this name does not have to match the actual kernel name but is used in our mlir file to reference a particular defined function. The function arguments though, do have to match the external function for it to be succesfully integrated.
    > There's is no current error checking to ensure this mapping matches

4. The core is then linked to an object file where the function is defined. What is the name of the object file that the core in tile(1,4) is defined in? <img src="../images/answer1.jpg" title="kernel.o" height=25> Matching kernel object files are necessary in order for successful elf integration at a later build stage. 
