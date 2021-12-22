// "blk" is block, it could be window or stream form in ADF graph.
module {
    func private @kfunc1(%in1 : !ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "int", ADF.blk_unit = 8})  
                             ->(!ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "int", ADF.blk_unit = 32})

    func private @kfunc2(%in1 : !ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "cint", ADF.blk_unit = 16}, 
                         %in2 : !ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "cacc", ADF.blk_unit = 80}) 
                             ->(!ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "cint", ADF.blk_unit = 16})

    func private @kfunc3(%in1 : !ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "cint", ADF.blk_unit = 16}, 
                         %in2 : !ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "cacc", ADF.blk_unit = 48}) 
                             ->(!ADF.itf<i1, i32> {ADF.data_access = "stream", ADF.blk_form = "cacc", ADF.blk_unit = 48})

    ADF.graph {
        %gi = ADF.create_ginp  [1:i1, -1:i32] -> !ADF.itf<i1, i32> // role: create const

        %2 = ADF.create_kernel @kfunc1(%gi) : (!ADF.itf<i1, i32>) 
                                            -> !ADF.itf<i1, i32>

        %3 = ADF.create_kernel @kfunc2(%gi, %2) : (!ADF.itf<i1, i32>, !ADF.itf<i1, i32>) 
                                                -> !ADF.itf<i1, i32>

        %4 = ADF.create_kernel @kfunc3(%gi, %3) : (!ADF.itf<i1, i32>, !ADF.itf<i1, i32>) 
                                                -> !ADF.itf<i1, i32>

        %go = ADF.create_gout %4 : (!ADF.itf<i1, i32>) 
                                 -> !ADF.itf<i1, i32>                                            
    } {graph_name = "simpleStream", graph_input_name = "gin", graph_output_name = "gout"} 
}


// --------------------------------------------------------------------------------
// target C++ code
// ----------------------- simpleStreamGraph.cpp ---------------------------------------

// #include <adf.h>
// #include "kernels.h"

// using namespace adf;

// class simpleGraph : public graph { 
// private:
//   kernel k1;   
//   kernel k2;   
//   kernel k3;   
// public:
//   input_port  gin;    // default graph input name
//   output_port gout;   // default graph output name
  
//   simpleGraph() {
//     k1 = kernel::create(kfunc1);
//     k2 = kernel::create(kfunc2);
//     k3 = kernel::create(kfunc3);

//     connect< window<128>> net0 (gin,       k1.in[0]); 
//     connect< window<128>> net1 (gin,       k2.in[0]); 
//     connect< window<128>> net2 (k1.out[0], k2.in[1]);
//     connect< window<128>> net3 (gin,       k3.in[0]);
//     connect< window<128>> net4 (k2.out[0], k3.in[1]);
//     connect< window<128>> net4 (k3.out[0], gout);
    
//   }
// }