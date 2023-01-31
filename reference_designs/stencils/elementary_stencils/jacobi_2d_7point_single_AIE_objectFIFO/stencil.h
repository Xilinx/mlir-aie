// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET     
//
//===----------------------------------------------------------------------===//


extern "C" {
void stencil_2d_7point(int32_t* restrict in1,int32_t* restrict in2,int32_t* restrict in3, int32_t* restrict out);
void stencil_2d_7point_fp32(float* restrict in1,float* restrict in2,float* restrict in3, float* restrict out);

}

