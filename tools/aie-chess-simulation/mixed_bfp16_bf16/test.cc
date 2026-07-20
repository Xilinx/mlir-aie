#include "helper.h"
#include "aie_kernel_utils.h"


void single_mac_8x8x8(bfloat16 *__restrict inA,
                      bfp16ebs8 *__restrict inB,
                      bfloat16 *__restrict outC) {
  aie::vector<bfloat16, 64> A_data_bf16 = aie::load_v<64>(inA);
  aie::accum<accfloat, 64> A_data_float;
  A_data_float = A_data_bf16;
  aie::block_vector<bfp16ebs8, 64> A_data_bfp = A_data_float.to_vector<bfp16ebs8>();

  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(inB);
  aie::block_vector<bfp16ebs8, 64> B_data = pB_stream.pop();
  aie::accum<accfloat, 64> acc_data = aie::zeros<accfloat, 64>();

  chess_report(A_data_bfp);
  chess_report(B_data);
  acc_data = mac_8x8_8x8T(A_data_bfp, B_data, acc_data);
  chess_report(acc_data);
  aie::vector<bfloat16, 64> C_data = acc_data.template to_vector<bfloat16>();
  chess_report(C_data);
  aie::store_v(outC, C_data);
}


constexpr int M = 8;  constexpr int K = 8;  constexpr int N = 8;
constexpr int m = 8;  constexpr int k = 8;  constexpr int n = 8;
constexpr int r = 8;   constexpr int s = 8;   constexpr int t = 8;



int main()
{

  printf("test start ...\n");
  int A_SIZE = M * K;
  int B_SIZE = N * K;
  int C_SIZE = M * N;
  size_t A_VOLUME = (A_SIZE * sizeof(uint8_t)) * 1.125;
  size_t B_VOLUME = (B_SIZE * sizeof(uint8_t)) * 1.125;
  size_t C_VOLUME = (C_SIZE * sizeof(uint8_t)) * 1.125;

  float* A_float = (float*)malloc(A_SIZE * sizeof(float));
  float* B_float = (float*)malloc(B_SIZE * sizeof(float));  
  for (int i = 0; i < A_SIZE; i++) {
    A_float[i] =  i % 8;
  }
  for (int i = 0; i < B_SIZE; i++) {
    B_float[i] =  i % 8  ;
  }
  
  // Test layout transpose function
  printf("Testing layout transpose...\n");
  float* B_transposed = (float*)malloc(B_SIZE * sizeof(float));
  layout_transpose_8x8block(B_float, B_transposed, N, K);

  float* Gold_float = (float*)malloc(C_SIZE * sizeof(float));
  calc_golden_result(A_float, B_float, Gold_float, M, K, N);

  print_matrix_float("output/A.txt", A_float, M, K);
  print_matrix_float("output/B.txt", B_float, N, K);
  print_matrix_float("output/B_transposed.txt", B_transposed, N, K);
  print_matrix_float("output/Gold.txt", Gold_float, M, N);

  alignas(aie::vector_decl_align) bfloat16 A_bfloat16[A_SIZE];
  for (int i = 0; i < A_SIZE; i++) {
    A_bfloat16[i] = (bfloat16)A_float[i];
  }

  std::vector<uint8_t> B_bfp16ebs8 = floatToBfp16(8, B_SIZE , B_transposed, 0);
  alignas(aie::vector_decl_align) bfloat16 C_bfloat16[64];
  single_mac_8x8x8(A_bfloat16, (bfp16ebs8*)B_bfp16ebs8.data(), C_bfloat16);

  print_matrix_bfloat16("output/C.txt", C_bfloat16, M, N);

  free(Gold_float);
  free(A_float);
  free(B_float);
  free(B_transposed);

  printf("test done!\n");
  return 0;
}
