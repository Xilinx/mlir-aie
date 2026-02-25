#include <aie_api/aie.hpp>
#include <stdint.h>

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  event0();
  // Find maximum for numerical stability
  float max_val = (float)input_vector[0];
  for (uint32_t i = 1; i < vector_size; i++) {
    float val = (float)input_vector[i];
    if (val > max_val) {
      max_val = val;
    }
  }

  // First pass: compute exp(x) and sum
  float sum = 0.0f;
  for (uint32_t i = 0; i < vector_size; i++) {
    float x = (float)input_vector[i] - max_val;

    // Improved exp approximation
    // log2(e) = 1.442695040888963
    int32_t ix = (int32_t)(x * 1.442695040888963f);
    float fx = x * 1.442695040888963f - ix;

    // Compute 2^ix using bit manipulation
    ix = (ix + 127) << 23;
    float pow2_ix;
    memcpy(&pow2_ix, &ix, sizeof(float));

    // Improved approximation for 2^fx with correction term
    // ln(2) = 0.6931471805599453
    float pow2_fx =
        1.0f + 0.6931471805599453f * fx + 0.2401598148889220f * fx * fx;

    float result = pow2_ix * pow2_fx;
    output_vector[i] = (bfloat16)result;
    sum += result;
  }

  // Second pass: normalize with improved precision
  const float eps = 1e-7f; // Small epsilon to prevent division by zero
  sum = sum + eps;
  float inv_sum = 1.0f / sum;

  for (uint32_t i = 0; i < vector_size; i++) {
    float val = (float)output_vector[i] * inv_sum;
    output_vector[i] = (bfloat16)val;
  }
  event1();
  return;
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t input_size) {
  softmax_simple_bf16(input, output, input_size);
}

} // extern "C"