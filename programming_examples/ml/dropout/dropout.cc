#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include <cstdint>
#include <cstdlib>

// Dropout layer for generic input type using AIE intrinsics
// T: input/output data type (e.g., float)
// input: pointer to input array
// output: pointer to output array
// mask: pointer to mask array (0 or 1 per element)
// size: number of elements
// dropout_prob: probability to drop (e.g., 0.5 for 50% dropout)
template <typename T, int N>
void drop_out_aie(const T* input, T* output, uint8_t* mask, int size, float dropout_prob) {
    srand(42);
    float scale = aie::inv(1.0f - dropout_prob);

    int i = 0;
    for (; i <= size - N; i += N) {
        uint8_t mask_vec[N];
        for (int j = 0; j < N; ++j) {
            float r = static_cast<float>(rand()) / RAND_MAX;
            mask_vec[j] = (r > dropout_prob) ? 1 : 0;
            mask[i + j] = mask_vec[j];
        }

        aie::vector<T, N> vin = aie::load_v<N>(&input[i]);
        aie::vector<T, N> vmask = aie::broadcast<T, N>(0);
        for (int j = 0; j < N; ++j)
            vmask[j] = static_cast<T>(mask_vec[j]);

        aie::vector<T, N> vout = vin * vmask * static_cast<T>(scale);
        aie::store_v(&output[i], vout);
    }

    for (; i < size; ++i) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        mask[i] = (r > dropout_prob) ? 1 : 0;
        output[i] = input[i] * static_cast<T>(mask[i]) * static_cast<T>(scale);
    }
}

extern "C" {
void dropout_aie(const bfloat16* input, bfloat16* output, bfloat16* mask, int size, bfloat16 dropout_prob) {
    drop_out_aie<bfloat16, 16>(input, output, mask, size, dropout_prob);
}
}
