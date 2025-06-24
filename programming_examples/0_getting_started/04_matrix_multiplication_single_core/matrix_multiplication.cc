#include <aie_api/aie.hpp>

constexpr unsigned m = 16;
constexpr unsigned k = 16;
constexpr unsigned n = 16;
constexpr unsigned r = 8;
constexpr unsigned s = 2;
constexpr unsigned t = 8;

using MMUL = aie::mmul<r, s, t, int16, int16>;

extern "C" {

void zero(int16 *__restrict C) {
    for(unsigned row = 0; row < m; row += r) {
        for(unsigned col = 0; col < n; col += t) {
            aie::store_v(C + (row * n + col) * MMUL::size_C, aie::zeros<int16, MMUL::size_C>());
        }
    }
}

void matrix_multiplication(const int16 *__restrict A, const int16 *__restrict B, int16 *__restrict C) {
    // To understand the indexing into A, B and C in this function, please note
    // that the DMAs in the design calling this function will have already
    // transposed these matrices into r*s-, s*t- and r*t-sized subtiles.

    for(unsigned row = 0; row < m / r; row += 2) {
        for(unsigned col = 0; col < n / t; col += 2) {
            aie::vector<int16, MMUL::size_C> C00_in = aie::load_v<MMUL::size_C>(C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C);
            aie::vector<int16, MMUL::size_C> C01_in = aie::load_v<MMUL::size_C>(C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C);
            aie::vector<int16, MMUL::size_C> C10_in = aie::load_v<MMUL::size_C>(C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C);
            aie::vector<int16, MMUL::size_C> C11_in = aie::load_v<MMUL::size_C>(C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C);
            MMUL C00(C00_in);
            MMUL C01(C01_in);
            MMUL C10(C10_in);
            MMUL C11(C11_in);

            for(unsigned i = 0; i < k / s; i += 1) {
                aie::vector<int16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(A + ((row + 0) * (k / s) +         i) * MMUL::size_A);
                aie::vector<int16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(A + ((row + 1) * (k / s) +         i) * MMUL::size_A);
                aie::vector<int16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(B + (        i * (n / t) + (col + 0)) * MMUL::size_B);
                aie::vector<int16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(B + (        i * (n / t) + (col + 1)) * MMUL::size_B);
            
                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
            }
        
            aie::store_v(C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C, C00.template to_vector<int16>());
            aie::store_v(C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C, C01.template to_vector<int16>());
            aie::store_v(C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C, C10.template to_vector<int16>());
            aie::store_v(C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C, C11.template to_vector<int16>());
        }
    }
}

}