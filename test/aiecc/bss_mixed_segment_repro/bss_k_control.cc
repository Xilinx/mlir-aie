#include <stdint.h>
// CONTROL: .bss only, no initialised data. The linker then emits a separate
// PT_LOAD with p_filesz == 0, which aie-rt DOES handle (its calloc path), so
// this variant loads correct zeros.
volatile uint8_t zero_state[512];
extern "C" void bss_probe(uint8_t *out) {
    for (int i = 0; i < 512; i++) out[i] = zero_state[i];
}
