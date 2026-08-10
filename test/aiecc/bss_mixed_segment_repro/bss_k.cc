#include <stdint.h>

// .data (PROGBITS): initialised, so it occupies file bytes
__attribute__((used, retain))
volatile uint8_t initialised[4096] = { 1 };

// .bss (NOBITS): C++ guarantees all-zero before first use
volatile uint8_t zero_state[512];

extern "C" void bss_probe(uint8_t *out) {
    for (int i = 0; i < 512; i++) out[i] = zero_state[i];
    out[0] = (uint8_t)(out[0] + (initialised[0] - initialised[0]));  // keep .data live
}
