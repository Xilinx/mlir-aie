#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "ipulog.h"

extern "C" {

void kernel(uint32_t *in_buffer, uint32_t *out_buffer, uint8_t *logbuffer) {

  IPULogger log(logbuffer, 2048);
  log.write("Starting kernel execution!\n");

  uint32_t col = (get_coreid() >> 16) & 0x0000FFFF;
  uint32_t row = get_coreid() & 0x0000FFFF;

  aie::tile tile = aie::tile::current();
  uint64_t Tstart = tile.cycles();
  log.write("Core Location col=%u row=%u\n", col, row);

  memcpy(out_buffer, in_buffer, 2048);

  uint64_t Tend = tile.cycles();
  uint64_t cycles = Tend - Tstart;
  log.write("Completed executing. cycles=%u\n", cycles);
}
}
