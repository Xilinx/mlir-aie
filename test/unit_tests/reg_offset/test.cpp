#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  //mlir_aie_configure_cores(_xaie);

  u64 tileAddr = _XAie_GetTileAddr(&(_xaie->DevInst), 0, 7);
  XAie_Write32(&(_xaie->DevInst), tileAddr + 0x00036048, !!1); // 1 == ResetEnable
  XAie_Write32(&(_xaie->DevInst), tileAddr + 0x00036048, !!0); // 0 == ResetDisable
  printf("test end.\n");
  return 0;
}
