void mlir_aie_configure_cores(aie_libxaie_ctx_t* ctx) {
XAie_CoreReset(&(ctx->DevInst), XAie_TileLoc(7,1));
XAie_CoreDisable(&(ctx->DevInst), XAie_TileLoc(7,1));
for (int l=0; l<16; l++)
  XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(l, 0x0), 0);
{
AieRC RC = XAie_LoadElf(&(ctx->DevInst), XAie_TileLoc(7,1), (const char*)"core_7_1.elf",0);
if (RC != XAIE_OK)
    printf("Failed to load elf for Core[%d,%d], ret is %d\n", 7, 1, RC);
assert(RC == XAIE_OK);
}
{
u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), 0, 7);
XAie_Write32(&(ctx->DevInst), tileAddr + 0x00036048, !!1); // 1 == ResetEnable
XAie_Write32(&(ctx->DevInst), tileAddr + 0x00036048, !!0); // 0 == ResetDisable
}
} // mlir_aie_configure_cores

void mlir_aie_start_cores(aie_libxaie_ctx_t* ctx) {
XAie_CoreUnreset(&(ctx->DevInst), XAie_TileLoc(7,1));
XAie_CoreEnable(&(ctx->DevInst), XAie_TileLoc(7,1));
} // mlir_aie_start_cores

void mlir_aie_configure_dmas(aie_libxaie_ctx_t* ctx) {
XAie_DmaDesc dma_tile71_bd0;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile71_bd0), XAie_TileLoc(7,1));
XAie_DmaSetLock(&(dma_tile71_bd0), XAie_LockInit(4,1),XAie_LockInit(4,0));
XAie_DmaSetAddrLen(&(dma_tile71_bd0),  /* addrA */ 0x2000,  /* len */ 256 * 4);
XAie_DmaSetNextBd(&(dma_tile71_bd0),  /* nextbd */ 1,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile71_bd0));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile71_bd0), XAie_TileLoc(7,1),  /* bd */ 0);
XAie_DmaDesc dma_tile71_bd1;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile71_bd1), XAie_TileLoc(7,1));
XAie_DmaSetLock(&(dma_tile71_bd1), XAie_LockInit(5,1),XAie_LockInit(5,0));
XAie_DmaSetAddrLen(&(dma_tile71_bd1),  /* addrA */ 0x2400,  /* len */ 256 * 4);
XAie_DmaSetNextBd(&(dma_tile71_bd1),  /* nextbd */ 0,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile71_bd1));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile71_bd1), XAie_TileLoc(7,1),  /* bd */ 1);
XAie_DmaDesc dma_tile71_bd2;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile71_bd2), XAie_TileLoc(7,1));
XAie_DmaSetLock(&(dma_tile71_bd2), XAie_LockInit(0,0),XAie_LockInit(0,1));
XAie_DmaSetAddrLen(&(dma_tile71_bd2),  /* addrA */ 0x1000,  /* len */ 256 * 4);
XAie_DmaSetNextBd(&(dma_tile71_bd2),  /* nextbd */ 3,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile71_bd2));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile71_bd2), XAie_TileLoc(7,1),  /* bd */ 2);
XAie_DmaDesc dma_tile71_bd3;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile71_bd3), XAie_TileLoc(7,1));
XAie_DmaSetLock(&(dma_tile71_bd3), XAie_LockInit(1,0),XAie_LockInit(1,1));
XAie_DmaSetAddrLen(&(dma_tile71_bd3),  /* addrA */ 0x1400,  /* len */ 256 * 4);
XAie_DmaSetNextBd(&(dma_tile71_bd3),  /* nextbd */ 4,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile71_bd3));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile71_bd3), XAie_TileLoc(7,1),  /* bd */ 3);
XAie_DmaDesc dma_tile71_bd4;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile71_bd4), XAie_TileLoc(7,1));
XAie_DmaSetLock(&(dma_tile71_bd4), XAie_LockInit(2,0),XAie_LockInit(2,1));
XAie_DmaSetAddrLen(&(dma_tile71_bd4),  /* addrA */ 0x1800,  /* len */ 256 * 4);
XAie_DmaSetNextBd(&(dma_tile71_bd4),  /* nextbd */ 5,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile71_bd4));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile71_bd4), XAie_TileLoc(7,1),  /* bd */ 4);
XAie_DmaDesc dma_tile71_bd5;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile71_bd5), XAie_TileLoc(7,1));
XAie_DmaSetLock(&(dma_tile71_bd5), XAie_LockInit(3,0),XAie_LockInit(3,1));
XAie_DmaSetAddrLen(&(dma_tile71_bd5),  /* addrA */ 0x1C00,  /* len */ 256 * 4);
XAie_DmaSetNextBd(&(dma_tile71_bd5),  /* nextbd */ 2,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile71_bd5));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile71_bd5), XAie_TileLoc(7,1),  /* bd */ 5);
XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(7,1), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */0);
XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(7,1), /* ChNum */ 0, /* dmaDir */ DMA_MM2S);
XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(7,1), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */2);
XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(7,1), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);
} // mlir_aie_configure_dmas

static u64 _mlir_aie_external_ddr_test_buffer_in0;
static bool _mlir_aie_external_set_ddr_test_buffer_in0 = false;
void mlir_aie_external_set_addr_ddr_test_buffer_in0(u64 addr) {
    _mlir_aie_external_set_ddr_test_buffer_in0 = true;
    _mlir_aie_external_ddr_test_buffer_in0 = addr;
}
static u64 _mlir_aie_external_ddr_test_buffer_out;
static bool _mlir_aie_external_set_ddr_test_buffer_out = false;
void mlir_aie_external_set_addr_ddr_test_buffer_out(u64 addr) {
    _mlir_aie_external_set_ddr_test_buffer_out = true;
    _mlir_aie_external_ddr_test_buffer_out = addr;
}
u64 mlir_aie_external_get_addr_myBuffer_70_0(void) {
    assert(_mlir_aie_external_set_ddr_test_buffer_out);
    return _mlir_aie_external_ddr_test_buffer_out + 0;
}
u64 mlir_aie_external_get_addr_myBuffer_70_1(void) {
    assert(_mlir_aie_external_set_ddr_test_buffer_in0);
    return _mlir_aie_external_ddr_test_buffer_in0 + 0;
}
void mlir_aie_configure_shimdma_70(aie_libxaie_ctx_t* ctx) {
XAie_DmaDesc dma_tile70_bd0;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile70_bd0), XAie_TileLoc(7,0));
XAie_DmaSetLock(&(dma_tile70_bd0), XAie_LockInit(1,0),XAie_LockInit(1,1));
XAie_DmaSetAddrLen(&(dma_tile70_bd0),  /* addr */ mlir_aie_external_get_addr_myBuffer_70_0(),  /* len */ 256 * 4);
XAie_DmaSetAxi(&(dma_tile70_bd0), /* smid */ 0, /* burstlen */ 4, /* QoS */ 0 , /* Cache */ 0, /* Secure */ XAIE_ENABLE);
XAie_DmaSetNextBd(&(dma_tile70_bd0),  /* nextbd */ 0,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile70_bd0));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile70_bd0), XAie_TileLoc(7,0),  /* bd */ 0);
XAie_DmaDesc dma_tile70_bd1;
XAie_DmaDescInit(&(ctx->DevInst), &(dma_tile70_bd1), XAie_TileLoc(7,0));
XAie_DmaSetLock(&(dma_tile70_bd1), XAie_LockInit(0,1),XAie_LockInit(0,0));
XAie_DmaSetAddrLen(&(dma_tile70_bd1),  /* addr */ mlir_aie_external_get_addr_myBuffer_70_1(),  /* len */ 768 * 4);
XAie_DmaSetAxi(&(dma_tile70_bd1), /* smid */ 0, /* burstlen */ 4, /* QoS */ 0 , /* Cache */ 0, /* Secure */ XAIE_ENABLE);
XAie_DmaSetNextBd(&(dma_tile70_bd1),  /* nextbd */ 1,  /* enableNextBd */ 1);
XAie_DmaEnableBd(&(dma_tile70_bd1));
XAie_DmaWriteBd(&(ctx->DevInst), &(dma_tile70_bd1), XAie_TileLoc(7,0),  /* bd */ 1);
XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(7,0), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0);
XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(7,0), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);
XAie_DmaChannelPushBdToQueue(&(ctx->DevInst), XAie_TileLoc(7,0), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */1);
XAie_DmaChannelEnable(&(ctx->DevInst), XAie_TileLoc(7,0), /* ChNum */ 0, /* dmaDir */ DMA_MM2S);
} // mlir_aie_configure_shimdma

void mlir_aie_initialize_locks(aie_libxaie_ctx_t* ctx) {
} // mlir_aie_initialize_locks
void mlir_aie_configure_switchboxes(aie_libxaie_ctx_t* ctx) {
  int x, y;
// Core Stream Switch column 7 row 1
x = 7;
y = 1;
XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 0, DMA, 0);
XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), DMA, 0, SOUTH, 0);
// Core Stream Switch column 7 row 0
x = 7;
y = 0;
XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 3, NORTH, 0);
XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2);
// ShimMux column 7 row 0
// NOTE ShimMux always connects from the south as directions are defined relative to the tile stream switch
x = 7;
y = 0;
XAie_EnableShimDmaToAieStrmPort(&(ctx->DevInst), XAie_TileLoc(x,y), 3);
XAie_EnableAieToShimDmaStrmPort(&(ctx->DevInst), XAie_TileLoc(x,y), 2);
} // mlir_aie_configure_switchboxes

const int of_1_buff_0_offset = 4096;
float mlir_aie_read_buffer_of_1_buff_0(aie_libxaie_ctx_t* ctx, int index) {
u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_0_offset + (index*4), &value);
  union caster { int32_t i; float f; };
  caster c; c.i = value;
  return c.f;
}
void mlir_aie_write_buffer_of_1_buff_0(aie_libxaie_ctx_t* ctx, int index, float value) {
  union caster { int32_t i; float f; };
  caster c; c.f = value;
  int32_t int_value = c.i;
u32 rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_0_offset + (index*4), int_value);
}
const int of_1_buff_1_offset = 5120;
float mlir_aie_read_buffer_of_1_buff_1(aie_libxaie_ctx_t* ctx, int index) {
u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_1_offset + (index*4), &value);
  union caster { int32_t i; float f; };
  caster c; c.i = value;
  return c.f;
}
void mlir_aie_write_buffer_of_1_buff_1(aie_libxaie_ctx_t* ctx, int index, float value) {
  union caster { int32_t i; float f; };
  caster c; c.f = value;
  int32_t int_value = c.i;
u32 rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_1_offset + (index*4), int_value);
}
const int of_1_buff_2_offset = 6144;
float mlir_aie_read_buffer_of_1_buff_2(aie_libxaie_ctx_t* ctx, int index) {
u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_2_offset + (index*4), &value);
  union caster { int32_t i; float f; };
  caster c; c.i = value;
  return c.f;
}
void mlir_aie_write_buffer_of_1_buff_2(aie_libxaie_ctx_t* ctx, int index, float value) {
  union caster { int32_t i; float f; };
  caster c; c.f = value;
  int32_t int_value = c.i;
u32 rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_2_offset + (index*4), int_value);
}
const int of_1_buff_3_offset = 7168;
float mlir_aie_read_buffer_of_1_buff_3(aie_libxaie_ctx_t* ctx, int index) {
u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_3_offset + (index*4), &value);
  union caster { int32_t i; float f; };
  caster c; c.i = value;
  return c.f;
}
void mlir_aie_write_buffer_of_1_buff_3(aie_libxaie_ctx_t* ctx, int index, float value) {
  union caster { int32_t i; float f; };
  caster c; c.f = value;
  int32_t int_value = c.i;
u32 rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_1_buff_3_offset + (index*4), int_value);
}
const int of_2_buff_0_offset = 8192;
float mlir_aie_read_buffer_of_2_buff_0(aie_libxaie_ctx_t* ctx, int index) {
u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_2_buff_0_offset + (index*4), &value);
  union caster { int32_t i; float f; };
  caster c; c.i = value;
  return c.f;
}
void mlir_aie_write_buffer_of_2_buff_0(aie_libxaie_ctx_t* ctx, int index, float value) {
  union caster { int32_t i; float f; };
  caster c; c.f = value;
  int32_t int_value = c.i;
u32 rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_2_buff_0_offset + (index*4), int_value);
}
const int of_2_buff_1_offset = 9216;
float mlir_aie_read_buffer_of_2_buff_1(aie_libxaie_ctx_t* ctx, int index) {
u32 value; auto rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_2_buff_1_offset + (index*4), &value);
  union caster { int32_t i; float f; };
  caster c; c.i = value;
  return c.f;
}
void mlir_aie_write_buffer_of_2_buff_1(aie_libxaie_ctx_t* ctx, int index, float value) {
  union caster { int32_t i; float f; };
  caster c; c.f = value;
  int32_t int_value = c.i;
u32 rc =    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(7,1), of_2_buff_1_offset + (index*4), int_value);
}
int mlir_aie_acquire_lock71_14(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 14;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_lock71_14(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 14;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_0_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 0;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,0), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_0_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 0;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,0), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_1_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 0;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_1_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 0;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_1_lock_1(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 1;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_1_lock_1(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 1;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_1_lock_2(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 2;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_1_lock_2(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 2;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_1_lock_3(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 3;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_1_lock_3(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 3;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_2_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 4;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_2_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 4;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_2_lock_1(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 5;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_2_lock_1(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 5;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,1), XAie_LockInit(id,value), timeout);
}
int mlir_aie_acquire_of_3_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 1;
  return XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(7,0), XAie_LockInit(id,value), timeout);
}
int mlir_aie_release_of_3_lock_0(aie_libxaie_ctx_t* ctx, int value, int timeout) {
  const int id = 1;
  return XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(7,0), XAie_LockInit(id,value), timeout);
}
