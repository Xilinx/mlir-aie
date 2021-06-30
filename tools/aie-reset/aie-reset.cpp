// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>
#include "test_library.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define MLIR_STACK_OFFSET 4096

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

// #include "aie_inc.cpp"

}

#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

void devmemRW32(uint32_t address, uint32_t value, bool write){
    int fd;
    uint32_t *map_base;
    uint32_t read_result;
    uint32_t offset = address - 0xF70A0000;

    if((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) printf("ERROR!!!! open(devmem)\n");
    printf("\n/dev/mem opened.\n");
    fflush(stdout);

    map_base = (uint32_t *)mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xF70A0000);
    if(map_base == (void *) -1) printf("ERROR!!!! map_base\n");
    printf("Memory mapped at address %p.\n", map_base);
    fflush(stdout);

    read_result = map_base[uint32_t(offset/4)];
    printf("Value at address 0x%X: 0x%X\n", address, read_result);
    fflush(stdout);

    if(write){
        map_base[uint32_t(offset/4)] = value;
        //msync(map_base, MAP_SIZE, MS_SYNC);
        read_result = map_base[uint32_t(offset/4)];
        printf("Written 0x%X; readback 0x%X\n", value, read_result);
        fflush(stdout);
    }

    //msync(map_base, MAP_SIZE, MS_SYNC);
    if(munmap(map_base, MAP_SIZE) == -1) printf("ERROR!!!! unmap_base\n");
    printf("/dev/mem closed.\n");
    fflush(stdout);
    close(fd);
}

void devmemRW(uint32_t address, uint32_t value, bool write){
    int fd;
    void *map_base, *virt_addr;
    uint32_t read_result;
    uint64_t read_64;

    if((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) printf("ERROR!!!! open(devmem)\n");
    printf("\n/dev/mem opened.\n");
    fflush(stdout);

    map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, address & ~MAP_MASK);
    if(map_base == (void *) -1) printf("ERROR!!!! map_base\n");
    printf("Memory mapped at address %p.\n", map_base);
    fflush(stdout);

    if((address % 8) == 4){
        virt_addr = (char*)map_base + ((address-4) & MAP_MASK);
        read_64 = *((volatile unsigned long long *) virt_addr);
        read_result = read_64 >> 32;
        printf("Value at address 0x%X (%p+4): 0x%X\n", address, virt_addr, read_result);
    }else{
        virt_addr = (char*)map_base + (address & MAP_MASK);
        read_result = *((volatile unsigned long *) virt_addr);
        printf("Value at address 0x%X (%p): 0x%X\n", address, virt_addr, read_result);
    }
    fflush(stdout);

    if(write){
        if((address % 8) == 4){
            //printf("DEBUG: read64: 0x%llX\n", read_64);
            uint64_t write_64 = (read_64 & 0x00000000FFFFFFFF) | ((uint64_t)value << 32);
            //printf("DEBUG: write64: 0x%llX\n", write_64);
            *((volatile unsigned long long *) virt_addr) = write_64;
            //msync(map_base, MAP_SIZE, MS_SYNC);
            read_64 = *((volatile unsigned long long *) virt_addr);
            //printf("DEBUG: read64: 0x%llX\n", read_64);
            read_result = read_64 >> 32;
        }else{
            *((volatile unsigned long *) virt_addr) = value;
            //msync(map_base, MAP_SIZE, MS_SYNC);
            read_result = *((volatile unsigned long *) virt_addr);
        }
        printf("Written 0x%X; readback 0x%X\n", value, read_result);
        fflush(stdout);
    }
    //msync(map_base, MAP_SIZE, MS_SYNC);
    if(munmap(map_base, MAP_SIZE) == -1) printf("ERROR!!!! unmap_base\n");
    printf("/dev/mem closed.\n");
    fflush(stdout);
    close(fd);
}

// void reset_all(void){
//     printf("reset all\n");

//     //XAieGbl_NPIWrite32(0x00000004, 0x0);
//     //XAieGbl_NPIWrite32(0x00000008, 0x0);

//     // 70
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x0003F118, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x0003F144, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x0003F040, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x0003F014, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x0001F000, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x0001F004, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[6][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[5][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[4][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[3][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[2][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[1][0]), 0x00036048, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[0][0]), 0x00036048, 0x0);
//     // 71
//     XAieTile_DmWriteWord(&(TileInst[7][1]), 0x0003F128, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][1]), 0x0003F144, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][1]), 0x0003F03C, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][1]), 0x0003F024, 0x0);
//     // 72
//     XAieTile_DmWriteWord(&(TileInst[7][2]), 0x0003F11C, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][2]), 0x0003F144, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][2]), 0x0003F03C, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][2]), 0x0003F01C, 0x0);
//     // 73
//     XAieTile_DmWriteWord(&(TileInst[7][3]), 0x0003F11C, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][3]), 0x0003F10C, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][3]), 0x0003F008, 0x0);
//     XAieTile_DmWriteWord(&(TileInst[7][3]), 0x0003F01C, 0x0);

// }

// void print_70(void){
//     u32 reg_val;
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x00036008);
//     printf("70_BISR = %x\n", reg_val);

//     //reg_val = XAieLib_NPIRead32(0x00000004);
//     //printf("NPI Control = %x\n", reg_val);

//     //reg_val = XAieLib_NPIRead32(0x00000008);
//     //printf("NPI Status = %x\n", reg_val);

//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x0003F118);
//     printf("70_SSouth3_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x0003F144);
//     printf("70_SNorth2_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x0003F040);
//     printf("70_MNorth3_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x0003F014);
//     printf("70_MSouth2_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x0001F000);
//     printf("70_MUX_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][0]), 0x0001F004);
//     printf("70_DEMUX_Conf = %x\n", reg_val);
// }

// void print_71(void){
//     u32 reg_val;
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][1]), 0x0003F128);
//     printf("71_SSouth3_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][1]), 0x0003F144);
//     printf("71_SNorth0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][1]), 0x0003F03C);
//     printf("71_MNorth0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][1]), 0x0003F024);
//     printf("71_MSouth2_Conf = %x\n", reg_val);
// }

// void print_72(void){
//     u32 reg_val;
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][2]), 0x0003F11C);
//     printf("72_SSouth0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][2]), 0x0003F144);
//     printf("72_SNorth0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][2]), 0x0003F03C);
//     printf("72_MNorth0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][2]), 0x0003F01C);
//     printf("72_MSouth0_Conf = %x\n", reg_val);
// }

// void print_73(void){
//     u32 reg_val;
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][3]), 0x0003F11C);
//     printf("73_SSouth0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][3]), 0x0003F10C);
//     printf("73_SDMA1_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][3]), 0x0003F008);
//     printf("73_MDMA0_Conf = %x\n", reg_val);
//     reg_val = XAieTile_DmReadWord(&(TileInst[7][3]), 0x0003F01C);
//     printf("73_MSouth0_Conf = %x\n", reg_val);
// }

int
main(int argc, char *argv[])
{
    u32 sleep_u = 10000;

    devmemRW32(0xF70A000C, 0xF9E8D7C6, true);
    devmemRW32(0xF70A0000, 0x04000000, true);
    devmemRW32(0xF70A0004, 0x040381B1, true);
    devmemRW32(0xF70A0000, 0x04000000, true);
    devmemRW32(0xF70A0004, 0x000381B1, true);
    devmemRW32(0xF70A000C, 0x12341234, true);

    return 0;


    /*
    printf("NPI_state = %x\n", npi_ptr[0x2]);
    printf("NPI_cfg = %x\n", npi_ptr[0x1]);
    printf("NPI_lock = %x\n", npi_ptr[0x3]);

    npi_ptr[0x3] = 0xF9E8D7C6;
    npi_ptr[0x0] = 0x0D000000;
    npi_ptr[0x1] = 0x0D0381FE;
    usleep(sleep_u);
    npi_ptr[0x0] = 0x0D000000;
    npi_ptr[0x1] = 0x000381FE;
    npi_ptr[0x3] = 0x12341234;
    usleep(sleep_u);

    printf("NPI_state = %x\n", npi_ptr[0x2]);
    printf("NPI_cfg = %x\n", npi_ptr[0x1]);
    printf("NPI_lock = %x\n", npi_ptr[0x3]);
    */

    //unlock NPI registers
    //system("devmem 0xF70A000C w 0xF9E8D7C6");
    //set bit for global aie reset mask
    //system("devmem 0xF70A0000 w 0x4000000");
    //toggle reset bit for global aie reset
    //system("devmem 0xF70A0004 w 0x40381B1");

    //system("devmem 0xF70A0000 w 0x4000000");

    //system("devmem 0xF70A0004 w 0x00381B1");
    //lock NPI registers
    //system("devmem 0xF70A000C w 0x12341234");
    printf("test start.\n");

    // size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    // XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
    // XAieGbl_HwInit(&AieConfig);
    // AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    // XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    // //printf("NPI_state = %x\n", npi_ptr[0x2]);
    // //printf("NPI_cfg = %x\n", npi_ptr[0x1]);
    // //printf("NPI_lock = %x\n", npi_ptr[0x3]);

    // //reset_all();
    // printf("stream switch settings(init):\n");
    // print_70();
    // print_71();
    // print_72();
    // print_73();

    // for (int bd=0;bd<16;bd++) {
    //     // Take no prisoners.  No regerts
    //     // Overwrites the DMA_BDX_Control registers
    //     for(int ofst=0;ofst<0x14;ofst+=0x4){
    //         XAieGbl_Write32(TileInst[7][0].TileAddr + 0x0001D000+(bd*0x14)+ofst, 0x0);
    //     }
    //     for(int ofst=0;ofst<0x20;ofst+=0x4){
    //         XAieGbl_Write32(TileInst[7][3].TileAddr + 0x0001D000+(bd*0x14)+ofst, 0x0);
    //         XAieGbl_Write32(TileInst[7][2].TileAddr + 0x0001D000+(bd*0x14)+ofst, 0x0);
    //         XAieGbl_Write32(TileInst[7][1].TileAddr + 0x0001D000+(bd*0x14)+ofst, 0x0);
    //     }
    // }

    // for (int dma=0;dma<4;dma++) {
    //     for(int ofst=0;ofst<0x8;ofst+=0x4){
    //         //u32 rb = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x0001D140+(dma*0x8)+ofst);
    //         //printf("Before : dma%d_%x control is %08X\n", dma, ofst, rb);
    //         XAieGbl_Write32(TileInst[7][0].TileAddr + 0x0001D140+(dma*0x8)+ofst, 0x0);
    //         XAieGbl_Write32(TileInst[7][3].TileAddr + 0x0001DE00+(dma*0x8)+ofst, 0x0);
    //         XAieGbl_Write32(TileInst[7][2].TileAddr + 0x0001DE00+(dma*0x8)+ofst, 0x0);
    //         XAieGbl_Write32(TileInst[7][1].TileAddr + 0x0001DE00+(dma*0x8)+ofst, 0x0);
    //     }
    // }

    // mlir_configure_cores();
    // mlir_configure_switchboxes();
    // for (int l=0; l<16; l++){
    //     XAieTile_LockRelease(&(TileInst[7][0]), l, 0x0, 0);
    // }

    // mlir_initialize_locks();


    // usleep(sleep_u);
    // printf("before DMA config\n");
    // ACDC_print_tile_status(TileInst[7][3]);

    // mlir_configure_dmas();

    // usleep(sleep_u);
    // printf("after DMA config\n");
    // ACDC_print_tile_status(TileInst[7][3]);

    // printf("stream switch settings(configured):\n");
    // print_70();
    // print_71();
    // print_72();
    // print_73();

    // int errors = 0;

    // uint32_t *ddr_ptr_in, *ddr_ptr_out;
    // #define DDR_ADDR_IN  (0x4000+0x020100000000LL)
    // #define DDR_ADDR_OUT (0x6000+0x020100000000LL)
    // #define DMA_COUNT 512

    // int fd = open("/dev/mem", O_RDWR | O_SYNC);
    // if (fd != -1) {
    //     ddr_ptr_in  = (uint32_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_IN);
    //     ddr_ptr_out = (uint32_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_OUT);
    //     for (int i=0; i<DMA_COUNT; i++) {
    //         ddr_ptr_in[i] = i+1;
    //         ddr_ptr_out[i] = 0;
    //     }
    // }

    // ACDC_clear_tile_memory(TileInst[7][3]);

    // // Set iteration to 2 TODO: fix this
    // // XAieTile_DmWriteWord(&(TileInst[7][3]), 5120 , 2);

    // for (int i=0; i<DMA_COUNT/2; i++) {
    //   mlir_write_buffer_a_ping(i, 0x4);
    //   mlir_write_buffer_a_pong(i, 0x4);
    //   mlir_write_buffer_b_ping(i, 0x4);
    //   mlir_write_buffer_b_pong(i, 0x4);
    // }

    // ACDC_check("Before", mlir_read_buffer_a_ping(3), 4);
    // ACDC_check("Before", mlir_read_buffer_a_pong(3), 4);
    // ACDC_check("Before", mlir_read_buffer_b_ping(5), 4);
    // ACDC_check("Before", mlir_read_buffer_b_pong(5), 4);

    // usleep(sleep_u);
    // printf("before core start\n");
    // ACDC_print_tile_status(TileInst[7][3]);

    // printf("Start cores\n");
    // mlir_start_cores();

    // usleep(sleep_u);
    // printf("after core start\n");
    // ACDC_print_tile_status(TileInst[7][3]);
    // u32 locks70;
    // locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
    // printf("Locks70 = %08X\n", locks70);

    // printf("Release lock for accessing DDR.\n");
    // XAieTile_LockRelease(&(TileInst[7][0]), /*lockid*/ 1, /*r/w*/ 1, 0);
    // //usleep(10000);
    // XAieTile_LockRelease(&(TileInst[7][0]), /*lockid*/ 2, /*r/w*/ 1, 0);

    // usleep(sleep_u);
    // printf("after lock release\n");
    // ACDC_print_tile_status(TileInst[7][3]);
    // locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
    // printf("Locks70 = %08X\n", locks70);

    // ACDC_check("After", mlir_read_buffer_a_ping(3), 4);
    // ACDC_check("After", mlir_read_buffer_a_pong(3), 256+4);
    // ACDC_check("After", mlir_read_buffer_b_ping(5), 20);
    // ACDC_check("After", mlir_read_buffer_b_pong(5), (256+4)*5);

    // ACDC_check("DDR out",ddr_ptr_out[5],20);
    // ACDC_check("DDR out",ddr_ptr_out[256+5],(256+4)*5);

    // close(fd);

    // if (!errors) {
    //     printf("PASS!\n");
    // } else {
    //     printf("Fail!\n");
    // }
    // printf("test done.\n");
}