//===- xaie_elfloader.c -------------------------------------------*- C -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xaie_elfloader.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "elf.h"
#include "xaiengine/xaie_ecc.h"
#include "xaiengine/xaie_helper.h"
#include "xaiengine/xaie_mem.h"

#define XAIE_LOAD_ELF_TXT (1U << 0U)
#define XAIE_LOAD_ELF_BSS (1U << 1U)
#define XAIE_LOAD_ELF_DATA (1U << 2U)
#define XAIE_LOAD_ELF_ALL                                                      \
  (XAIE_LOAD_ELF_TXT | XAIE_LOAD_ELF_BSS | XAIE_LOAD_ELF_DATA)

static void _XAie_PrintElfHdr(const Elf32_Ehdr *Ehdr) {
  XAIE_DBG("**** ELF HEADER ****\n");
  XAIE_DBG("e_type\t\t: 0x%08x\n", Ehdr->e_type);
  XAIE_DBG("e_machine\t: 0x%08x\n", Ehdr->e_machine);
  XAIE_DBG("e_version\t: 0x%08x\n", Ehdr->e_version);
  XAIE_DBG("e_entry\t\t: 0x%08x\n", Ehdr->e_entry);
  XAIE_DBG("e_phoff\t\t: 0x%08x\n", Ehdr->e_phoff);
  XAIE_DBG("e_shoff\t\t: 0x%08x\n", Ehdr->e_shoff);
  XAIE_DBG("e_flags\t\t: 0x%08x\n", Ehdr->e_flags);
  XAIE_DBG("e_ehsize\t: 0x%08x\n", Ehdr->e_ehsize);
  XAIE_DBG("e_phentsize\t: 0x%08x\n", Ehdr->e_phentsize);
  XAIE_DBG("e_phnum\t\t: 0x%08x\n", Ehdr->e_phnum);
  XAIE_DBG("e_shentsize\t: 0x%08x\n", Ehdr->e_shentsize);
  XAIE_DBG("e_shnum\t\t: 0x%08x\n", Ehdr->e_shnum);
  XAIE_DBG("e_shstrndx\t: 0x%08x\n", Ehdr->e_shstrndx);

  (void)Ehdr;
}

static void _XAie_PrintProgSectHdr(const Elf32_Phdr *Phdr) {
  XAIE_DBG("**** PROGRAM HEADER ****\n");
  XAIE_DBG("p_type\t\t: 0x%08x\n", Phdr->p_type);
  XAIE_DBG("p_offset\t: 0x%08x\n", Phdr->p_offset);
  XAIE_DBG("p_vaddr\t\t: 0x%08x\n", Phdr->p_vaddr);
  XAIE_DBG("p_paddr\t\t: 0x%08x\n", Phdr->p_paddr);
  XAIE_DBG("p_filesz\t: 0x%08x\n", Phdr->p_filesz);
  XAIE_DBG("p_memsz\t\t: 0x%08x\n", Phdr->p_memsz);
  XAIE_DBG("p_flags\t\t: 0x%08x\n", Phdr->p_flags);
  XAIE_DBG("p_align\t\t: 0x%08x\n", Phdr->p_align);

  (void)Phdr;
}

static AieRC _XAie_GetTargetTileLoc(XAie_DevInst *DevInst, XAie_LocType Loc,
                                    u32 Addr, XAie_LocType *TgtLoc) {
  u8 CardDir;
  u8 RowParity;
  u8 TileType;
  const XAie_CoreMod *CoreMod;

  CoreMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_AIETILE].CoreMod;

  /*
   * Find the cardinal direction and get tile address.
   * CardDir can have values of 4, 5, 6 or 7 for valid data memory
   * addresses..
   */
  CardDir = (u8)(Addr / CoreMod->DataMemSize);

  RowParity = Loc.Row % 2U;
  /*
   * Checkerboard architecture is valid for AIE. Force RowParity to 1
   * otherwise.
   */
  if (CoreMod->IsCheckerBoard == 0U) {
    RowParity = 1U;
  }

  switch (CardDir) {
  case 4U:
    /* South */
    Loc.Row -= 1U;
    break;
  case 5U:
    /*
     * West - West I/F could be the same tile or adjacent
     * tile based on the row number
     */
    if (RowParity == 1U) {
      /* Adjacent tile */
      Loc.Col -= 1U;
    }
    break;
  case 6U:
    /* North */
    Loc.Row += 1U;
    break;
  case 7U:
    /*
     * East - East I/F could be the same tile or adjacent
     * tile based on the row number
     */
    if (RowParity == 0U) {
      /* Adjacent tile */
      Loc.Col += 1U;
    }
    break;
  default:
    /* Invalid CardDir */
    XAIE_ERROR("Invalid address - 0x%x\n", Addr);
    return XAIE_ERR;
  }

  /* Return errors if modified rows and cols are invalid */
  if (Loc.Row >= DevInst->NumRows || Loc.Col >= DevInst->NumCols) {
    XAIE_ERROR("Target row/col out of range\n");
    return XAIE_ERR;
  }

  TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
  if (TileType != XAIEGBL_TILE_TYPE_AIETILE) {
    XAIE_ERROR("Invalid tile type for address\n");
    return XAIE_ERR;
  }

  TgtLoc->Row = Loc.Row;
  TgtLoc->Col = Loc.Col;

  return XAIE_OK;
}

static AieRC _XAie_LoadDataMemSection(XAie_DevInst *DevInst, XAie_LocType Loc,
                                      const unsigned char *SectionPtr,
                                      const Elf32_Phdr *Phdr) {
  AieRC RC;
  u32 OverFlowBytes;
  u32 BytesToWrite;
  u32 SectionAddr;
  u32 SectionSize;
  u32 AddrMask;
  u64 Addr;
  XAie_LocType TgtLoc;
  const unsigned char *Buffer = SectionPtr;
  unsigned char *Tmp;
  const XAie_CoreMod *CoreMod;

  CoreMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_AIETILE].CoreMod;

  /* Check if section can access out of bound memory location on device */
  if (((Phdr->p_paddr > CoreMod->ProgMemSize) &&
       (Phdr->p_paddr < CoreMod->DataMemAddr)) ||
      ((Phdr->p_paddr + Phdr->p_memsz) >
       (CoreMod->DataMemAddr + CoreMod->DataMemSize * 4U))) {
    XAIE_ERROR("Invalid section starting at 0x%x\n", Phdr->p_paddr);
    return XAIE_INVALID_ELF;
  }

  /* Write initialized section to data memory */
  SectionSize = Phdr->p_memsz;
  SectionAddr = Phdr->p_paddr;
  AddrMask = CoreMod->DataMemSize - 1U;
  /* Check if file size is 0. If yes, allocate memory and init to 0 */
  if (Phdr->p_filesz == 0U) {
    Buffer = (const unsigned char *)calloc(Phdr->p_memsz, sizeof(char));
    if (Buffer == XAIE_NULL) {
      XAIE_ERROR("Memory allocation failed for buffer\n");
      return XAIE_ERR;
    }
    /* Copy pointer to free allocated memory in case of error. */
    Tmp = (unsigned char *)Buffer;
  }

  while (SectionSize > 0U) {
    RC = _XAie_GetTargetTileLoc(DevInst, Loc, SectionAddr, &TgtLoc);
    if (RC != XAIE_OK) {
      XAIE_ERROR("Failed to get target "
                 "location for p_paddr 0x%x\n",
                 SectionAddr);
      if (Phdr->p_filesz == 0U) {
        free(Tmp);
      }
      return RC;
    }

    /*Bytes to write in this section */
    OverFlowBytes = 0U;
    if ((SectionAddr & AddrMask) + SectionSize > CoreMod->DataMemSize) {
      OverFlowBytes =
          (SectionAddr & AddrMask) + SectionSize - CoreMod->DataMemSize;
    }

    BytesToWrite = SectionSize - OverFlowBytes;
    Addr = (u64)(SectionAddr & AddrMask);

    /* Turn ECC On if EccStatus flag is set. */
    if (DevInst->EccStatus) {
      RC = _XAie_EccOnDM(DevInst, TgtLoc);
      if (RC != XAIE_OK) {
        XAIE_ERROR("Unable to turn ECC On for Data Memory\n");
        if (Phdr->p_filesz == 0U) {
          free(Tmp);
        }
        return RC;
      }
    }

    RC = XAie_DataMemBlockWrite(DevInst, TgtLoc, (u32)Addr,
                                (const void *)Buffer, BytesToWrite);
    if (RC != XAIE_OK) {
      XAIE_ERROR("Write to data memory failed\n");
      if (Phdr->p_filesz == 0U) {
        free(Tmp);
      }
      return RC;
    }

    SectionSize -= BytesToWrite;
    SectionAddr += BytesToWrite;
    Buffer += BytesToWrite;
  }

  if (Phdr->p_filesz == 0U) {
    free(Tmp);
  }

  return XAIE_OK;
}

static AieRC _XAie_LoadProgMemSection(XAie_DevInst *DevInst, XAie_LocType Loc,
                                      const unsigned char *SectionPtr,
                                      const Elf32_Phdr *Phdr) {
  u64 Addr;
  const XAie_CoreMod *CoreMod;

  CoreMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_AIETILE].CoreMod;

  /* Write to Program Memory */
  if ((Phdr->p_paddr + Phdr->p_memsz) > CoreMod->ProgMemSize) {
    XAIE_ERROR("Overflow of program memory\n");
    return XAIE_INVALID_ELF;
  }

  Addr = (u64)(CoreMod->ProgMemHostOffset + Phdr->p_paddr) +
         _XAie_GetTileAddr(DevInst, Loc.Row, Loc.Col);

  /*
   * The program memory sections in the elf can end at 32bit
   * unaligned addresses. To factor this, we round up the number
   * of 32-bit words that has to be written to the program
   * memory. Since the elf has footers at the end, accessing
   * memory out of Progsec will not result in a segmentation
   * fault.
   */
  return XAie_BlockWrite32(DevInst, Addr, (u32 *)SectionPtr,
                           (Phdr->p_memsz + 4U - 1U) / 4U);
}

static AieRC _XAie_WriteProgramSection(XAie_DevInst *DevInst, XAie_LocType Loc,
                                       const unsigned char *ProgSec,
                                       const Elf32_Phdr *Phdr, u8 Sections) {
  const XAie_CoreMod *CoreMod;

  CoreMod = DevInst->DevProp.DevMod[XAIEGBL_TILE_TYPE_AIETILE].CoreMod;

  /* Write to Program Memory */
  if (Phdr->p_paddr < CoreMod->ProgMemSize) {
    if (Sections & XAIE_LOAD_ELF_TXT) {
      return _XAie_LoadProgMemSection(DevInst, Loc, ProgSec, Phdr);
    } else {
      XAIE_WARN("Program memory section is skipped as XAIE_LOAD_ELF_TXT flag "
                "is not set\n");
      return XAIE_OK;
    }
  }

  if ((((Sections & XAIE_LOAD_ELF_BSS) != 0U) && (Phdr->p_filesz == 0U)) ||
      (((Sections & XAIE_LOAD_ELF_DATA) != 0U) && (Phdr->p_filesz != 0U))) {
    return _XAie_LoadDataMemSection(DevInst, Loc, ProgSec, Phdr);
  } else {
    XAIE_WARN("Mismatch in program header to data memory loadable section. "
              "Skipping this program section.\n");
    return XAIE_OK;
  }
}

static AieRC _XAie_LoadElfFromMem(XAie_DevInst *DevInst, XAie_LocType Loc,
                                  const unsigned char *ElfMem, u8 Sections) {
  AieRC RC;
  const Elf32_Ehdr *Ehdr;
  const Elf32_Phdr *Phdr;
  const unsigned char *SectionPtr;

  Ehdr = (const Elf32_Ehdr *)ElfMem;
  _XAie_PrintElfHdr(Ehdr);

  /* For AIE, turn ECC Off before program memory load */
  if ((DevInst->DevProp.DevGen == XAIE_DEV_GEN_AIE) &&
      (DevInst->EccStatus == XAIE_ENABLE)) {
    _XAie_EccEvntResetPM(DevInst, Loc);
  }

  for (u32 phnum = 0U; phnum < Ehdr->e_phnum; phnum++) {
    Phdr = (Elf32_Phdr *)(ElfMem + sizeof(*Ehdr) + phnum * sizeof(*Phdr));
    _XAie_PrintProgSectHdr(Phdr);
    if (Phdr->p_type == (u32)PT_LOAD) {
      SectionPtr = ElfMem + Phdr->p_offset;
      RC = _XAie_WriteProgramSection(DevInst, Loc, SectionPtr, Phdr, Sections);
      if (RC != XAIE_OK) {
        return RC;
      }
    }
  }

  /* Turn ECC On after program memory load */
  if (DevInst->EccStatus) {
    RC = _XAie_EccOnPM(DevInst, Loc);
    if (RC != XAIE_OK) {
      XAIE_ERROR("Unable to turn ECC On for Program Memory\n");
      return RC;
    }
  }

  return XAIE_OK;
}

AieRC XAie_LoadElfMem(XAie_DevInst *DevInst, XAie_LocType Loc,
                      const unsigned char *ElfMem) {
  u8 TileType;

  if ((DevInst == XAIE_NULL) || (ElfMem == XAIE_NULL) ||
      (DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
    XAIE_ERROR("Invalid arguments\n");
    return XAIE_INVALID_ARGS;
  }

  TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
  if (TileType != XAIEGBL_TILE_TYPE_AIETILE) {
    XAIE_ERROR("Invalid tile type\n");
    return XAIE_INVALID_TILE;
  }

  return _XAie_LoadElfFromMem(DevInst, Loc, ElfMem, XAIE_LOAD_ELF_ALL);
}

#ifdef __AIESIM__
/*****************************************************************************/
/**
 *
 * This is the routine to derive the stack start and end addresses from the
 * specified map file. This function basically looks for the line
 * <b><init_address>..<final_address> ( <num> items) : Stack</b> in the
 * map file to derive the stack address range.
 *
 * @param	MapPtr: Path to the Map file.
 * @param	StackSzPtr: Pointer to the stack range structure.
 *
 * @return	XAIE_OK on success, else XAIE_ERR.
 *
 * @note		None.
 *
 *******************************************************************************/
static AieRC XAieSim_GetStackRange(const char *MapPtr,
                                   XAieSim_StackSz *StackSzPtr) {
  FILE *Fd;
  u8 buffer[200U];

  /*
   * Read map file and look for line:
   * <init_address>..<final_address> ( <num> items) : Stack
   */
  StackSzPtr->start = 0xFFFFFFFFU;
  StackSzPtr->end = 0U;

  Fd = fopen(MapPtr, "r");
  if (Fd == NULL) {
    XAIE_ERROR("Invalid Map file, %d: %s\n", errno, strerror(errno));
    return XAIE_ERR;
  }

  while (fgets(buffer, 200U, Fd) != NULL) {
    if (strstr(buffer, "items) : Stack") != NULL) {
      sscanf(buffer, "    0x%8x..0x%8x (%*s", &StackSzPtr->start,
             &StackSzPtr->end);
      break;
    }
  }

  fclose(Fd);

  if (StackSzPtr->start == 0xFFFFFFFFU) {
    return XAIE_ERR;
  } else {
    return XAIE_OK;
  }
}
#endif

AieRC XAie_LoadElfPartial(XAie_DevInst *DevInst, XAie_LocType Loc,
                          const char *ElfPtr, u8 Sections) {
  FILE *Fd;
  int Ret;
  unsigned char *ElfMem;
  u8 TileType;
  u64 ElfSz;
  AieRC RC;

  if ((DevInst == XAIE_NULL) || (DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
    XAIE_ERROR("Invalid device instance\n");
    return XAIE_INVALID_ARGS;
  }

  TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
  if (TileType != XAIEGBL_TILE_TYPE_AIETILE) {
    XAIE_ERROR("Invalid tile type\n");
    return XAIE_INVALID_TILE;
  }

  if (ElfPtr == XAIE_NULL) {
    XAIE_ERROR("Invalid ElfPtr\n");
    return XAIE_INVALID_ARGS;
  }

  Fd = fopen(ElfPtr, "r");
  if (Fd == XAIE_NULL) {
    XAIE_ERROR("Unable to open elf file, %d: %s\n", errno, strerror(errno));
    return XAIE_INVALID_ELF;
  }

  /* Get the file size of the elf */
  Ret = fseek(Fd, 0L, SEEK_END);
  if (Ret != 0) {
    XAIE_ERROR("Failed to get end of file, %d: %s\n", errno, strerror(errno));
    fclose(Fd);
    return XAIE_INVALID_ELF;
  }

  ElfSz = (u64)ftell(Fd);
  rewind(Fd);
  XAIE_DBG("Elf size is %ld bytes\n", ElfSz);

  /* Read entire elf file into memory */
  ElfMem = (unsigned char *)malloc(ElfSz);
  if (ElfMem == NULL) {
    fclose(Fd);
    XAIE_ERROR("Memory allocation failed\n");
    return XAIE_ERR;
  }

  Ret = (int)fread((void *)ElfMem, ElfSz, 1U, Fd);
  if (Ret == 0) {
    fclose(Fd);
    free(ElfMem);
    XAIE_ERROR("Failed to read Elf into memory\n");
    return XAIE_ERR;
  }

  fclose(Fd);

  RC = _XAie_LoadElfFromMem(DevInst, Loc, ElfMem, Sections);
  free(ElfMem);

  return RC;
}

AieRC XAie_LoadElf(XAie_DevInst *DevInst, XAie_LocType Loc, const char *ElfPtr,
                   u8 LoadSym) {
  u8 TileType;
  AieRC RC;

  if ((DevInst == XAIE_NULL) || (DevInst->IsReady != XAIE_COMPONENT_IS_READY)) {
    XAIE_ERROR("Invalid device instance\n");
    return XAIE_INVALID_ARGS;
  }

  TileType = DevInst->DevOps->GetTTypefromLoc(DevInst, Loc);
  if (TileType != XAIEGBL_TILE_TYPE_AIETILE) {
    XAIE_ERROR("Invalid tile type\n");
    return XAIE_INVALID_TILE;
  }

  if (ElfPtr == XAIE_NULL) {
    XAIE_ERROR("Invalid ElfPtr\n");
    return XAIE_INVALID_ARGS;
  }

#ifdef __AIESIM__
  /*
   * The code under this macro guard is used in simulation mode only.
   * According to our understanding from tools team, this is critical for
   * profiling an simulation. This code is retained as is from v1 except
   * minor changes to priting error message.
   */
  AieRC Status;
  char *MapPath;
  const char *MapPathSuffix = ".map";
  XAieSim_StackSz StackSz;

  /* Get the stack range */
  MapPath = malloc(strlen(ElfPtr) + strlen(MapPathSuffix) + 1);
  if (MapPath == NULL) {
    XAIE_ERROR("failed to malloc for .map file path.\n");
    return XAIE_ERR;
  }
  strcpy(MapPath, ElfPtr);
  strcat(MapPath, MapPathSuffix);
  Status = XAieSim_GetStackRange(MapPath, &StackSz);
  free(MapPath);
  XAIE_DBG("Stack start:%08x, end:%08x\n", StackSz.start, StackSz.end);
  if (Status != XAIE_OK) {
    XAIE_ERROR("Stack range definition failed\n");
    return Status;
  }

  /* Send the stack range set command */
  RC = XAie_CmdWrite(DevInst, DevInst->StartCol + Loc.Col, Loc.Row,
                     XAIESIM_CMDIO_CMD_SETSTACK, StackSz.start, StackSz.end,
                     XAIE_NULL);
  if (RC != XAIE_OK) {
    return RC;
  }

  /* Load symbols if enabled */
  if (LoadSym == XAIE_ENABLE) {
    RC = XAie_CmdWrite(DevInst, DevInst->StartCol + Loc.Col, Loc.Row,
                       XAIESIM_CMDIO_CMD_LOADSYM, 0, 0, ElfPtr);
    if (RC != XAIE_OK) {
      return RC;
    }
  }
#endif
  (void)LoadSym;
  (void)RC;

  return XAie_LoadElfPartial(DevInst, Loc, ElfPtr, XAIE_LOAD_ELF_ALL);
}
