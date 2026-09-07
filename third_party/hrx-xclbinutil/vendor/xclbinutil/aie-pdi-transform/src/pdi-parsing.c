/******************************************************************************
* Copyright (c) 2020-2022 Xilinx, Inc.  All rights reserved.
* Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
*
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
*
* @file pdi-parsing.c
* @addtogroup test PDI parsing
* @{
* @cond pdi-parsing
* This is the file which contains general commands.
*
* @note
* @endcond
*
******************************************************************************/

/***************************** Include Files *********************************/
#include <stdio.h>
#include <string.h>
#include "cdo_cmd.h"
#include "load_pdi.h"
#include "xpdi_compiler.h"
#include "pdi-transform.h"
#include "pdi-parsing-debug.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

/************************** Constant Definitions *****************************/

/**************************** Type Definitions *******************************/

/***************** Macros (Inline Functions) Definitions *********************/
/************************** Variable Definitions *****************************/
extern const char binary_aie_pdi_start[];
extern const char binary_aie_pdi_end[];
extern void SetDebugPdi(uint32_t* Pdi, uint32_t len, uint8_t checkDmaData);
extern uint32_t GetPdiOffset();
int SetChecksum(void *Buffer)
{
  const uint32_t Len = XIH_IHT_LEN / XIH_PRTN_WORD_LEN;
  // int Status;
  uint32_t Checksum = 0U;
  uint32_t Count = 0;
  uint32_t *BufferPtr = (uint32_t *)Buffer;

  /* Len has to be at least equal to 2 */
  if (Len < 2U) {
    return XCDO_INVALID_ARGS;
  }

  /*
   * Checksum = ~(X1 + X2 + X3 + .... + Xn)
   * Calculate the checksum
   */
  for (Count = 0U; Count < (Len - 1U); Count++) {
    /*
     * Read the word from the header
     */
    Checksum += BufferPtr[Count];
  }

  /* Invert checksum */
  Checksum ^= 0xFFFFFFFFU;

  /* Validate the checksum */
  BufferPtr[Len - 1U] = Checksum;
  return XCDO_OK;
}

int SetHeaderChecksum(void *CdoPtr)
{
  uint32_t *CdoHdr = (uint32_t *)CdoPtr;
  uint32_t CheckSum = 0U;
  uint32_t Index = 0;

  for (Index = 0U; Index < (XCDO_CDO_HDR_LEN - 1U); Index++) {
    CheckSum += CdoHdr[Index];
  }

  /* Invert checksum */
  CheckSum ^= 0xFFFFFFFFU;
  CdoHdr[Index] = CheckSum;

  return XCDO_OK;
}

void test_read_pdi(char* pdi, char** data, int* len)
{
  #define BUF_SIZE (1024*1024)
  *data = NULL;
  *len = 0;
  /* "rb": the PDI is binary — text mode would translate newlines on Windows. */
  FILE* fp = fopen(pdi, "rb");
  if (fp == NULL)
  {
    printf("%s open failed Error Number % d\n", pdi, errno);
    return;
  }
  *data = (char *)malloc((size_t)BUF_SIZE);
  *len = (int)fread(*data, 1, (size_t)BUF_SIZE, fp);
  fclose(fp);
}


// cdo_common.h
FILE* file_pointer;

XPDI_EXPORT int pdi_transform(char* pdi_file,  char* pdi_file_out, const char* out_file)
{
   if (!out_file || (out_file[0] == '\0')) 
     file_pointer = stdout;
   else 
     file_pointer = fopen(out_file, "a");
   if (file_pointer == NULL)   /* fopen failed; do not hand NULL to setvbuf */
     file_pointer = stdout;

   /* Line-buffer the log. size must be >= 2 on the MSVC CRT: it documents
      "Allowable range: 2 <= size <= INT_MAX" and otherwise invokes the invalid-parameter
      handler, which terminates the process. POSIX tolerates 0. (On Win32 _IOLBF behaves
      as _IOFBF anyway.) */
   setvbuf(file_pointer, NULL, _IOLBF, BUFSIZ);

  int Ret = 0;
  printf("Get pdi file %s, do tranform pdi check and parsing.\n", pdi_file);
  int len = 0;
  char *data = NULL;
  test_read_pdi(pdi_file, &data, &len);

  XPdiLoad PdiLoad = {0};
  if (len) {
    PdiLoad.PdiLen = len;
    PdiLoad.PdiPtr = data;
  } else {
    printf("Invalid PDI file\n");
    if (data) free(data);
    return -1;
  }
  PdiLoad.BasePtr = 0;

  XCdo_Print("\n--------------------------------------------------\n");
  XCdo_Print("Pdi parsing... file = %s; len = %u\n", pdi_file, PdiLoad.PdiLen);
  #define MAX_DEBUG_PDI_LEN (1024*500)
  const uint8_t cmpDmaData = 1;
  /* 2 x 500 KiB. As locals these overflow Windows' 1 MiB default thread stack (Linux
     defaults to 8 MiB, which is why it only ever crashed there). Heap-allocate; calloc
     also does the zeroing the memsets used to. */
  char *DebugPdi = (char *)calloc((size_t)MAX_DEBUG_PDI_LEN, 1);
  char *DebugTransformPdi = (char *)calloc((size_t)MAX_DEBUG_PDI_LEN, 1);
  if (DebugPdi == NULL || DebugTransformPdi == NULL) {
    printf("Out of memory allocating the PDI debug buffers\n");
    free(DebugPdi);
    free(DebugTransformPdi);
    free(data);
    return -1;
  }
  SetDebugPdi((uint32_t *)DebugPdi, MAX_DEBUG_PDI_LEN, cmpDmaData);
  // printf("Original ");
  XPdi_Load(&PdiLoad);
  SetDebugPdi((uint32_t *)DebugTransformPdi, MAX_DEBUG_PDI_LEN, cmpDmaData);
  XPdi_Compress_Transform(&PdiLoad, pdi_file_out);

  //Verify the data
  for (int i = 0; i < MAX_DEBUG_PDI_LEN; i++) {
    if(DebugTransformPdi[i] != DebugPdi[i]) {
      XCdo_Print("num %d value is mismatch\n", i);
      printf("Generating Original PDI log\n");
      errorLog("OriginalError.log",(uint32_t *)DebugPdi, i);
      XCdo_Print("Generating Transformed PDI log\n");
      errorLog("TransformError.log",(uint32_t *)DebugTransformPdi, i);

      assert(DebugTransformPdi[i] == DebugPdi[i]);
    }
  }

  printf("The transform PDI check pass!!! Transformed PDI is consistent with traditional PDI\n");
  free(DebugPdi);
  free(DebugTransformPdi);
  if (data) free(data);
  return Ret;
}

/**
 * @}
 * @endcond
 */

 /** @} */
