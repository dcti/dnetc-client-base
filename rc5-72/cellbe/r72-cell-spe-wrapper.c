/* 
 * Copyright distributed.net 1997-2011 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_spe_wrapper_cpp(void) {
return "@(#)$Id: r72-cell-spe-wrapper.c,v 1.9 2011/03/31 05:07:33 jlawson Exp $"; }

#ifndef CORE_NAME
#define CORE_NAME cellv1
#endif

#include "ccoreio.h"
#include "r72-cell.h"
#include <spu_mfcio.h>

#define SPE_CORE_FUNCTION(name) SPE_CORE_FUNCTION2(name)
#define SPE_CORE_FUNCTION2(name) rc5_72_unit_func_ ## name ## _spe_core

#ifdef __cplusplus
extern "C"
#endif
s32 CDECL SPE_CORE_FUNCTION(CORE_NAME) ( RC5_72UnitWork *, u32 *, void * );

CellR72CoreArgs myCellR72CoreArgs __attribute__((aligned (128)));

#define DMA_ID  31

int main(unsigned long long speid, addr64 argp, addr64 envp)
{
  STATIC_ASSERT(sizeof(RC5_72UnitWork) == 44);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, rc5_72unitwork) == 16);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, iterations    ) == 16 + 44);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, sign2         ) == 16 + 48);
  STATIC_ASSERT(sizeof(CellR72CoreArgs) == 16 + 48 + 16);

  (void)speid; (void)envp;
  
  // One DMA used in program
  mfc_write_tag_mask(1<<DMA_ID);

  // Fetch arguments from main memory
  mfc_get(&myCellR72CoreArgs, argp.a32[1], sizeof(CellR72CoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  s32 retval;
  /* check for memory corruption in incoming arguments */
  if (myCellR72CoreArgs.sign1 != SIGN_PPU_TO_SPU_1)
  {
    retval = RETVAL_ERR_BAD_SIGN1;
    goto done;
  }
  if (myCellR72CoreArgs.sign2 != SIGN_PPU_TO_SPU_2)
  {
    retval = RETVAL_ERR_BAD_SIGN2;
    goto done;
  }

  // Prepare arguments to be passed to the core
  RC5_72UnitWork* rc5_72unitwork = &myCellR72CoreArgs.rc5_72unitwork;
  u32* iterations = &myCellR72CoreArgs.iterations;

  // Call the core
  retval = SPE_CORE_FUNCTION(CORE_NAME) (rc5_72unitwork, iterations, NULL);

  // Check for memory corruption after core exit
  if (myCellR72CoreArgs.sign1 != SIGN_PPU_TO_SPU_1)
    retval = RETVAL_ERR_TRASHED_SIGN1;
  else if (myCellR72CoreArgs.sign2 != SIGN_PPU_TO_SPU_2)
    retval = RETVAL_ERR_TRASHED_SIGN2;

done:
  // Update changes in main memory
  myCellR72CoreArgs.sign1 = SIGN_SPU_TO_PPU_1;
  myCellR72CoreArgs.sign2 = SIGN_SPU_TO_PPU_2;
  mfc_put(&myCellR72CoreArgs, argp.a32[1], sizeof(CellR72CoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  return retval;
}
