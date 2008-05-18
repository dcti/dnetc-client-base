/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_spe_wrapper_cpp(void) {
return "@(#)$Id: r72-cell-spe-wrapper.c,v 1.3 2008/05/18 15:38:31 stream Exp $"; }

#ifndef CORE_NAME
#error CORE_NAME not defined
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
  // One DMA used in program
  mfc_write_tag_mask(1<<DMA_ID);

  // Fetch arguments from main memory
  mfc_get(&myCellR72CoreArgs, argp.a32[1], sizeof(CellR72CoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  // Prepare arguments to be passed to the core
  RC5_72UnitWork* rc5_72unitwork = &myCellR72CoreArgs.rc5_72unitwork;
  u32* iterations = &myCellR72CoreArgs.iterations;

  // Call the core
  s32 retval = SPE_CORE_FUNCTION(CORE_NAME) (rc5_72unitwork, iterations, NULL);

  // Update changes in main memory
  mfc_put(&myCellR72CoreArgs, argp.a32[1], sizeof(CellR72CoreArgs), DMA_ID, 0, 0);
  mfc_read_tag_status_all();

  return retval;
}
