/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_spe_wrapper_cpp(void) {
return "@(#)$Id: r72-cell-spe-wrapper.cpp,v 1.1.2.1 2007/08/02 08:08:37 decio Exp $"; }

#ifndef CORE_NAME
#error CORE_NAME not defined
#endif

#include "ccoreio.h"
#include "r72-cell.h"
#include <spu_mfcio.h>
#include <cstdio>

#define SPE_CORE_FUNCTION(name) SPE_CORE_FUNCTION2(name)
#define SPE_CORE_FUNCTION2(name) rc5_72_unit_func_ ## name ## _spe_core

#ifdef __cplusplus
extern "C" s32 CDECL SPE_CORE_FUNCTION(CORE_NAME) ( RC5_72UnitWork *, u32 *, void * );
#endif

CellR72CoreArgs myCellR72CoreArgs __attribute__((aligned (128)));

int main(unsigned long long speid, addr64 argp, addr64 envp)
{
  // Fetch arguments from main memory
  mfc_get(&myCellR72CoreArgs, argp.a32[1], sizeof(CellR72CoreArgs), 31, 0, 0);
  mfc_write_tag_mask(1<<31);
  mfc_read_tag_status_all();

  // Prepare arguments to be passed to the core
  RC5_72UnitWork* rc5_72unitwork = &myCellR72CoreArgs.rc5_72unitwork;
  u32* iterations = &myCellR72CoreArgs.iterations;

  // Call the core
  s32 retval = SPE_CORE_FUNCTION(CORE_NAME) (rc5_72unitwork, iterations, NULL);

  // Update changes in main memory
  mfc_put(&myCellR72CoreArgs, argp.a32[1], sizeof(CellR72CoreArgs), 20, 0, 0);

  return retval;
}
