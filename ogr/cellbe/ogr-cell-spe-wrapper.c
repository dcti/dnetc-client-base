/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_spe_wrapper_cpp(void) {
return "@(#)$Id: ogr-cell-spe-wrapper.c,v 1.4 2008/03/08 20:18:29 kakace Exp $"; }

#ifndef CORE_NAME
#error CORE_NAME not defined
#endif

#include <spu_intrinsics.h>
#include "ccoreio.h"
#include "cputypes.h"
#include "ogr-cell.h"
#include <spu_mfcio.h>

#include "ansi/ogr_dat.cpp"

#define SPE_CORE_FUNCTION(name) SPE_CORE_FUNCTION2(name)
#define SPE_CORE_FUNCTION2(name) ogr_cycle_ ## name ## _spe_core

#ifdef __cplusplus
extern "C"
#endif
s32 CDECL SPE_CORE_FUNCTION(CORE_NAME) ( struct State*, int*, const unsigned char* );

CellOGRCoreArgs myCellOGRCoreArgs __attribute__((aligned (128)));

int main(unsigned long long speid, addr64 argp, addr64 envp)
{
  // Check size of structures, these offsets must match assembly
  STATIC_ASSERT(sizeof(struct Level) == 80);
  STATIC_ASSERT(sizeof(CellOGRCoreArgs) == 2464);
  STATIC_ASSERT(offsetof(CellOGRCoreArgs, state.Levels) == 32);

  // Fetch arguments from main memory
  mfc_get(&myCellOGRCoreArgs, argp.a32[1], sizeof(CellOGRCoreArgs), 31, 0, 0);
  mfc_write_tag_mask(1<<31);
  mfc_read_tag_status_all();

  // Prepare arguments to be passed to the core
  struct State* state = &myCellOGRCoreArgs.state;
  int* pnodes = &myCellOGRCoreArgs.pnodes;

  // Call the core
  s32 retval = SPE_CORE_FUNCTION(CORE_NAME) (state, pnodes, ogr_choose_dat);

  // Update changes in main memory
  mfc_put(&myCellOGRCoreArgs, argp.a32[1], sizeof(CellOGRCoreArgs), 20, 0, 0);

  return retval;
}
