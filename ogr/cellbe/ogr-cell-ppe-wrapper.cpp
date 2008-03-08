/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *ogr_cell_ppe_wrapper_cpp(void) {
return "@(#)$Id: ogr-cell-ppe-wrapper.cpp,v 1.4 2008/03/08 20:18:29 kakace Exp $"; }

#ifndef CORE_NAME
#error CORE_NAME undefined.
#endif


#include "ogr-cell.h"
#include <libspe2.h>
#include <cstdlib>
#include <cstring>
#include "unused.h"

#undef  OGR_GET_DISPATCH_TABLE_FXN
#define OGR_GET_DISPATCH_TABLE_FXN    spe_ogr_get_dispatch_table

#include "ansi/ogrp2_codebase.cpp"


#define SPE_WRAPPER_FUNCTION(name) SPE_WRAPPER_FUNCTION2(name)
#define SPE_WRAPPER_FUNCTION2(name) ogr_cycle_ ## name ## _spe_wrapper

extern spe_program_handle_t SPE_WRAPPER_FUNCTION(CORE_NAME);

static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  // Check size of structures, these offsets must match assembly
  STATIC_ASSERT(sizeof(struct Level) == 80);
  STATIC_ASSERT(sizeof(struct State) == 2448);
  STATIC_ASSERT(sizeof(CellOGRCoreArgs) == 2464);
  DNETC_UNUSED_PARAM(with_time_constraints);

  static spe_context_ptr_t context;
  static bool isInit = false;

  unsigned int entry = SPE_DEFAULT_ENTRY;
  spe_stop_info_t stop_info;
  s32 retval = 0;
  void* myCellOGRCoreArgs_void; // Dummy variable to avoid compiler warnings
  posix_memalign(&myCellOGRCoreArgs_void, 128, sizeof(CellOGRCoreArgs));
  CellOGRCoreArgs* myCellOGRCoreArgs = (CellOGRCoreArgs*)myCellOGRCoreArgs_void;

  if (!isInit)
  {
    // Create SPE thread
    context = spe_context_create(SPE_EVENTS_ENABLE, NULL);
    spe_program_load(context, &SPE_WRAPPER_FUNCTION(CORE_NAME));

    isInit = true;
  }

  // Copy function arguments to CellOGRCoreArgs struct
  memcpy(&myCellOGRCoreArgs->state, state, sizeof(struct State));
  memcpy(&myCellOGRCoreArgs->pnodes, pnodes, sizeof(int));

  spe_context_run(context, &entry, 0, (void*)myCellOGRCoreArgs, NULL, &stop_info);
  spe_stop_info_read(context, &stop_info);

  __asm__ __volatile__ ("sync" : : : "memory");

  // Fetch return value of the SPE core
  if (stop_info.stop_reason == SPE_EXIT)
    retval = stop_info.result.spe_exit_code;
  else
    retval = -1;

  // Copy data from CellCoreArgs struct back to the function arguments
  memcpy(state, &myCellOGRCoreArgs->state, sizeof(struct State));
  memcpy(pnodes, &myCellOGRCoreArgs->pnodes, sizeof(int));

  free(myCellOGRCoreArgs_void);

  return retval;
}
