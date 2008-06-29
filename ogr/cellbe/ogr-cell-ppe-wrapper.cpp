/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *ogr_cell_ppe_wrapper_cpp(void) {
return "@(#)$Id: ogr-cell-ppe-wrapper.cpp,v 1.7 2008/06/29 11:08:38 stream Exp $"; }

#ifndef CORE_NAME
#define CORE_NAME cellv1
#endif

#include <libspe2.h>
#include <cstdlib>
#include <cstring>
#include "unused.h"
#include "logstuff.h"

#define OGR_GET_DISPATCH_TABLE_FXN    spe_ogr_get_dispatch_table
#define OGROPT_HAVE_OGR_CYCLE_ASM     2

#include "ogr-cell.h"

#define SPE_WRAPPER_FUNCTION(name) SPE_WRAPPER_FUNCTION2(name)
#define SPE_WRAPPER_FUNCTION2(name) ogr_cycle_ ## name ## _spe_wrapper

extern spe_program_handle_t SPE_WRAPPER_FUNCTION(CORE_NAME);

#ifndef HAVE_MULTICRUNCH_VIA_FORK
  #error Code for fork'ed crunchers only - see static Args buffer below
#endif

static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
{
  // Check size of structures, these offsets must match assembly
  STATIC_ASSERT(sizeof(struct Level) == 80);
  STATIC_ASSERT(sizeof(struct State) == 2448);
  STATIC_ASSERT(sizeof(CellOGRCoreArgs) == 2464);
//  STATIC_ASSERT(sizeof(struct State) <= OGR_PROBLEM_SIZE);
  DNETC_UNUSED_PARAM(with_time_constraints);

  static spe_context_ptr_t context;
  static bool isInit = false;
  static void* myCellOGRCoreArgs_void; // Dummy variable to avoid compiler warnings

  unsigned int entry = SPE_DEFAULT_ENTRY;
  spe_stop_info_t stop_info;
  s32 retval;
  unsigned thread_index = 99; // todo. enough hacks.

  if (!isInit)
  {
    // Create SPE thread
    context = spe_context_create(SPE_EVENTS_ENABLE, NULL);
    if (context == NULL)
    {
      Log("Alert SPE#%d! spe_context_create() failed\n", thread_index);
      abort();
    }
    retval = spe_program_load(context, &SPE_WRAPPER_FUNCTION(CORE_NAME));
    if (retval != 0)
    {
      Log("Alert SPE#%d: spe_program_load() returned %d\n", thread_index, retval);
      abort();
    }
    if (posix_memalign(&myCellOGRCoreArgs_void, 128, sizeof(CellOGRCoreArgs)))
    {
      Log("Alert SPE#%d! posix_memalign() failed\n", thread_index);
      abort();
    }

    isInit = true;
  }

  CellOGRCoreArgs* myCellOGRCoreArgs = (CellOGRCoreArgs*)myCellOGRCoreArgs_void;

  // Copy function arguments to CellOGRCoreArgs struct
  memcpy(&myCellOGRCoreArgs->state, state, sizeof(struct State));
          myCellOGRCoreArgs->pnodes = *pnodes;

  retval = spe_context_run(context, &entry, 0, (void*)myCellOGRCoreArgs, NULL, &stop_info);
  if (retval != 0)
  {
    Log("Alert SPE#%d: spe_context_run() returned %d\n", thread_index, retval);
    abort();
  }
  retval = spe_stop_info_read(context, &stop_info);
  if (retval != 0)
  {
    Log("Alert SPE#%d: spe_stop_info_read() returned %d\n", thread_index, retval);
    abort();
  }

  __asm__ __volatile__ ("sync" : : : "memory");

  // Fetch return value of the SPE core
  if (stop_info.stop_reason == SPE_EXIT)
    retval = stop_info.result.spe_exit_code;
  else
  {
    Log("Alert: SPE#%d exit status is %d\n", thread_index, stop_info.stop_reason);
    retval = -1;
  }

  // Copy data from CellCoreArgs struct back to the function arguments
  memcpy(state, &myCellOGRCoreArgs->state, sizeof(struct State));
        *pnodes = myCellOGRCoreArgs->pnodes;

  return retval;
}
