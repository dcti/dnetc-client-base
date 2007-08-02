/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_ppe_wrapper_cpp(void) {
return "@(#)$Id: r72-cell-ppe-wrapper.cpp,v 1.1.2.1 2007/08/02 08:08:37 decio Exp $"; }

#ifndef CORE_NAME
#error CORE_NAME defined.
#endif

#include "ccoreio.h"
#include "r72-cell.h"
#include <libspe2.h>
#include <cstdlib>
#include <cstring>

#define PPE_WRAPPER_FUNCTION(name) PPE_WRAPPER_FUNCTION2(name)
#define SPE_WRAPPER_FUNCTION(name) SPE_WRAPPER_FUNCTION2(name)

#define PPE_WRAPPER_FUNCTION2(name) rc5_72_unit_func_ ## name ## _spe
#define SPE_WRAPPER_FUNCTION2(name) rc5_72_unit_func_ ## name ## _spe_wrapper

#ifdef __cplusplus
extern "C" s32 CDECL PPE_WRAPPER_FUNCTION(CORE_NAME) ( RC5_72UnitWork *, u32 *, void * );
#endif

extern spe_program_handle_t SPE_WRAPPER_FUNCTION(CORE_NAME);

s32 CDECL PPE_WRAPPER_FUNCTION(CORE_NAME) (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  spe_context_ptr_t context;
  unsigned int entry = SPE_DEFAULT_ENTRY;
  spe_stop_info_t stop_info;
  s32 retval = 0;
  void* myCellR72CoreArgs_void; // Dummy variable to avoid compiler warnings
  posix_memalign(&myCellR72CoreArgs_void, 128, sizeof(CellR72CoreArgs));
  CellR72CoreArgs* myCellR72CoreArgs = (CellR72CoreArgs*)myCellR72CoreArgs_void;

  // Copy function arguments to CellR72CoreArgs struct
  memcpy(&myCellR72CoreArgs->rc5_72unitwork, rc5_72unitwork, sizeof(RC5_72UnitWork));
  memcpy(&myCellR72CoreArgs->iterations, iterations, sizeof(u32));

  // Create SPE thread
  context = spe_context_create(SPE_EVENTS_ENABLE, NULL);
  spe_program_load(context, &SPE_WRAPPER_FUNCTION(CORE_NAME));
  spe_context_run(context, &entry, 0, (void*)myCellR72CoreArgs, NULL, &stop_info);
  spe_stop_info_read(context, &stop_info);

  __asm__ __volatile__ ("sync" : : : "memory");

  // Fetch return value of the SPE core
  if (stop_info.stop_reason == SPE_EXIT)
    retval = stop_info.result.spe_exit_code;
  else
    retval = -1;

  // Copy data from CellR72CoreArgs struct back to the function arguments
  memcpy(rc5_72unitwork, &myCellR72CoreArgs->rc5_72unitwork, sizeof(RC5_72UnitWork));
  memcpy(iterations, &myCellR72CoreArgs->iterations, sizeof(u32));

  free(myCellR72CoreArgs_void);

  spe_context_destroy(context);

  return retval;
}
