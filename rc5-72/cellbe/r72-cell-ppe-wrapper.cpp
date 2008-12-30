/* 
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_ppe_wrapper_cpp(void) {
return "@(#)$Id: r72-cell-ppe-wrapper.cpp,v 1.6 2008/12/30 20:58:45 andreasb Exp $"; }

#ifndef CORE_NAME
#define CORE_NAME cellv1
#endif

#include "ccoreio.h"
#include "r72-cell.h"
#include <libspe2.h>
#include <cstdlib>
#include <cstring>
#include "logstuff.h"

#define PPE_WRAPPER_FUNCTION(name) PPE_WRAPPER_FUNCTION2(name)
#define SPE_WRAPPER_FUNCTION(name) SPE_WRAPPER_FUNCTION2(name)

#define PPE_WRAPPER_FUNCTION2(name) rc5_72_unit_func_ ## name ## _spe
#define SPE_WRAPPER_FUNCTION2(name) rc5_72_unit_func_ ## name ## _spe_wrapper

#ifdef __cplusplus
extern "C" s32 CDECL PPE_WRAPPER_FUNCTION(CORE_NAME) ( RC5_72UnitWork *, u32 *, void * );
#endif

extern spe_program_handle_t SPE_WRAPPER_FUNCTION(CORE_NAME);

#ifndef HAVE_MULTICRUNCH_VIA_FORK
  #error Code for fork'ed crunchers only - see static Args buffer below
#endif

/* Todo: move this function to platform-specific separate file */
spe_context_ptr_t ps3_assign_context_to_program(spe_program_handle_t *program);

s32 CDECL PPE_WRAPPER_FUNCTION(CORE_NAME) (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  static void* myCellR72CoreArgs_void; // Dummy variable to avoid compiler warnings

  spe_context_ptr_t context;
  unsigned int      entry = SPE_DEFAULT_ENTRY;
  spe_stop_info_t   stop_info;
  int               retval;
  unsigned          thread_index = 99; // todo. enough hacks.

  STATIC_ASSERT(sizeof(RC5_72UnitWork) == 44);
  STATIC_ASSERT(sizeof(CellR72CoreArgs) == 64);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, signature) == 48);

  /* One-time init of static exchange buffer */
  if (myCellR72CoreArgs_void == NULL)
  {
    if (posix_memalign(&myCellR72CoreArgs_void, 128, sizeof(CellR72CoreArgs)))
    {
      Log("Alert SPE#%d! posix_memalign() failed\n", thread_index);
      abort();
    }
  }

  CellR72CoreArgs* myCellR72CoreArgs = (CellR72CoreArgs*)myCellR72CoreArgs_void;

  // Copy function arguments to CellR72CoreArgs struct
  memcpy(&myCellR72CoreArgs->rc5_72unitwork, rc5_72unitwork, sizeof(RC5_72UnitWork));
          myCellR72CoreArgs->iterations = *iterations;
	  myCellR72CoreArgs->signature  = CELL_RC5_72_SIGNATURE;

  context = ps3_assign_context_to_program(&SPE_WRAPPER_FUNCTION(CORE_NAME));
  retval  = spe_context_run(context, &entry, 0, (void*)myCellR72CoreArgs, NULL, &stop_info);
  if (retval != 0)
  {
    Log("Alert SPE#%d: spe_context_run() returned %d\n", thread_index, retval);
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

  // Copy data from CellR72CoreArgs struct back to the function arguments
  memcpy(rc5_72unitwork, &myCellR72CoreArgs->rc5_72unitwork, sizeof(RC5_72UnitWork));
        *iterations = myCellR72CoreArgs->iterations;

  return retval;
}

/*-------------------------------------------------------------*/

/* Todo: move this function to platform-specific separate file */

#include <unistd.h>  /* getpid() */

#ifndef HAVE_MULTICRUNCH_VIA_FORK
  #error Code for fork'ed crunchers only - see static Args buffer below
#endif

/*
 * This function will ensure that only one SPU context exist for one cruncher/pid.
 * SPU scheduler is completely screwed, it's trying to run inactive contexts
 * (even if spe_context_run() wasn't executed for them), spending time.
 * On project change, when cruncher 'program' changed, we'll destroy old context
 * and create new one, making sure that number of context is always equal
 * to number of crunchers and SPU's.
 */
 
spe_context_ptr_t ps3_assign_context_to_program(spe_program_handle_t *program)
{
  static spe_context_ptr_t      cached_context;
  static spe_program_handle_t  *cached_program;
  static int                    cached_pid;
  
  int current_pid  = getpid();
  int thread_index = 99; /* Todo: get true cruncher index */
  int retval;
  
  if (cached_context)
  {
    if (cached_pid != current_pid)
    {
      Log("!!! FATAL !!! Cached SPE context forked from another pid (%d)\n", cached_pid);
      abort();
    }
    if (cached_program != program)
    {
      // Log("Replacing SPE context because SPE program changed\n");
      if (spe_context_destroy(cached_context))
        Log("Alert SPE%d! spe_context_destroy() failed, errno=%d\n", thread_index, errno);
      cached_context = NULL;
    }
  }
  
  if (cached_context == NULL)
  {
    cached_context = spe_context_create(0, NULL);
    if (cached_context == NULL)
    {
      Log("Alert SPE#%d! spe_context_create() failed\n", thread_index);
      abort();
    }
    retval = spe_program_load(cached_context, program);
    if (retval != 0)
    {
      Log("Alert SPE#%d: spe_program_load() returned %d\n", thread_index, retval);
      abort();
    }
    cached_program = program;
    cached_pid     = current_pid;
  }
  
  return cached_context;
}
