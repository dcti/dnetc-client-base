/* 
 * Copyright distributed.net 1997-2011 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *r72_cell_ppe_wrapper_cpp(void) {
return "@(#)$Id: r72-cell-ppe-wrapper.cpp,v 1.9 2011/03/31 05:07:33 jlawson Exp $"; }

#ifndef CORE_NAME
#define CORE_NAME cellv1
#endif

#include "ccoreio.h"
#include "r72-cell.h"
#include <libspe2.h>
#include <stddef.h>  // offsetof
#include <stdlib.h>  // abort
#include <string.h>
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
  #error Code for forked crunchers only - see static Args buffer below
#endif

/* Todo: move this function to platform-specific separate file */
spe_context_ptr_t ps3_assign_context_to_program(spe_program_handle_t *program);
void              ps3_kernel_bug(void);

s32 CDECL PPE_WRAPPER_FUNCTION(CORE_NAME) (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void * /*memblk*/)
{
  static void* myCellR72CoreArgs_void; // Dummy variable to avoid compiler warnings

  spe_context_ptr_t context;
  unsigned int      entry = SPE_DEFAULT_ENTRY;
  spe_stop_info_t   stop_info;
  int               retval;
  unsigned          thread_index = 99; // todo. enough hacks.

  STATIC_ASSERT(sizeof(RC5_72UnitWork) == 44);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, rc5_72unitwork) == 16);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, iterations    ) == 16 + 44);
  STATIC_ASSERT(offsetof(CellR72CoreArgs, sign2         ) == 16 + 48);
  STATIC_ASSERT(sizeof(CellR72CoreArgs) == 16 + 48 + 16);

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
          myCellR72CoreArgs->sign1      = SIGN_PPU_TO_SPU_1;
  memcpy(&myCellR72CoreArgs->rc5_72unitwork, rc5_72unitwork, sizeof(RC5_72UnitWork));
          myCellR72CoreArgs->iterations = *iterations;
          myCellR72CoreArgs->sign2      = SIGN_PPU_TO_SPU_2;

  context = ps3_assign_context_to_program(&SPE_WRAPPER_FUNCTION(CORE_NAME));
  retval  = spe_context_run(context, &entry, 0, (void*)myCellR72CoreArgs, NULL, &stop_info);
  if (retval != 0)
  {
    Log("Alert SPE#%d: spe_context_run() returned %d\n", thread_index, retval);
    abort();
  }
  if (myCellR72CoreArgs->sign1 != SIGN_SPU_TO_PPU_1)
  {
    Log("R72-SPE#%d: core returned bad head signature! (expected: 0x%08X, returned: 0x%08X)\n",
        thread_index, SIGN_SPU_TO_PPU_1, myCellR72CoreArgs->sign1);
    ps3_kernel_bug();
  }
  if (myCellR72CoreArgs->sign2 != SIGN_SPU_TO_PPU_2)
  {
    Log("R72-SPE#%d: core returned bad tail signature! (expected: 0x%08X, returned: 0x%08X)\n",
        thread_index, SIGN_SPU_TO_PPU_2, myCellR72CoreArgs->sign2);
    ps3_kernel_bug();
  }

  // Check SPU thread exit status (must be normal exit)
  if (stop_info.stop_reason != SPE_EXIT)
  {
    Log("R72-SPE#%d: abnormal SPU thread exit status (%d)\n", thread_index, stop_info.stop_reason);
    abort();
  }

  // Fetch and validate return value of the SPE core
  retval = stop_info.result.spe_exit_code;
  if (retval != RESULT_WORKING && retval != RESULT_NOTHING && retval != RESULT_FOUND)
  {
    Log("R72-SPE%d: abnormal exit code (%d) from SPU thread!\n", thread_index, retval);

    /* magic numbers meaning internal core errors */
    const char *msg;
    switch (retval)
    {
      case RETVAL_ERR_BAD_SIGN1:     msg = "passed bad head signature"; break;
      case RETVAL_ERR_BAD_SIGN2:     msg = "passed bad tail signature"; break;
      case RETVAL_ERR_TRASHED_SIGN1: msg = "head signature corrupted during processing"; break;
      case RETVAL_ERR_TRASHED_SIGN2: msg = "tail signature corrupted during processing"; break;
      default:                       msg = NULL; break;
    }
    if (msg)
      Log("R72-SPE%d: possible reason: %s\n", thread_index, msg);
      
    ps3_kernel_bug();
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
  #error Code for forked crunchers only - see static Args buffer below
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

void ps3_kernel_bug(void)
{
  Log("This may be caused by kernel bug in SPU scheduler (spufs)\n");
  Log("See readme.cell for more information and possible solutions.\n");
  abort();
}

