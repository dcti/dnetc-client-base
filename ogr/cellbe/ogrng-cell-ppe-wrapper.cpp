/* 
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *ogrng_cell_ppe_wrapper_cpp(void) {
return "@(#)$Id: ogrng-cell-ppe-wrapper.cpp,v 1.2 2008/06/29 14:20:28 stream Exp $"; }


#include <libspe2.h>
#include <cstdlib>
#include <cstring>
#include "unused.h"
#include "logstuff.h"

#define OGR_NG_GET_DISPATCH_TABLE_FXN    spe_ogrng_get_dispatch_table
#define OGROPT_HAVE_OGR_CYCLE_ASM     2

// #define CELL_FULL_CYCLE

#include "ogrng-cell.h"

#ifndef CORE_NAME
#error CORE_NAME undefined.
#endif

#define SPE_WRAPPER_FUNCTION(name) SPE_WRAPPER_FUNCTION2(name)
#define SPE_WRAPPER_FUNCTION2(name) ogrng_cycle_ ## name ## _spe_wrapper

extern spe_program_handle_t SPE_WRAPPER_FUNCTION(CORE_NAME);

#ifndef HAVE_MULTICRUNCH_VIA_FORK
  #error Code for fork'ed crunchers only - see static Args buffer below
#endif

// #define INTERNAL_TEST

#ifdef INTERNAL_TEST
#define ogr_cycle_256 ogr_cycle_256_test
#endif

#ifndef CELL_FULL_CYCLE
static int ogr_cycle_256(struct OgrState *oState, int *pnodes, const u16 *pchoose)
#else
static int ogr_cycle_entry(void *state, int *pnodes, int dummy)
#endif
{
#ifdef CELL_FULL_CYCLE
  struct OgrState *oState = (struct OgrState *)state;
  u16* pchoose = precomp_limits[oState->maxdepth - OGR_NG_MIN].choose_array;
#endif

  // Check size of structures, these offsets must match assembly
  STATIC_ASSERT(sizeof(struct OgrLevel) == 7*16);
  STATIC_ASSERT(sizeof(struct OgrState) == 2*16 + 7*16*29); /* 29 == OGR_MAXDEPTH */
  STATIC_ASSERT(sizeof(CellOGRCoreArgs) == sizeof(struct OgrState) + 16);
  STATIC_ASSERT(sizeof(struct OgrState) <= OGRNG_PROBLEM_SIZE);
  STATIC_ASSERT(offsetof(CellOGRCoreArgs, state.Levels) == 32);
  STATIC_ASSERT(sizeof(pchoose) == 4); /* pchoose cast to u32 */

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
      Log("OGRNG-SPE#%d! spe_context_create() failed\n", thread_index);
      abort();
    }
    retval = spe_program_load(context, &SPE_WRAPPER_FUNCTION(CORE_NAME));
    if (retval != 0)
    {
      Log("OGRNG-SPE#%d: spe_program_load() returned %d\n", thread_index, retval);
      abort();
    }
    if (posix_memalign(&myCellOGRCoreArgs_void, 128, sizeof(CellOGRCoreArgs)))
    {
      Log("OGRNG-SPE#%d! posix_memalign() failed\n", thread_index);
      abort();
    }

    isInit = true;
  }

  CellOGRCoreArgs* myCellOGRCoreArgs = (CellOGRCoreArgs*)myCellOGRCoreArgs_void;

  // Copy function arguments to CellOGRCoreArgs struct
  memcpy(&myCellOGRCoreArgs->state, oState, sizeof(struct OgrState));
          myCellOGRCoreArgs->pnodes   = *pnodes;
	  myCellOGRCoreArgs->upchoose = (u32)pchoose;

  retval = spe_context_run(context, &entry, 0, (void*)myCellOGRCoreArgs, NULL, &stop_info);
  if (retval != 0)
  {
    Log("OGRNG-SPE#%d: spe_context_run() returned %d\n", thread_index, retval);
    abort();
  }
  retval = spe_stop_info_read(context, &stop_info);
  if (retval != 0)
  {
    Log("OGRNG-SPE#%d: spe_stop_info_read() returned %d\n", thread_index, retval);
    abort();
  }

  __asm__ __volatile__ ("sync" : : : "memory");

  // Fetch return value of the SPE core
  if (stop_info.stop_reason == SPE_EXIT)
    retval = stop_info.result.spe_exit_code;
  else
  {
    Log("Alert: OGRNG-SPE#%d exit status is %d\n", thread_index, stop_info.stop_reason);
    abort();
  }

  // Copy data from CellCoreArgs struct back to the function arguments
  memcpy(oState, &myCellOGRCoreArgs->state, sizeof(struct OgrState));
        *pnodes = myCellOGRCoreArgs->pnodes;

#ifdef CELL_FULL_CYCLE
  return retval;
#else
  return myCellOGRCoreArgs->ret_depth;
#endif
}

#ifdef INTERNAL_TEST
#undef ogr_cycle_256
#include <time.h>
/*
 * Test for direct DMA fetch of pchoose
 */
static int ogr_cycle_256(struct OgrState *state, int *pnodes, const u16 *pchoose)
{
   u32 i;

#if 0
   for (i = 0; i < 128; i++, pchoose++)
   {
     int ret = ogr_cycle_256_test(state, pnodes, pchoose);
     printf("Expected = %u, got = %d%s\n", *pchoose, ret, (*pchoose == ret ? "" : " ** error **"));
   }
#endif
#if 0
   for (i = 0; i < 33; i++)
   {
     unsigned ret, expected = 0xFFFFFFFF >> i;
     *pnodes = i;
     ret = ogr_cycle_256_test(state, pnodes, pchoose);
     printf("%08X %08X%s\n", ret, expected, (ret == expected ? "" : " ** error **"));
   }
#endif     
#if 1
#define START_VALUE  0xFFF00000
   time_t start_time = time(NULL);
   for (i = START_VALUE;;)
   {
     unsigned ret, expected;
     
     *pnodes  = i;
     ret      = ogr_cycle_256_test(state, pnodes, pchoose);
     expected = LOOKUP_FIRSTBLANK(i);
     if (ret != expected)
       printf("Error: input = %u, ret = %u, expected = %u\n", i, ret, expected);
     else
     {
       if ((i & (0x10000 - 1)) == 0)
          printf("%08X iterations OK\n", i);
     }
     if (++i == START_VALUE + 0x100000)
       break;
   }
   printf("SPU call ratio: %u runs/sec.\n", (i-START_VALUE) / (time(NULL)-start_time));
#endif
   exit(0);
}
#endif
