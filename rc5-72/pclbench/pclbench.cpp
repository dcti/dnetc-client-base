/*
 * Copyright distributed.net 1998-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: pclbench.cpp,v 1.1.2.1 2003/03/03 01:45:59 andreasb Exp $
 */

#include "baseincs.h"
#include "ccoreio.h"
#include <pcl.h>

// define _only one_ of these:
#define ATHLON
//#define PENTIUM
//#define PENTIUM4


struct perfctr_set_t {
  int event[PCL_MAX_EVENT_PER_CALL+1];
  PCL_CNT_TYPE i_overhead[PCL_MAX_EVENT_PER_CALL];
  PCL_CNT_TYPE i_result[PCL_MAX_EVENT_PER_CALL];
  PCL_FP_CNT_TYPE fp_overhead[PCL_MAX_EVENT_PER_CALL];
  PCL_FP_CNT_TYPE fp_result[PCL_MAX_EVENT_PER_CALL];
} 
#if defined(ATHLON)
/* 4 counters may count any of the events (from hpm -s)
        PCL_L1DCACHE_READWRITE
        PCL_L1DCACHE_HIT
        PCL_L1DCACHE_MISS
        PCL_L1ICACHE_READ
        PCL_L1ICACHE_HIT
        PCL_L1ICACHE_MISS
        PCL_L2CACHE_READWRITE
        PCL_ITLB_MISS
        PCL_DTLB_MISS
        PCL_CYCLES
        PCL_ELAPSED_CYCLES
        PCL_INSTR
        PCL_JUMP_SUCCESS
        PCL_JUMP_UNSUCCESS
        PCL_JUMP
        PCL_STALL_FP
        PCL_IPC                     // rate
        PCL_L1DCACHE_MISSRATE       // rate
*/
perfctr_set[] = {
  /* preload cache */
  { { PCL_CYCLES,
      PCL_INSTR,
      -1
  } },
  /* calculate IPC */
  { { PCL_CYCLES,
      PCL_INSTR,
      -1
  } },
  #if 1
  /* all events */
  { { PCL_L1DCACHE_READWRITE,
      PCL_L1DCACHE_HIT,
      PCL_L1DCACHE_MISS,
      -1
  } },
  { { PCL_L1ICACHE_READ,
      PCL_L1ICACHE_HIT,
      PCL_L1ICACHE_MISS,
      -1
  } },
  { { PCL_L2CACHE_READWRITE,
      PCL_ITLB_MISS,
      PCL_DTLB_MISS,
      -1
  } },
  { { PCL_CYCLES,
      PCL_ELAPSED_CYCLES,
      PCL_INSTR,
      -1
  } },
  { { PCL_JUMP_SUCCESS,
      PCL_JUMP_UNSUCCESS,
      PCL_JUMP,
      PCL_STALL_FP,
      -1
  } },
  { { PCL_IPC,
      -1
  } },
  { { PCL_L1DCACHE_MISSRATE,
      -1
  } },
  #endif
  /*
  { { PCL_ITLB_MISS,
      PCL_L1ICACHE_MISS,
      PCL_DTLB_MISS,
      PCL_L1DCACHE_MISS,
      -1
  } },
  */
  { { -1 } } // terminate list
}
#elif defined(PENTIUM4)
perfctr_set[] = {
  #error FIXME!
  { { -1 } } // terminate list
}
#else
  #error FIXME!
#endif
;

/* define count mode */
unsigned int pcl_mode = PCL_MODE_USER;
PCL_DESCR_TYPE descr;


#ifndef MINIMUM_ITERATIONS
#define MINIMUM_ITERATIONS 24
#endif

#define CoNCaT(a,b) a##b
#define CONCAT(a,b) CoNCaT(a,b)
#define COREFUNCNAME CONCAT(rc5_72_unit_func_, COREFUNC)

extern "C" s32 CDECL COREFUNCNAME(RC5_72UnitWork *, u32 *, void *);
extern "C" s32 CDECL dummycore(RC5_72UnitWork *, u32 *, void *);

s32 CDECL dummycore(RC5_72UnitWork *, u32 *, void *)
{
  return 42;
}


int Run_Overhead(u32 *keyscheckedP)
{
  u32 keystocheck = *keyscheckedP;

  /* Allocate a handle */
  if (PCLinit (&descr) != PCL_SUCCESS)
    printf ("cannot get handle\n");

  /* loop over all the counter sets */
  for (int set = 0; /* */; ++set)
  {
    int numevents;
    for (numevents = 0; perfctr_set[set].event[numevents] != -1; ++numevents)
      ;
    if (!numevents)
      break;
    
    /* initialize data to do work */
    *keyscheckedP = keystocheck;

    /* Check if this is possible on the machine. */
    if (PCLquery (descr, perfctr_set[set].event, numevents, pcl_mode) != PCL_SUCCESS) {
      printf ("requested events not possible\n");
      continue;
    }

    /* Start performance counting.
     *        We have checked already the requested functionality
     *               with PCL_query, so no error check would be necessary. */
    if (PCLstart (descr, perfctr_set[set].event, numevents, pcl_mode) != PCL_SUCCESS)
      printf ("something went wrong\n");

    /************************* DO WORK **********************************/

    int resultcode = dummycore(NULL, keyscheckedP, NULL);

    /************************** DID WORK *********************************/

    /* Stop performance counting and get the counter values. */
    if (PCLstop (descr, perfctr_set[set].i_overhead, perfctr_set[set].fp_overhead, numevents) != PCL_SUCCESS)
      printf ("problems with stopping counters\n");

    /* print out results */
    if (set == 0)
      printf("processed %d of %d keys (%d)\n", *keyscheckedP, keystocheck, resultcode);
    for (int i = 0; i < numevents; ++i) {
      printf("%-22s: ", PCLeventname(perfctr_set[set].event[i]));
      if (PCL_EVENT_IS_INT(perfctr_set[set].event[i]))
        printf("%15.0f ", 
            ((double) perfctr_set[set].i_overhead[i]));
      else
        printf("%21.5f", ((double) perfctr_set[set].fp_overhead[i]));
      printf("\n");
    }
    
  }

  /* Deallocate handle */
  if (PCLexit (descr) != PCL_SUCCESS)
    printf ("cannot release handle\n");

  return 0;
}

int Run_RC5_72(u32 *keyscheckedP)
{
  u32 keystocheck = *keyscheckedP;

  RC5_72UnitWork work;

  if (keystocheck < MINIMUM_ITERATIONS)
    keystocheck = MINIMUM_ITERATIONS;
  else if ((keystocheck % MINIMUM_ITERATIONS) != 0)
    keystocheck += (MINIMUM_ITERATIONS - (keystocheck % MINIMUM_ITERATIONS));

  /* Allocate a handle */
  if (PCLinit (&descr) != PCL_SUCCESS)
    printf ("cannot get handle\n");

  /* loop over all the counter sets */
  for (int set = 0; /* */; ++set)
  {
    int numevents;
    for (numevents = 0; perfctr_set[set].event[numevents] != -1; ++numevents)
      ;
    if (!numevents)
      break;
    
    /* initialize data to do work */
    *keyscheckedP = keystocheck;
    memset(&work, 0, sizeof(work));

    /* Check if this is possible on the machine. */
    if (PCLquery (descr, perfctr_set[set].event, numevents, pcl_mode) != PCL_SUCCESS) {
      printf ("requested events not possible\n");
      continue;
    }

    /* Start performance counting.
     *        We have checked already the requested functionality
     *               with PCL_query, so no error check would be necessary. */
    if (PCLstart (descr, perfctr_set[set].event, numevents, pcl_mode) != PCL_SUCCESS)
      printf ("something went wrong\n");

    /************************* DO WORK **********************************/

    int resultcode = COREFUNCNAME(&work, keyscheckedP, NULL);

    /************************** DID WORK *********************************/

    /* Stop performance counting and get the counter values. */
    if (PCLstop (descr, perfctr_set[set].i_result, perfctr_set[set].fp_result, numevents) != PCL_SUCCESS)
      printf ("problems with stopping counters\n");

    /* print out results */
    if (set == 0)
      printf("processed %d of %d keys (%d)\n", *keyscheckedP, keystocheck, resultcode);
    for (int i = 0; i < numevents; ++i) {
      printf("%-22s: ", PCLeventname(perfctr_set[set].event[i]));
      if (PCL_EVENT_IS_INT(perfctr_set[set].event[i]))
        printf("%15.0f %10.3f/key %15.0f %10.3f/key ", 
            ((double) perfctr_set[set].i_result[i]),
            ((double) perfctr_set[set].i_result[i]) / ((double) *keyscheckedP),
            ((double) (perfctr_set[set].i_result[i] - perfctr_set[set].i_overhead[i])),
            ((double) (perfctr_set[set].i_result[i] - perfctr_set[set].i_overhead[i])) / ((double) *keyscheckedP));
      else
        printf("%21.5f", ((double) perfctr_set[set].fp_result[i]));
      if (perfctr_set[set].event[0] == PCL_CYCLES && perfctr_set[set].event[1] == PCL_INSTR && i == 1)
        printf("%10.6f IPC ", (   ((double) (perfctr_set[set].i_result[1] - perfctr_set[set].i_overhead[1]))
                                / ((double) (perfctr_set[set].i_result[0] - perfctr_set[set].i_overhead[0])) ) );
      printf("\n");
    }
    
  }

  /* Deallocate handle */
  if (PCLexit (descr) != PCL_SUCCESS)
    printf ("cannot release handle\n");

  return 0;
}

int main()
{
  u32 keys = 0;
  Run_Overhead(&keys);

  keys = 1;
  Run_RC5_72(&keys);
  keys = 100;
  Run_RC5_72(&keys);
  keys = 1000;
  Run_RC5_72(&keys);
  keys = 10000;
  Run_RC5_72(&keys);
  keys = 50000;
  Run_RC5_72(&keys);
  keys = 100000;
  Run_RC5_72(&keys);
  keys = 1000000;
  Run_RC5_72(&keys);
  keys = 10000000;
  Run_RC5_72(&keys);
}
