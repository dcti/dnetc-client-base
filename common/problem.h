/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PROBLEM_H__
#define __PROBLEM_H__ "@(#)$Id: problem.h,v 1.55 1999/04/08 20:05:50 patrick Exp $"

#include "cputypes.h"
#include "ccoreio.h" /* Crypto core stuff (including RESULT_* enum members) */
#include "ogr.h"     /* OGR core stuff */

#if (CLIENT_CPU == CPU_X86)
  #define MAX_MEM_REQUIRED_BY_CORE (17*1024)
#endif

#if !defined(MEGGS) && !defined(DES_ULTRA) && !defined(DWORZ)
  #define MIN_DES_BITS  8
  #define MAX_DES_BITS 24
#else
  #if defined(BIT_32)
    #define MIN_DES_BITS 19
    #define MAX_DES_BITS 19
  #elif (defined(BIT_64) && defined(BITSLICER_WITH_LESS_BITS) && !defined(DWORZ))
    #define MIN_DES_BITS 16
    #define MAX_DES_BITS 16
  #elif defined(BIT_64)
    #define MIN_DES_BITS 20
    #define MAX_DES_BITS 20
  #endif
#endif

typedef union
{
  struct {
    u64 key;              // starting key
    u64 iv;               // initialization vector
    u64 plain;            // plaintext we're searching for
    u64 cypher;           // cyphertext
    u64 keysdone;         // iterations done (also current position in block)
    u64 iterations;       // iterations to do
  } crypto;
  struct {
    Stub stub;            // stub to work on (24 bytes)
    char unused[24];
  } ogr;
} ContestWork;

class Problem
{
public:
  int finished;
  int resultcode; /* previously rc5result.result */
  u32 startpercent;
  u32 percent;
  u32 runtime_sec, runtime_usec; /* ~time spent in core */
  int restart;
  u32 timehi, timelo;
  int started;
  unsigned int contest;
  int cputype;

  unsigned int pipeline_count;
  u32 tslice; 
  int loaderflags; /* used by problem loader (probfill.cpp) */

  #ifdef MAX_MEM_REQUIRED_BY_CORE
  char core_membuffer[MAX_MEM_REQUIRED_BY_CORE];
  #endif

  unsigned int threadindex; /* index of this problem in the problem table */
  int threadindex_is_valid; /* 0 if the problem is not managed by probman*/
  
  u32 initialized;
  ContestWork contestwork;
  RC5UnitWork rc5unitwork;
  u64 refL0;
  CoreDispatchTable *ogr;
  void *ogrstate;

  #if (CLIENT_CPU == CPU_X86)
  u32 (*unit_func)( RC5UnitWork * , u32 timeslice );
  #elif (CLIENT_CPU == CPU_68K)
  extern "C" __asm u32 (*rc5_unit_func)
       ( register __a0 RC5UnitWork * , register __d0 u32 timeslice);
  #elif (CLIENT_CPU == CPU_ARM)
  u32 (*rc5_unit_func)( RC5UnitWork * , unsigned long iterations );
  u32 (*des_unit_func)( RC5UnitWork * , u32 timeslice );
  #elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_DEC_UNIX)
  u32 (*rc5_unit_func)( RC5UnitWork * );
  u32 (*des_unit_func)( RC5UnitWork * , u32 nbits );
  #elif (CLIENT_CPU == CPU_ALPHA)
  u32 (*rc5_unit_func)( RC5UnitWork * , unsigned long iterations );
  u32 (*des_unit_func)( RC5UnitWork * , u32 nbits );
  #elif (CLIENT_OS == OS_AIX)
  s32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
  #elif (CLIENT_CPU == CPU_POWERPC)
  int (*rc5_unit_func)( RC5UnitWork * , unsigned long iterations );
  #endif

  int Run_RC5(u32 *timeslice); /* \    run for n timeslices.                */
  int Run_DES(u32 *timeslice); /*  > - set actual number of slices that ran */
  int Run_OGR(u32 *timeslice); /* /    return same retcode as Run()         */

  Problem(long _threadindex = -1L);
  ~Problem();

  int IsInitialized() { return (initialized!=0); }

  int LoadState( ContestWork * work, unsigned int _contest, 
		 u32 _timeslice, int _cputype );
    // Load state into internal structures.
    // state is invalid (will generate errors) until this is called.
    // returns: -1 on error, 0 is OK

  int RetrieveState( ContestWork * work, unsigned int *contestid, int dopurge );
    // Retrieve state from internal structures.
    // state is invalid (will generate errors) once the state is purged.
    // returns: -1 on error, resultcode otherwise

  s32 Run( u32 /* unused */ );
    // Runs calling rc5_unit for timeslice times...
    // Returns:
    //   -1 if something goes wrong (state not loaded, already done etc...)
    //   0 if more work to be done
    //   1 if we're done, go get results

  u32 CalcPercent() { return (u32)( ((double)(100.0)) *
    /* Return the % completed in the current block, to nearest 1%. */
        (((((double)(contestwork.crypto.keysdone.hi))*((double)(4294967296.0)))+
                                 ((double)(contestwork.crypto.keysdone.lo))) /
        ((((double)(contestwork.crypto.iterations.hi))*((double)(4294967296.0)))+
                                 ((double)(contestwork.crypto.iterations.lo)))) ); }

#if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
  u32 GetKeysDone() { return(contestwork.crypto.keysdone.lo); }
    // Returns keys completed for Mac GUI display.
#endif

};

#endif /* __PROBLEM_H__ */


