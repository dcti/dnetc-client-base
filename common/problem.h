/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __PROBLEM_H__
#define __PROBLEM_H__ "@(#)$Id: problem.h,v 1.62 1999/06/22 20:06:54 chrisb Exp $"

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
    WorkStub workstub;    // stub to work on (28 bytes)
    u64 nodes;            // nodes completed
    char unused[12];
  } ogr;
} ContestWork;

class Problem
{
protected: /* these members *must* be protected for thread safety */
  int last_resultcode; /* the rescode the last time contestwork was stable */
  int started;
  int initialized;
  ContestWork contestwork;
  RC5UnitWork rc5unitwork;
  u64 refL0;
  CoreDispatchTable *ogr;
  void *ogrstate;
  unsigned int pipeline_count;
  #ifdef MAX_MEM_REQUIRED_BY_CORE
  char core_membuffer[MAX_MEM_REQUIRED_BY_CORE];
  #endif
  u32 timehi, timelo;
public: /* anything public must be thread safe */
  u32 runtime_sec, runtime_usec; /* ~total time spent in core */
  u32 last_runtime_sec, last_runtime_usec; /* time spent in core in last run */
  u32 core_run_count; /* used by go_mt and other things */

  struct
  { u32 avg_coretime_usecs;
  } profiling;                   /* -- managed by non-preemptive OSs     */

  u32 startpermille;             /* -,                                   */
  unsigned int contest;          /*  |__ assigned in LoadState()         */
  int cputype;                   /*  |                                   */
  u32 tslice;                    /* -' -- adjusted by non-preemptive OSs */

  u32 permille;    /* used by % bar */
  int loaderflags; /* used by problem loader (probfill.cpp) */

  unsigned int threadindex; /* index of this problem in the problem table */
  int threadindex_is_valid; /* 0 if the problem is not managed by probman*/
  
  #if (CLIENT_CPU == CPU_X86)
  u32 (*unit_func)( RC5UnitWork * , u32 timeslice );
  #elif (CLIENT_CPU == CPU_68K)
  extern "C" __asm u32 (*rc5_unit_func)
       ( register __a0 RC5UnitWork * , register __d0 u32 timeslice);
  #elif (CLIENT_CPU == CPU_ARM)
  u32 (*rc5_unit_func)( RC5UnitWork * , unsigned long iterations );
  u32 (*des_unit_func)( RC5UnitWork * , unsigned long iterations );
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

  int Run_RC5(u32 *timeslice,int *core_retcode); /* \  run for n timeslices.                */
  int Run_DES(u32 *timeslice,int *core_retcode); /*  > set actual number of slices that ran */
  int Run_OGR(u32 *timeslice,int *core_retcode); /* /  returns RESULT_* or -1 if error      */
  int Run_CSC(u32 *timeslice,int *core_retcode); /* /                                       */

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
    // Returns RESULT_* or -1 if error.

  int Run(void);
    // Runs calling rc5_unit for timeslice times...
    // Returns RESULT_* or -1 if error.

  u32 CalcPermille();
    /* Return the % completed in the current block, to nearest 0.1%. */

#if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
  u64 GetKeysDone() { return(contestwork.crypto.keysdone); }
    // Returns keys completed for Mac GUI display.
  int GetResultCode() { return(last_resultcode); }
    // Returns result code at completion (no thread safety issue).
#endif

};

#endif /* __PROBLEM_H__ */


