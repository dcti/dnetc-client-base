/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ** header is included from cores, so guard against c++ constructs **
*/

#ifndef __PROBLEM_H__
#define __PROBLEM_H__ "@(#)$Id: problem.h,v 1.61.2.13 1999/12/08 00:41:56 cyp Exp $"

#include "cputypes.h"
#include "ccoreio.h" /* Crypto core stuff (including RESULT_* enum members) */
#include "ogr.h"     /* OGR core stuff */

/* ---------------------------------------------------------------------- */

int IsProblemLoadPermitted(long prob_index, unsigned int contest_i);
/* result depends on #ifdefs, threadsafety issues etc */

/* ----------------------------------------------------------------------- */

#undef MAX_MEM_REQUIRED_BY_CORE
#define MAX_MEM_REQUIRED_BY_CORE  8  //64 bits

#if defined(HAVE_DES_CORES) && defined(MMX_BITSLICER)
  #if MAX_MEM_REQUIRED_BY_CORE < (17*1024)
     #undef MAX_MEM_REQUIRED_BY_CORE
     #define MAX_MEM_REQUIRED_BY_CORE (17*1024)
  #endif
#endif
#if defined(HAVE_CSC_CORES)
  #if MAX_MEM_REQUIRED_BY_CORE < (17*1024)
     #undef MAX_MEM_REQUIRED_BY_CORE
     #define MAX_MEM_REQUIRED_BY_CORE (17*1024)
  #endif      
#endif
#if defined(HAVE_OGR_CORES)
  #if MAX_MEM_REQUIRED_BY_CORE < OGR_PROBLEM_SIZE
     #undef MAX_MEM_REQUIRED_BY_CORE
     #define MAX_MEM_REQUIRED_BY_CORE OGR_PROBLEM_SIZE
  #endif     
#endif    
  
/* ---------------------------------------------------------------------- */

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
    struct WorkStub workstub; // stub to work on (28 bytes)
    u64 nodes;            // nodes completed
    char unused[12];
  } ogr;
} ContestWork;

#ifdef __cplusplus

class Problem
{
protected: /* these members *must* be protected for thread safety */
  /* --------------------------------------------------------------- */
  RC5UnitWork rc5unitwork; /* MUST BE longword (64bit) aligned */
  u64 refL0;               
  ContestWork contestwork;
  CoreDispatchTable *ogr;
  /* --------------------------------------------------------------- */
  char core_membuffer[MAX_MEM_REQUIRED_BY_CORE];
  u32 timehi, timelo;
  int last_resultcode; /* the rescode the last time contestwork was stable */
  int started;
  int initialized;
public: /* anything public must be thread safe */
  unsigned int pipeline_count;
  u32 runtime_sec, runtime_usec; /* ~total time spent in core */
  u32 last_runtime_sec, last_runtime_usec; /* time spent in core in last run */
  u32 core_run_count; /* used by go_mt and other things */

  struct
  { u32 avg_coretime_usecs;
  } profiling;                   /* -- managed by non-preemptive OSs     */

  u32 startpermille;             /* -,                                   */
  unsigned int contest;          /*  |__ assigned in LoadState()         */
  int coresel;                   /*  |                                   */
  int client_cpu;                /*  | effective CLIENT_CPU              */
  u32 tslice;                    /* -' -- adjusted by non-preemptive OSs */

  u32 permille;    /* used by % bar */
  int loaderflags; /* used by problem loader (probfill.cpp) */

  unsigned int threadindex; /* index of this problem in the problem table */
  int threadindex_is_valid; /* 0 if the problem is not managed by probman*/

  /* this is our generic prototype */
  s32 (*unit_func)( RC5UnitWork *, u32 *iterations, void *memblk );
  
  #if ((CLIENT_CPU == CPU_68K) && \
    ((CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_AMIGAOS)))
    __asm u32 (*rc5_unit_func)(register __a0 RC5UnitWork *, register __d0 u32);
  #elif (CLIENT_CPU == CPU_ARM)
    u32 (*rc5_unit_func)( RC5UnitWork * , unsigned long iterations );
  #elif (CLIENT_CPU == CPU_ALPHA)
    u32 (*rc5_unit_func)( RC5UnitWork * , unsigned long iterations );
  #else
    u32 (*rc5_unit_func)( RC5UnitWork * , u32 iterations );
  #endif

#if defined(HAVE_DES_CORES)
  u32 (*des_unit_func)( RC5UnitWork * , u32 *iterations, char *membuf );
#endif  

  int Run_RC5(u32 *iterations,int *core_retcode); /* \  run for n iterations.              */
  int Run_DES(u32 *iterations,int *core_retcode); /*  > set actual number of iter that ran */
  int Run_OGR(u32 *iterations,int *core_retcode); /* /  returns RESULT_* or -1 if error    */
  int Run_CSC(u32 *iterations,int *core_retcode); /* /                                     */

  Problem(long _threadindex = -1L);
  ~Problem();

  int IsInitialized() { return (initialized!=0); }

  int LoadState( ContestWork * work, unsigned int _contest, u32 _iterations, 
     int _unused );
    // Load state into internal structures.
    // state is invalid (will generate errors) until this is called.
    // expected_[core|cpu|buildnum] are those loaded with the workunit
    //   and allow LoadState to reset the problem if deemed necessary.
    // returns: -1 on error, 0 is OK

  int RetrieveState( ContestWork * work, unsigned int *contestid, int dopurge );
    // Retrieve state from internal structures.
    // state is invalid (will generate errors) once the state is purged.
    // Returns RESULT_* or -1 if error.

  int Run(void);
    // Runs calling rc5_unit for iterations times...
    // Returns RESULT_* or -1 if error.

  u32 CalcPermille();
    /* Return the % completed in the current block, to nearest 0.1%. */
};

#endif /* __cplusplus */

#endif /* __PROBLEM_H__ */

