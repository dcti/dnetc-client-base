/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ** header is included from cores, so guard against c++ constructs **
*/

#ifndef __PROBLEM_H__
#define __PROBLEM_H__ "@(#)$Id: problem.h,v 1.61.2.12 1999/11/30 00:04:52 cyp Exp $"

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
  #ifdef MAX_MEM_REQUIRED_BY_CORE
  char core_membuffer[MAX_MEM_REQUIRED_BY_CORE];
  #endif
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
  u32 tslice;                    /* -' -- adjusted by non-preemptive OSs */

  u32 permille;    /* used by % bar */
  int loaderflags; /* used by problem loader (probfill.cpp) */

  unsigned int threadindex; /* index of this problem in the problem table */
  int threadindex_is_valid; /* 0 if the problem is not managed by probman*/

  /* this is our generic prototype */
  s32 (*unit_func)( RC5UnitWork *, u32 *timeslice, void *memblk );
  
  #if (CLIENT_CPU == CPU_X86)
  u32 (*x86_unit_func)( RC5UnitWork * , u32 timeslice );
  #endif
  #if (CLIENT_CPU == CPU_68K)
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
  #elif (CLIENT_CPU == CPU_POWERPC) || defined(_AIXALL)
    #if (CLIENT_OS == OS_AIX)     //straight lintilla (or even ansi for POWER)
    s32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
    #elif (CLIENT_OS == OS_WIN32) //ansi core
    s32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
    #else                         //lintilla wrappers
    s32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 *timeslice );
    #endif
  #elif (CLIENT_CPU == CPU_POWER) //POWER must always be _after_ PPC
  s32 (*rc5_unit_func)( RC5UnitWork * rc5unitwork, u32 timeslice );
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
};

#endif /* __cplusplus */

#endif /* __PROBLEM_H__ */


