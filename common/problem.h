/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ** header is included from cores, so guard against c++ constructs **
*/

#ifndef __PROBLEM_H__
#define __PROBLEM_H__ "@(#)$Id: problem.h,v 1.81 2000/07/11 07:42:32 mfeiri Exp $"

#include "cputypes.h"
#include "ccoreio.h" /* Crypto core stuff (including RESULT_* enum members) */
#include "ogr.h"     /* OGR core stuff */

/* ---------------------------------------------------------------------- */

int IsProblemLoadPermitted(long prob_index, unsigned int contest_i);
/* result depends on #ifdefs, threadsafety issues etc */


#undef MAX_MEM_REQUIRED_BY_CORE
#define MAX_MEM_REQUIRED_BY_CORE  8  //64 bits
// Problem->core_membuffer should be aligned to 2^CORE_MEM_ALIGNMENT
#define CORE_MEM_ALIGNMENT 3

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
  // CSC membuffer should be aligned to a 16-byte boundary
  #if CORE_MEM_ALIGNMENT < 4
     #undef CORE_MEM_ALIGNMENT
     #define CORE_MEM_ALIGNMENT 4
  #endif
#endif
#if defined(HAVE_OGR_CORES)
  #if MAX_MEM_REQUIRED_BY_CORE < OGR_PROBLEM_SIZE
     #undef MAX_MEM_REQUIRED_BY_CORE
     #define MAX_MEM_REQUIRED_BY_CORE OGR_PROBLEM_SIZE
  #endif
  // OGR membuffer should be aligned to a 8-byte boundary
  // (essential for non-x86 CPUs)
  #if CORE_MEM_ALIGNMENT < 3
     #undef CORE_MEM_ALIGNMENT
     #define CORE_MEM_ALIGNMENT 3
  #endif
#endif

/* ---------------------------------------------------------------------- */

typedef union
{
  struct {
    struct {u32 hi,lo;} key;              // starting key
    struct {u32 hi,lo;} iv;               // initialization vector
    struct {u32 hi,lo;} plain;            // plaintext we're searching for
    struct {u32 hi,lo;} cypher;           // cyphertext
    struct {u32 hi,lo;} keysdone;         // iterations done (also current position in block)
    struct {u32 hi,lo;} iterations;       // iterations to do
  } crypto;
  struct {
    struct WorkStub workstub; // stub to work on (28 bytes)
    struct {u32 hi,lo;} nodes;            // nodes completed
    char unused[12];
  } ogr;
} ContestWork;

typedef union
{
    /* this is our generic prototype */
    s32 (*gen)( RC5UnitWork *, u32 *iterations, void *memblk );
    #if (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_68K)
    u32 __regargs (*rc5)( RC5UnitWork * , u32 iterations );
    #else
    u32 (*rc5)( RC5UnitWork * , u32 iterations );
    #endif
    #if defined(HAVE_DES_CORES)
    u32 (*des)( RC5UnitWork * , u32 *iterations, char *membuf );
    #endif  
    #if defined(HAVE_OGR_CORES)
    CoreDispatchTable *ogr;
    #endif  
} unit_func_union;


#ifdef __cplusplus

class Problem
{
protected: /* these members *must* be protected for thread safety */
  /* --------------------------------------------------------------- */
  RC5UnitWork rc5unitwork; /* MUST BE longword (64bit) aligned */
  struct {u32 hi,lo;} refL0;               
  ContestWork contestwork;
  /* --------------------------------------------------------------- */
  char __core_membuffer_space[(MAX_MEM_REQUIRED_BY_CORE+(1UL<<CORE_MEM_ALIGNMENT)-1)];
  void *core_membuffer; /* aligned pointer to __core_membuffer_space */
  /* --------------------------------------------------------------- */
  u32 timehi, timelo;
  int last_resultcode; /* the rescode the last time contestwork was stable */
  int started;
  int initialized;
  unsigned int threadindex; /* 0-n (globally unique identifier) */
  volatile int running; /* RetrieveState(,,purge) has to wait while Run()ning */

public: /* anything public must be thread safe */
  u32 completion_timehi, completion_timelo; /* wall clock time between start/finish */
  u32 runtime_sec, runtime_usec; /* ~total user time spent in core */
  u32 last_runtime_sec, last_runtime_usec; /* time spent in core in last run */
  int last_runtime_is_invalid; /* last_runtime was bad (clock change etc) */
  u32 core_run_count; /* used by go_mt and other things */

  struct
  { u32 avg_coretime_usecs;
  } profiling;                   /* -- managed by non-preemptive OSs     */

  u32 startpermille;             /* -,                                   */
  struct {u32 hi,lo;} startkeys;
  unsigned int contest;          /*  |__ assigned in LoadState()         */
  int coresel;                   /*  |                                   */
  int client_cpu;                /*  | effective CLIENT_CPU              */
  u32 tslice;                    /* -' -- adjusted by non-preemptive OSs */
  int was_reset;                 /* set if loadstate reset the block     */

  int loaderflags; /* used by problem loader (probfill.cpp) */

  unsigned int pipeline_count;
  unit_func_union unit_func;
  int use_generic_proto; /* RC5/DES unit_func prototype is generic form */
  int cruncher_is_asynchronous; /* on a co-processor or similar */

  int Run_RC5(u32 *iterations,int *core_retcode); /* \  run for n iterations.              */
  int Run_DES(u32 *iterations,int *core_retcode); /*  > set actual number of iter that ran */
  int Run_OGR(u32 *iterations,int *core_retcode); /* /  returns RESULT_* or -1 if error    */
  int Run_CSC(u32 *iterations,int *core_retcode); /* /                                     */

  Problem();
  ~Problem();

  int IsInitialized() { return (initialized!=0); }

  // LoadState() and RetrieveState() work in pairs. A LoadState() without
  // a previous RetrieveState(,,purge) will fail, and vice-versa.
  
  int LoadState( ContestWork * work, unsigned int _contest, u32 _iterations, 
     int expected_cpunum, int expected_corenum, 
     int expected_os, int expected_buildfrac );
    // Load state into internal structures.
    // state is invalid (will generate errors) until this is called.
    // expected_[core|cpu|os|buildnum] are those loaded with the workunit
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

/* ------------------------------------------------------------------- */

/* RC5/DES/CSC 2^28 key count conversion.
   belongs in ccoreio.c[pp], but that doesn't exist, and its not worth
   creating for this itty-bitty thing.
*/
inline u32 __iter2norm( u32 iterlo, u32 iterhi )
{
  iterlo = ((iterlo >> 28) + (iterhi << 4));
  if (!iterlo)
    iterlo++;
  return iterlo;
}

#endif /* __cplusplus */

#endif /* __PROBLEM_H__ */
