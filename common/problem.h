/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ** header is included from cores, so guard against c++ constructs **
*/

#ifndef __PROBLEM_H__
#define __PROBLEM_H__ "@(#)$Id: problem.h,v 1.61.2.46 2000/12/21 16:56:44 cyp Exp $"

#include "cputypes.h" /* u32 */
#include "ccoreio.h"  /* Crypto core stuff (including RESULT_* enum members) */
#if defined(HAVE_OGR_CORES)
#include "ogr.h"      /* OGR core stuff */
#endif

enum {
  RC5, // http://www.rsa.com/rsalabs/97challenge/
  DES, // http://www.rsa.com/rsalabs/des3/index.html
  OGR, // http://members.aol.com/golomb20/
  CSC  // http://www.cie-signaux.fr/security/index.htm
};
#define CONTEST_COUNT       4  /* RC5,DES,OGR,CSC */

/* ---------------------------------------------------------------------- */

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
  #if __VEC__ /* We might use AltiVec */
     #if CORE_MEM_ALIGNMENT < 4
       #undef CORE_MEM_ALIGNMENT
       #define CORE_MEM_ALIGNMENT 4
     #endif
  #else
     #if CORE_MEM_ALIGNMENT < 3
       #undef CORE_MEM_ALIGNMENT
       #define CORE_MEM_ALIGNMENT 3
     #endif
  #endif
#endif

/* ---------------------------------------------------------------------- */

#ifndef MIPSpro
#pragma pack(1)
#endif

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
  #if defined(HAVE_OGR_CORES)
  struct {
    struct WorkStub workstub; // stub to work on (28 bytes)
    struct {u32 hi,lo;} nodes;            // nodes completed
    char unused[12];
  } ogr;
  #endif
} ContestWork;

typedef struct
{
  ContestWork work;/* {key,iv,plain,cypher,keysdone,iter} or {stub,pad} */
  u32  resultcode; /* core state: RESULT_WORKING:0|NOTHING:1|FOUND:2 */
  char id[59];     /* d.net id of worker that last used this */
  u8   contest;    /* 0=rc5,1=des,etc. If this is changed, make this u32 */
  u8   cpu;        /* 97.11.25 If this is ever changed, make this u32 */
  u8   os;         /* 97.11.25 If this is ever changed, make this u32 */
  u8   buildhi;    /* 97.11.25 If this is ever changed, make this u32 */
  u8   buildlo;    /* 97.11.25 If this is ever changed, make this u32 */
} WorkRecord;

#ifndef MIPSpro
# pragma pack()
#endif /* ! MIPSpro */

/* ---------------------------------------------------------------------- */

/* the unit_func_union union and struct selcore are here rather than in
 * selcore.h because they need RC5UnitWork/CoreDispatchTable to
 * be prototyped first. (yes, its yucky)
*/

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

struct selcore
{
  int client_cpu;
  int pipeline_count;
  int use_generic_proto;
  int cruncher_is_asynchronous;
  unit_func_union unit_func;
};

/* ---------------------------------------------------------------------- */

struct problem_publics
{
  u32 elapsed_time_sec, elapsed_time_usec; /* wall clock time between
        start/finish, only valid after Run() returned RESULT_NOTHING/_FOUND */
  u32 runtime_sec, runtime_usec; /* ~total user time spent in core */
  u32 last_runtime_sec, last_runtime_usec; /* time spent in core in last run */
  int last_runtime_is_invalid; /* last_runtime was bad (clock change etc) */
  u32 core_run_count; /* used by go_mt and other things */

  struct
  { 
    u32 avg_coretime_usecs;
  } profiling;                   /* -- managed by non-preemptive OSs     */
  struct
  {
    u32 ccounthi, ccountlo;
    u32 ctimehi,  ctimelo;
    u32 utimehi,  utimelo;
    int init;
  } live_rate[2];                /* -- payload for ContestGetLiveRate    */

  u32 startpermille;             /* -,                                   */
  struct {u32 hi,lo;} startkeys;
  unsigned int contest;          /*  |__ assigned in LoadState()         */
  int coresel;                   /*  |                                   */
  int client_cpu;                /*  | effective CLIENT_CPU              */
  u32 tslice;                    /* -' -- adjusted by non-preemptive OSs */
  int was_reset;                 /* set if loadstate reset the block     */
  int is_random;                 /* set if problem was RC5 'random'      */
  int is_benchmark;              /* set if problem is benchmark          */

  int loaderflags; /* used by problem loader (probfill.cpp) */

  unsigned int pipeline_count;
  unit_func_union unit_func;
  int use_generic_proto; /* RC5/DES unit_func prototype is generic form */
  int cruncher_is_asynchronous; /* on a co-processor or similar */

};

typedef struct
{
  struct problem_publics pub_data;
} Problem;

/*
 * in the following functions that take a __thisprob argument, __thisprob
 * must be a void * to suppress the name mangling for struct Problem.
*/

// Load state into internal structures.
// state is invalid (will generate errors) until this is called.
// expected_[core|cpu|os|buildnum] are those loaded with the workunit
//   and allow LoadState to reset the problem if deemed necessary.
// returns: -1 on error, 0 is OK
// LoadState() and RetrieveState() work in pairs. A LoadState() without
// a previous RetrieveState(,,purge) will fail, and vice-versa.

#define CONTESTWORK_MAGIC_RANDOM    ((const ContestWork *)0)
#define CONTESTWORK_MAGIC_BENCHMARK ((const ContestWork *)1)
int ProblemLoadState( void *__thisprob,
                      const ContestWork * work, unsigned int _contest, 
                      u32 _iterations, int expected_cpunum, 
                      int expected_corenum,
                      int expected_os, int expected_buildfrac );

// Retrieve state from internal structures.
// state is invalid (will generate errors) once the state is purged.
// 'dontwait' signifies that the purge need not wait for the cruncher
// to be in a stable state before purging. *not* waiting is necessary
// when the client is aborting (in which case threads may be hung).
// Returns RESULT_* or -1 if error.
// LoadState() and RetrieveState() work in pairs. A LoadState() without
// a previous RetrieveState(,,purge) will fail, and vice-versa.
int ProblemRetrieveState( void *__thisprob,
                          ContestWork * work, unsigned int *contestid, 
                          int dopurge, int dontwait );


// is the problem initialized? (LoadState() successful, no RetrieveState yet)
// returns > 0 if completed (RESULT_xxx), < 0 if still working
int ProblemIsInitialized(void *__thisprob);

// Runs calling unit_func for iterations times...
// Returns RESULT_* or -1 if error.
int ProblemRun(void *__thisprob);

// more than you'll ever care to know :) any arg can be 0/null */
// returns RESULT_* or -1 if bad state
// *tcount* == total (n/a if not finished), *ccount* == numdone so far 
// this time, *dcount* == numdone so far all times. 
// numstring_style: -1=unformatted, 0=commas, 
// 1=0+space between magna and number (or at end), 2=1+"nodes"/"keys"
int ProblemGetInfo(void *__thisprob,
                     unsigned int *cont_id, const char **cont_name, 
                     u32 *elapsed_secs, u32 *elapsed_usecs, 
                     unsigned int *swucount, int numstring_style,
                     const char **unit_name, 
                     unsigned int *c_permille, unsigned int *s_permille,
                     int permille_only_if_exact,
                     char *idbuf, unsigned int idbufsz,
                     u32 *ratehi, u32 *ratelo,
                     char *ratebuf, unsigned int ratebufsz,
                     u32 *ubtcounthi, u32 *ubtcountlo, 
                     char *tcountbuf, unsigned int tcountbufsz,
                     u32 *ubccounthi, u32 *ubccountlo, 
                     char *ccountbuf, unsigned int ccountbufsz,
                     u32 *ubdcounthi, u32 *ubdcountlo, 
                     char *dcountbuf, unsigned int dcountbufsz);

Problem *ProblemAlloc(void);
void ProblemFree(void *__thisprob);

/* Get the number of problems for a particular contest, */
/*  or, if contestid is -1 then the total for all contests */
unsigned int ProblemCountLoaded(int contestid);

/* Get the size of the problem (which is not! sizeof(Problem) */
/* used for IPC, shmem et al */
unsigned int ProblemGetSize(void);

const char *ProblemComputeRate( unsigned int contestid, 
                                u32 secs, u32 usecs, u32 iterhi, u32 iterlo, 
                                u32 *ratehi, u32 *ratelo,
                                char *ratebuf, unsigned int ratebufsz ); 

int ProblemGetSWUCount( const ContestWork *work,
                        int rescode, unsigned int contestid,
                        unsigned int *swucount );

int IsProblemLoadPermitted(long prob_index, unsigned int contest_i);
/* result depends on #ifdefs, threadsafety issues etc */

#endif /* __PROBLEM_H__ */
