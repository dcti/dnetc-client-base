/*
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -------------------------------------------------------------------
 * Eagleson's Law:
 *    Any code of your own that you haven't looked at for six or more
 *    months, might as well have been written by someone else.  (Eagleson
 *    is an optimist, the real number is more like three weeks.)
 * -------------------------------------------------------------------
*/
const char *problem_cpp(void) {
return "@(#)$Id: problem.cpp,v 1.194 2008/12/31 00:26:17 kakace Exp $"; }

//#define TRACE
#define TRACE_U64OPS(x) TRACE_OUT(x)

#include "cputypes.h"
#include "baseincs.h"
#include "version.h"  //CLIENT_BUILD_FRAC
#include "projdata.h" //general project data: ids, flags, states; names, ...
#include "client.h"   //CONTEST_COUNT
#include "clitime.h"  //CliClock()
#include "logstuff.h" //LogScreen()
#include "probman.h"  //GetProblemPointerFromIndex()
#include "random.h"   //Random()
#include "rsadata.h"  //Get cipher/etc for random blocks
#include "clicdata.h" //CliSetContestWorkUnitSpeed()
#include "selcore.h"  //selcoreGetSelectedCoreForContest()
#include "util.h"     //trace, DNETC_UNUSED_*
#include "cpucheck.h" //hardware detection
#include "console.h"  //ConOutErr
#include "triggers.h" //RaiseExitRequestTrigger()
#include "clisync.h"  //synchronisation primitives
#include "coremem.h"  //cmem_alloc() and cmem_free()
#include "problem.h"  //ourselves

#if (CLIENT_OS == OS_QNX)
  #undef offsetof
  #define offsetof(__typ,__id) ((size_t)&(((__typ*)0)->__id))
#else
  #include <stddef.h> /* offsetof */
#endif

#if (CLIENT_OS == OS_MORPHOS)
  #define SAVE_CLIENT_OS_CONTEXT    APTR *__ehptr = (APTR *) (((IPTR) FindTask(NULL)->tc_ETask) + 130); *__ehptr = MyEmulHandle;
  #define RESTORE_CLIENT_OS_CONTEXT *__ehptr = NULL;
#else
  #define SAVE_CLIENT_OS_CONTEXT
  #define RESTORE_CLIENT_OS_CONTEXT
#endif

/* ------------------------------------------------------------------- */

//#define STRESS_THREADS_AND_BUFFERS /* !be careful with this! */

#ifndef MINIMUM_ITERATIONS
#define MINIMUM_ITERATIONS 48
/*
   MINIMUM_ITERATIONS determines minimum number of iterations that will
   be requested, as well as the boundary on which number of iterations will
   be aligned. This then automatically implies keysdone alignment as well.
   [This applies to partially completed work loaded from disk as well,
   since partially completed work will (should!) never end up on a cruncher
   that it did not originate on because core cpu, core #, client os and
   client version are saved in partially completed work, and work is reset
   by LoadState if any of them don't match.
   24 was chosen because it is evenly divisible by any/all the
   data.pub.pipeline_counts currently in use (1,2,3,4[,6?])
*/
#endif

/* ------------------------------------------------------------------- */

static unsigned int __problem_counter = 0;
/*
   problem_counter is what we copy as threadindex and then increment
   for the next problem. Its addressed/incremented in the constructor
   and addressed/decremented in the destructor, both of which are
   'thread safe' in the sense that they are never called from the
   actual crunching threads themselves.
*/

/* ------------------------------------------------------------------- */

#if (SIZEOF_LONG == 8)  /* SIZEOF_LONG is defined in cputypes.h */
# include "pack8.h"
#else
# include "pack4.h"
#endif

typedef struct
{
  struct problem_publics pub_data; /* public members - must come first */
  struct
  {
    /* the following must be protected for thread safety */
    /* --------------------------------------------------------------- */
// TODO: acidblood/trashover
#ifdef HAVE_CRYPTO_V2
    RC5_72UnitWork rc5_72unitwork;
#endif
    struct {u32 hi,mid,lo;} refL0;
    ContestWork contestwork;
    /* --------------------------------------------------------------- */
    void *core_membuffer; /* aligned pointer to __core_membuffer_space */
    char __core_membuffer_space[(MAX_MEM_REQUIRED_BY_CORE+(1UL<<CORE_MEM_ALIGNMENT)-1)];
    /* --------------------------------------------------------------- */
    u32 loadtime_sec, loadtime_usec; /* LoadState() time */
    int last_resultcode; /* the rescode the last time contestwork was stable */
    int started;
    int initialized;
    unsigned int threadindex; /* 0-n (globally unique identifier) */
  } DNETC_PACKED priv_data;
} DNETC_PACKED InternalProblem;

#include "pack0.h"

/* ======================================================================= */

#include "pack1.h"

/* SuperProblem() is an InternalInternal problem struct                */
/* SuperProblem members are never accessed by any functions other than */
/* those listed immediately below (ProblemAlloc and friends)           */
/*                                                                     */
/* This is how the different copies are used:                          */
/*   LoadState()                                                       */
/*     on entry: locked copy from PICKPROB_MAIN -> PICKPROB_TEMP       */
/*     modifies: PICKPROB_TEMP                                         */
/*     on success: locked copy from PICKPROB_TEMP -> PICKPROB_MAIN     */
/*   RetrieveState()                                                   */
/*   IsProblemInitialized()                                            */
/*     on entry: lock PICKPROB_MAIN                                    */
/*     modifies (only for RetrieveState(purge)): PICKPROB_MAIN         */
/*     on return: unlock PICKPROB_MAIN                                 */
/*   Run()                                                             */
/*     on entry: locked copy from PICKPROB_MAIN -> PICKPROB_CORE       */
/*     modifies: PICKPROB_CORE                                         */
/*     on success and if the PICKPROB_MAIN initialized/started state   */
/*     hasn't changed (that is, RetrieveState() hasn't purged it in    */
/*     meantime), locked copy from PICKPROB_CORE -> PICKPROB_MAIN      */
/*                                                                     */
/*  **NOTE**: this scheme implies that the cores may NOT have          */
/*  absolute pointers in their mem buffers that point to other         */
/*  parts of the membuffer.                                            */
/*                                                                     */
/* The reason why we use a temp area for LoadState() is twofold:       */
/* a) LoadState() can take time, for instance when selcore will need   */
/*    to do a benchmark. If that happens, and Run() starts, it will    */
/*    spin in the lock and (minimally) skew the results.               */
/* b) LoadState() can fail at anytime during the intitalization,       */
/*    which could leave the problem in an inconsistant state.          */

typedef struct
{
  InternalProblem iprobs[3];
  #define PICKPROB_MAIN 0 /* MAIN must be first */
  #define PICKPROB_CORE 1
  #define PICKPROB_TEMP 2 /* temporary copy used by load */
  fastlock_t copy_lock; /* locked when a sync is in progress */
} DNETC_PACKED SuperProblem;

#include "pack0.h"

unsigned int ProblemGetSize(void)
{ /* needed by IPC/shmem */
  return sizeof(SuperProblem);
}

void ProblemFree(void *__thisprob)
{
  SuperProblem *thisprob = (SuperProblem *)__thisprob;
  if (thisprob)
  {
    memset( thisprob, 0, sizeof(SuperProblem) );
    __problem_counter--;
    cmem_free((void *)thisprob);
  }
  return;
}

Problem *ProblemAlloc(void)
{
  char *p;
  SuperProblem *thisprob = (SuperProblem *)0;
  int err = 0;

  err = offsetof(Problem, pub_data);
  if (err != 0)
  {
    Log("alignment error: offsetof(Problem, pub_data) != 0 [%d]\n", err);
    err = 1;
  }
  err = offsetof(InternalProblem, pub_data);
  if (err != 0)
  {
    Log("alignment error: offsetof(InternalProblem, pub_data) != 0 [%d]\n",err);
    err = 1;
  }
  err = offsetof(SuperProblem, iprobs[0]);
  if (err != 0)
  {
    Log("alignment error: offsetof(SuperProblem, iprobs[0]) != 0 [%d]\n",err);
    err = 1;
  }

  #ifdef STRESS_THREADS_AND_BUFFERS
  if (err == 0)
  {
    static int runlevel = 0;
    if (runlevel == 0)
    {
      RaisePauseRequestTrigger();
      LogScreen("Warning! STRESS_THREADS_AND_BUFFERS is defined.\n"
                "Are you sure that the client is pointing at\n"
                "a test proxy? If so, type 'yes': ");
      char getyes[10];
      ConInStr(getyes,4,0);
      ClearPauseRequestTrigger();
      if (strcmp( getyes, "yes" ) == 0)
        runlevel = +1;
      else
        runlevel = -1;
    }
    if (runlevel < 0)
      err = 1;
  }
  #endif

  if (!err)
  {
    thisprob = (SuperProblem *)cmem_alloc(sizeof(SuperProblem));
    if (!thisprob)
    {
      #ifdef HAVE_MULTICRUNCH_VIA_FORK
      Log("Insufficient memory to allocate problem data\n"
          "Do you have CONFIG_SYSVIPC enabled in your kernel?\n");
      #else
      Log("Insufficient memory to allocate problem data\n");
      #endif
      err = 1;
    }
  }

  if (thisprob && !err)
  {
// TODO: acidblood/trashover
#ifdef HAVE_CRYPTO_V2
    p = (char *)&(thisprob->iprobs[PICKPROB_CORE].priv_data.rc5_72unitwork);
    if ((((unsigned long)p) & (sizeof(void *)-1)) != 0)
    {
      /* Ensure that the core data is going to be aligned */
      Log("priv_data.rc5_72unitwork for problem %d is misaligned!\n", __problem_counter);
      err = 1;
    }
#endif
  }

  if (thisprob && !err)
  {
    memset( thisprob, 0, sizeof(SuperProblem) );
    fastlock_init(&(thisprob->copy_lock));

    thisprob->iprobs[PICKPROB_CORE].priv_data.threadindex =
    thisprob->iprobs[PICKPROB_MAIN].priv_data.threadindex =
    thisprob->iprobs[PICKPROB_TEMP].priv_data.threadindex =
                                              __problem_counter++;

    //align core_membuffer to 16byte boundary
    p = &(thisprob->iprobs[PICKPROB_CORE].priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->iprobs[PICKPROB_CORE].priv_data.core_membuffer = p;

    p = &(thisprob->iprobs[PICKPROB_MAIN].priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->iprobs[PICKPROB_MAIN].priv_data.core_membuffer = p;

    p = &(thisprob->iprobs[PICKPROB_TEMP].priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->iprobs[PICKPROB_TEMP].priv_data.core_membuffer = p;
  }

  if (thisprob && err)
  {
    cmem_free((void *)thisprob);
    thisprob = (SuperProblem *)0;
  }
  /* the first member of thisprob is InternalProblem[PICKPROB_MAIN], and
     the first member of an InternalProblem is a Problem.
  */
  return (Problem *)((void *)thisprob);
}

static InternalProblem *__pick_probptr(void *__thisprob, unsigned int which)
{
  if (__thisprob)
  {
    SuperProblem *p = (SuperProblem *)__thisprob;

    // XXX
    // which may be one of the three PICKPROB_* types
    // if we want to, we can put an assert(which < NUM_PICKPROB_TYPES)
    // or something. note that this is only called with #defined
    // positive constants, so changing prototype to unsigned.  - sampo

    return &(p->iprobs[which]);
  }
  return (InternalProblem *)0;
}

static inline void __assert_lock( void *__thisprob )
{
  SuperProblem *p = (SuperProblem *)__thisprob;
  fastlock_lock(&(p->copy_lock));
}

static inline void __release_lock( void *__thisprob )
{
  SuperProblem *p = (SuperProblem *)__thisprob;
  fastlock_unlock(&(p->copy_lock));
}

#undef SuperProblem /* no references beyond this point */

static inline void __copy_internal_problem( InternalProblem *dest,
                                            const InternalProblem *source )
{
    // core_membuffer is a pointer into __core_membuffer_space. This space is
    // not a malloc()ed memblock, but another member (char[]) of
    // InternalProblem.
    // So core_membuffer must not be copied, otherwise it would point outside
    // of it's own InternalProblem.
    // But when copying one InternalProblem into another we must take care of
    // the content of __core_membuffer_space. It must be aligned in the copy,
    // too. The alignment is defined by core_membuffer and the padding may
    // differ from source. -- andreasb

  void *p = dest->priv_data.core_membuffer;
  memcpy( dest, source, sizeof(InternalProblem));
  dest->priv_data.core_membuffer = p;

  // inefficient, because this is the second copy of that area, but we need
  // the core memory aligned and core_membuffer pointing to it.
  memcpy( dest->priv_data.core_membuffer, source->priv_data.core_membuffer,
          MAX_MEM_REQUIRED_BY_CORE );
}

/* ======================================================================= */

#ifdef HAVE_RC5_72_CORES

// Here's the same mangling for RC5-72:
//            key.hi key.mid  key.lo
// unmangled      AB:CDEFGHIJ:KLMNOPQR
// mangled        QR:OPMNKLIJ:GHEFCDAB

static void  __SwitchRC572Format(u32 *hi, u32 *mid, u32 *lo)
{
    register u32 tempkeylo = *lo;
    register u32 tempkeymid = *mid;
    register u32 tempkeyhi = *hi;

    *lo  = ((tempkeyhi)        & 0x000000FFL) |
           ((tempkeymid >> 16) & 0x0000FF00L) |
           ((tempkeymid)       & 0x00FF0000L) |
           ((tempkeymid << 16) & 0xFF000000L);
    *mid = ((tempkeymid)       & 0x000000FFL) |
           ((tempkeylo >> 16)  & 0x0000FF00L) |
           ((tempkeylo)        & 0x00FF0000L) |
           ((tempkeylo << 16)  & 0xFF000000L);
    *hi  = ((tempkeylo)        & 0x000000FFL);
}

#endif

/* ------------------------------------------------------------------- */

// Input:  - an RC5 key in 'mangled' (reversed) format or a DES key
//         - an incrementation count
//         - a contest identifier (0==RC5 1==DES 2==OGR 3==CSC)
//
// Output: the key incremented

static void __IncrementKey(u32 *keyhi, u32 *keymid, u32 *keylo, u32 iters, int contest)
{
  switch (contest)
  {
#ifdef HAVE_RC5_72_CORES
    case RC5_72:
      __SwitchRC572Format(keyhi,keymid,keylo);
      *keylo = *keylo + iters;
      if (*keylo < iters)
      {
        *keymid = *keymid + 1;
        if (*keymid == 0) *keyhi = *keyhi + 1;
      }
      __SwitchRC572Format(keyhi,keymid,keylo);
      break;
#endif
    case OGR_NG:
    case OGR_P2:
      /* This should never be called for OGR */
      break;
    default:
      PROJECT_NOT_HANDLED(contest);
      break;
  }
}

/* ------------------------------------------------------------------- */

/* generate a priv_data.contestwork for benchmarking (should be
   large enough to not complete in < 20 seconds)
*/
static int __gen_benchmark_work(unsigned int contestid, ContestWork * work)
{
  switch (contestid)
  {
    #if defined(HAVE_CRYPTO_V2)
    case RC5_72:
    {
      work->bigcrypto.key.lo = ( 0 );
      work->bigcrypto.key.mid = ( 0 );
      work->bigcrypto.key.hi = ( 0 );
      work->bigcrypto.iv.lo = ( 0 );
      work->bigcrypto.iv.hi = ( 0 );
      work->bigcrypto.plain.lo = ( 0 );
      work->bigcrypto.plain.hi = ( 0 );
      work->bigcrypto.cypher.lo = ( 0 );
      work->bigcrypto.cypher.hi = ( 0 );
      work->bigcrypto.keysdone.lo = ( 0 );
      work->bigcrypto.keysdone.hi = ( 0 );
      work->bigcrypto.iterations.lo = ( 0 );
      work->bigcrypto.iterations.hi = ( 1 );
      return contestid;
    }
    #endif
    #if defined(HAVE_OGR_PASS2)
    case OGR_P2:
    {
      //24/2-22-32-21-5-1-12
      //25/6-9-30-14-10-11
      work->ogr_p2.workstub.stub.marks = 25;    //24;
      work->ogr_p2.workstub.worklength = 6;     //7;
      work->ogr_p2.workstub.stub.length = 6;    //7;
      work->ogr_p2.workstub.stub.diffs[0] = 6;  //2;
      work->ogr_p2.workstub.stub.diffs[1] = 9;  //22;
      work->ogr_p2.workstub.stub.diffs[2] = 30;  //32;
      work->ogr_p2.workstub.stub.diffs[3] = 14; //21;
      work->ogr_p2.workstub.stub.diffs[4] = 10;  //5;
      work->ogr_p2.workstub.stub.diffs[5] = 11;  //1;
      work->ogr_p2.workstub.stub.diffs[6] = 0;  //12;
      work->ogr_p2.nodes.lo = 0;
      work->ogr_p2.nodes.hi = 0;
      work->ogr_p2.minpos = 0;
      return contestid;
    }
    #endif
    #if defined(HAVE_OGR_CORES)
    case OGR_NG:
    {
      //26/6-9-30-14-10-11
      work->ogr_ng.workstub.stub.marks = 25;
      work->ogr_ng.workstub.worklength = 6;
      work->ogr_ng.workstub.collapsed = 0;
      work->ogr_ng.workstub.stub.length = 6;
      work->ogr_ng.workstub.stub.diffs[0] = 6;
      work->ogr_ng.workstub.stub.diffs[1] = 9;
      work->ogr_ng.workstub.stub.diffs[2] = 30;
      work->ogr_ng.workstub.stub.diffs[3] = 14;
      work->ogr_ng.workstub.stub.diffs[4] = 10;
      work->ogr_ng.workstub.stub.diffs[5] = 11;
      work->ogr_ng.workstub.stub.diffs[6] = 0;
      work->ogr_ng.nodes.lo = 0;
      work->ogr_ng.nodes.hi = 0;
      return contestid;
    }
    #endif
    default:
      // PROJECT_NOT_HANDLED(contestid);
      break;
  }
  return -1;
}

/* ------------------------------------------------------------------- */

#ifdef HAVE_RC5_72_CORES
// FIXME: need code to update this value
static int rc5_72_random_subspace = 1337;
#endif

static int __gen_random_work(unsigned int contestid, ContestWork * work)
{
  // the random prefix is updated by LoadState() for every RC5 block loaded
  // that is >= 2^28 (thus excludes test blocks)
  // make one up in the event that no block was every loaded.

  u32 rnd = Random(NULL,0);

  memset((void*)work, 0, sizeof(ContestWork));

  switch (contestid)
  {
  #ifdef HAVE_RC5_72_CORES
  case RC5_72:
    /* 1*2^32 from special random subspace*/
    work->bigcrypto.key.lo  = 0;
    work->bigcrypto.key.mid = (rnd & 0x0fffffff) | ((rc5_72_random_subspace & 0x0f) << 28);
    work->bigcrypto.key.hi  = (rc5_72_random_subspace >> 4) & 0x000000ff;
    //constants are in rsadata.h
    work->bigcrypto.iv.lo     = ( RC572_IVLO );
    work->bigcrypto.iv.hi     = ( RC572_IVHI );
    work->bigcrypto.cypher.lo = ( RC572_CYPHERLO );
    work->bigcrypto.cypher.hi = ( RC572_CYPHERHI );
    work->bigcrypto.plain.lo  = ( RC572_PLAINLO );
    work->bigcrypto.plain.hi  = ( RC572_PLAINHI );
    work->bigcrypto.keysdone.lo = 0;
    work->bigcrypto.keysdone.hi = 0;
    work->bigcrypto.iterations.lo = 0;
    work->bigcrypto.iterations.hi = 1;
    work->bigcrypto.randomsubspace = 0xffff; /* invalid, randoms don't propagate random subspaces */
    work->bigcrypto.check.count = 0;
    work->bigcrypto.check.hi  = 0;
    work->bigcrypto.check.mid = 0;
    work->bigcrypto.check.lo  = 0;
    break;
  #endif
  default:
    //PROJECT_NOT_HANDLED(contestid);
    return -1;
  }
  return contestid;
}

/* ------------------------------------------------------------------- */

static unsigned int loaded_problems[CONTEST_COUNT+1] = {0};

unsigned int ProblemCountLoaded(int contestid) /* -1=all contests */
{
  if (contestid >= CONTEST_COUNT)
    return 0;
  if (contestid < 0)
    contestid = CONTEST_COUNT;
  return loaded_problems[contestid];
}

/* ======================================================================= */

int ProblemIsInitialized(void *__thisprob)
{
  int rescode = -1;
  InternalProblem *thisprob = __pick_probptr(__thisprob,PICKPROB_MAIN);
  if (thisprob)
  {
    __assert_lock(__thisprob);
    rescode = 0;
    if (thisprob->priv_data.initialized)
    {
      rescode = thisprob->priv_data.last_resultcode;
      if (rescode <= 0) /* <0 = error, 0 = RESULT_WORKING */
        rescode = -1;
      /* otherwise 1==RESULT_NOTHING, 2==RESULT_FOUND */
    }
    __release_lock(__thisprob);
  }
  return rescode;
}

/* ------------------------------------------------------------------- */

/* forward reference */
static unsigned int __compute_permille(unsigned int cont_i, const ContestWork *work);

/*
** the guts of LoadState().
**
** 'thisprob' is a scratch area to work in, and on entry
** is a copy of the main InternalProblem. On successful return
** the scratch area will be copied back to the main InternalProblem.
**
** return values:  0 OK
**                -1 error -> retry
**                -2 error -> abort
*/
static int __InternalLoadState( InternalProblem *thisprob,
                      const ContestWork * work, unsigned int contestid,
                      u32 _iterations, int expected_cputype,
                      int expected_corenum, int expected_os,
                      int expected_build )
{
  ContestWork for_magic;
  int genned_random = 0, genned_benchmark = 0;
  struct selcore selinfo; int coresel;

  //has to be done before anything else
  if (work == CONTESTWORK_MAGIC_RANDOM) /* ((const ContestWork *)0) */
  {
    memset (&for_magic, 0, sizeof(for_magic));
    if ((int)contestid != __gen_random_work(contestid, &for_magic))
      return -2; /* ouch! random generation shouldn't fail if random block
                    availability has been checked before requesting
                    CONTESTWORK_MAGIC_RANDOM */
    work = &for_magic;
    genned_random = 1;
  }
  else if (work == CONTESTWORK_MAGIC_BENCHMARK) /* ((const ContestWork *)1) */
  {
    memset (&for_magic, 0, sizeof(for_magic));
    if ((int)contestid != __gen_benchmark_work(contestid, &for_magic))
      return -2; /* ouch! benchmark generation shouldn't fail */
    work = &for_magic;
    genned_benchmark = 1;
  }
  if (thisprob->priv_data.initialized)
  {
    /* This can only happen if RetrieveState(,,purge) was not called */
    Log("BUG! LoadState() without previous RetrieveState(,,purge)!\n");
    return -1;
  }
  if (!IsProblemLoadPermitted(thisprob->priv_data.threadindex, contestid))
  {
    return -1;
  }
  coresel = selcoreSelectCore( contestid, thisprob->priv_data.threadindex, 0, &selinfo );
  if (coresel < 0)
  {
    return -2; // abort - LoadState may loop forever
  }
  if (contestid == RC5_72 && (MINIMUM_ITERATIONS % selinfo.pipeline_count) != 0)
  {
    LogScreen("(MINIMUM_ITERATIONS %% thisprob->pub_data.pipeline_count) != 0)\n");
    return -2; // abort - LoadState may loop forever
  }
  /* +++++ no point of failure beyond here (once thisprob is changed) +++++ */

  thisprob->priv_data.last_resultcode = -1;
  thisprob->priv_data.started = thisprob->priv_data.initialized = 0;
  thisprob->priv_data.loadtime_sec = thisprob->priv_data.loadtime_usec = 0;
  thisprob->pub_data.elapsed_time_sec = thisprob->pub_data.elapsed_time_usec = 0;
  thisprob->pub_data.runtime_sec = thisprob->pub_data.runtime_usec = 0;
  thisprob->pub_data.last_runtime_sec = thisprob->pub_data.last_runtime_usec = 0;
  thisprob->pub_data.last_runtime_is_invalid = 1;
  memset((void *)&thisprob->pub_data.live_rate[0], 0, sizeof(thisprob->pub_data.live_rate));
  memset((void *)&thisprob->pub_data.profiling, 0, sizeof(thisprob->pub_data.profiling));
  thisprob->pub_data.startpermille = 0;
  thisprob->pub_data.startkeys.lo = 0;
  thisprob->pub_data.startkeys.hi = 0;
  thisprob->pub_data.loaderflags = 0;
  thisprob->pub_data.contest = contestid;
  thisprob->pub_data.tslice = _iterations;
  thisprob->pub_data.tslice_increment_hint = 0;
  thisprob->pub_data.was_reset = 0;
  thisprob->pub_data.was_truncated = 0;
  thisprob->pub_data.is_random = genned_random;
  thisprob->pub_data.is_benchmark = genned_benchmark;

  thisprob->pub_data.coresel = coresel;
  thisprob->pub_data.client_cpu = selinfo.client_cpu;
  thisprob->pub_data.pipeline_count = selinfo.pipeline_count;
  thisprob->pub_data.use_generic_proto = selinfo.use_generic_proto;
  thisprob->pub_data.cruncher_is_asynchronous = selinfo.cruncher_is_asynchronous;
  memcpy( (void *)&(thisprob->pub_data.unit_func),
          &selinfo.unit_func, sizeof(thisprob->pub_data.unit_func));

  thisprob->pub_data.cruncher_is_time_constrained = 0;
  if (!genned_benchmark)
  {
    /* may also be overridden in go_mt */
    #if (CLIENT_OS == OS_RISCOS)
    if (riscos_check_taskwindow() && thisprob->pub_data.client_cpu != CPU_X86)
      thisprob->pub_data.cruncher_is_time_constrained = 1;
    #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
    if (winGetVersion() < 400)
      thisprob->pub_data.cruncher_is_time_constrained = 1;
    #elif (CLIENT_OS == OS_NETWARE)
      thisprob->pub_data.cruncher_is_time_constrained = 1;
    #endif
  }

  //----------------------------------------------------------------

  switch (thisprob->pub_data.contest)
  {
  #if defined(HAVE_CRYPTO_V2)
  case RC5_72:
    {
      if (!thisprob->pub_data.is_random && (work->bigcrypto.randomsubspace < 0x1000))
      {
        rc5_72_random_subspace = work->bigcrypto.randomsubspace;
        // FIXME: permanently store this in client->... and dnetc.ini
      }

      // copy over the state information
      thisprob->priv_data.contestwork.bigcrypto.key.hi = ( work->bigcrypto.key.hi );
      thisprob->priv_data.contestwork.bigcrypto.key.mid = ( work->bigcrypto.key.mid );
      thisprob->priv_data.contestwork.bigcrypto.key.lo = ( work->bigcrypto.key.lo );
      thisprob->priv_data.contestwork.bigcrypto.iv.hi = ( work->bigcrypto.iv.hi );
      thisprob->priv_data.contestwork.bigcrypto.iv.lo = ( work->bigcrypto.iv.lo );
      thisprob->priv_data.contestwork.bigcrypto.plain.hi = ( work->bigcrypto.plain.hi );
      thisprob->priv_data.contestwork.bigcrypto.plain.lo = ( work->bigcrypto.plain.lo );
      thisprob->priv_data.contestwork.bigcrypto.cypher.hi = ( work->bigcrypto.cypher.hi );
      thisprob->priv_data.contestwork.bigcrypto.cypher.lo = ( work->bigcrypto.cypher.lo );
      thisprob->priv_data.contestwork.bigcrypto.keysdone.hi = ( work->bigcrypto.keysdone.hi );
      thisprob->priv_data.contestwork.bigcrypto.keysdone.lo = ( work->bigcrypto.keysdone.lo );
      thisprob->priv_data.contestwork.bigcrypto.iterations.hi = ( work->bigcrypto.iterations.hi );
      thisprob->priv_data.contestwork.bigcrypto.iterations.lo = ( work->bigcrypto.iterations.lo );
      thisprob->priv_data.contestwork.bigcrypto.randomsubspace = ( work->bigcrypto.randomsubspace );
      thisprob->priv_data.contestwork.bigcrypto.check.count = ( work->bigcrypto.check.count );
      thisprob->priv_data.contestwork.bigcrypto.check.hi = ( work->bigcrypto.check.hi );
      thisprob->priv_data.contestwork.bigcrypto.check.mid = ( work->bigcrypto.check.mid );
      thisprob->priv_data.contestwork.bigcrypto.check.lo = ( work->bigcrypto.check.lo );

      if (thisprob->priv_data.contestwork.bigcrypto.keysdone.lo || thisprob->priv_data.contestwork.bigcrypto.keysdone.hi)
        {
          if (thisprob->pub_data.client_cpu != expected_cputype || thisprob->pub_data.coresel != expected_corenum ||
              CLIENT_OS != expected_os || CLIENT_VERSION != expected_build)
            {
              thisprob->priv_data.contestwork.bigcrypto.keysdone.hi = 0;
              thisprob->priv_data.contestwork.bigcrypto.keysdone.lo = 0;
              thisprob->priv_data.contestwork.bigcrypto.check.count = 0;
              thisprob->priv_data.contestwork.bigcrypto.check.hi    = 0;
              thisprob->priv_data.contestwork.bigcrypto.check.mid   = 0;
              thisprob->priv_data.contestwork.bigcrypto.check.lo    = 0;
              thisprob->pub_data.was_reset = 1;
            }
        }

      thisprob->priv_data.rc5_72unitwork.L0.hi  = thisprob->priv_data.contestwork.bigcrypto.key.hi;
      thisprob->priv_data.rc5_72unitwork.L0.mid =
          thisprob->priv_data.contestwork.bigcrypto.key.mid +
          thisprob->priv_data.contestwork.bigcrypto.keysdone.hi;
      if (thisprob->priv_data.rc5_72unitwork.L0.mid < thisprob->priv_data.contestwork.bigcrypto.keysdone.hi)
        ++thisprob->priv_data.rc5_72unitwork.L0.hi;
      thisprob->priv_data.rc5_72unitwork.L0.lo =
          thisprob->priv_data.contestwork.bigcrypto.key.lo +
          thisprob->priv_data.contestwork.bigcrypto.keysdone.lo;
      if (thisprob->priv_data.rc5_72unitwork.L0.lo < thisprob->priv_data.contestwork.bigcrypto.keysdone.lo)
      {
        ++thisprob->priv_data.rc5_72unitwork.L0.mid;
        if (thisprob->priv_data.rc5_72unitwork.L0.mid == 0)
          ++thisprob->priv_data.rc5_72unitwork.L0.hi;
      }
      thisprob->priv_data.rc5_72unitwork.L0.hi &= 0xff;
      __SwitchRC572Format(&(thisprob->priv_data.rc5_72unitwork.L0.hi), &(thisprob->priv_data.rc5_72unitwork.L0.mid), &(thisprob->priv_data.rc5_72unitwork.L0.lo));
      thisprob->priv_data.refL0.lo  = thisprob->priv_data.rc5_72unitwork.L0.lo;
      thisprob->priv_data.refL0.mid = thisprob->priv_data.rc5_72unitwork.L0.mid;
      thisprob->priv_data.refL0.hi  = thisprob->priv_data.rc5_72unitwork.L0.hi;
      thisprob->priv_data.rc5_72unitwork.plain.hi = thisprob->priv_data.contestwork.bigcrypto.plain.hi ^ thisprob->priv_data.contestwork.bigcrypto.iv.hi;
      thisprob->priv_data.rc5_72unitwork.plain.lo = thisprob->priv_data.contestwork.bigcrypto.plain.lo ^ thisprob->priv_data.contestwork.bigcrypto.iv.lo;
      thisprob->priv_data.rc5_72unitwork.cypher.hi = thisprob->priv_data.contestwork.bigcrypto.cypher.hi;
      thisprob->priv_data.rc5_72unitwork.cypher.lo = thisprob->priv_data.contestwork.bigcrypto.cypher.lo;
      thisprob->priv_data.rc5_72unitwork.check.count =
          thisprob->priv_data.contestwork.bigcrypto.check.count;
      thisprob->priv_data.rc5_72unitwork.check.hi =
          thisprob->priv_data.contestwork.bigcrypto.check.hi;
      thisprob->priv_data.rc5_72unitwork.check.mid =
          thisprob->priv_data.contestwork.bigcrypto.check.mid;
      thisprob->priv_data.rc5_72unitwork.check.lo =
          thisprob->priv_data.contestwork.bigcrypto.check.lo;
      __SwitchRC572Format(&(thisprob->priv_data.rc5_72unitwork.check.hi),
                          &(thisprob->priv_data.rc5_72unitwork.check.mid),
                          &(thisprob->priv_data.rc5_72unitwork.check.lo));

      thisprob->pub_data.startkeys.hi = thisprob->priv_data.contestwork.bigcrypto.keysdone.hi;
      thisprob->pub_data.startkeys.lo = thisprob->priv_data.contestwork.bigcrypto.keysdone.lo;
      thisprob->pub_data.startpermille = __compute_permille( thisprob->pub_data.contest, &thisprob->priv_data.contestwork );

      #if (CLIENT_CPU == CPU_CUDA)
      thisprob->priv_data.rc5_72unitwork.optimal_timeslice_increment = 0;
      thisprob->priv_data.rc5_72unitwork.best_time = -1;
      #endif

      break;
    }
  #endif

  #if defined(HAVE_OGR_PASS2)
  case OGR_P2:
  {
    int r;
    thisprob->priv_data.contestwork.ogr_p2 = work->ogr_p2;
    if (thisprob->priv_data.contestwork.ogr_p2.nodes.hi != 0 || thisprob->priv_data.contestwork.ogr_p2.nodes.lo != 0)
    {
      if (thisprob->pub_data.client_cpu != expected_cputype || thisprob->pub_data.coresel != expected_corenum ||
          CLIENT_OS != expected_os || CLIENT_VERSION != expected_build)
      {
        thisprob->pub_data.was_reset = 1;
        thisprob->priv_data.contestwork.ogr_p2.workstub.worklength = thisprob->priv_data.contestwork.ogr_p2.workstub.stub.length;
        thisprob->priv_data.contestwork.ogr_p2.nodes.hi = thisprob->priv_data.contestwork.ogr_p2.nodes.lo = 0;
      }
    }

    r = (thisprob->pub_data.unit_func.ogr)->init();
    if (r == CORE_S_OK)
    {
      r = (thisprob->pub_data.unit_func.ogr)->create(&thisprob->priv_data.contestwork.ogr_p2.workstub,
                      sizeof(WorkStub), thisprob->priv_data.core_membuffer, MAX_MEM_REQUIRED_BY_CORE,
                      thisprob->priv_data.contestwork.ogr_p2.minpos);
    }
    if (r != CORE_S_OK)
    {
      /* if it got here, then the stub is truly bad or init failed and
      ** it is ok to discard the stub (and let the network recycle it)
      */
      const char *contname = CliGetContestNameFromID(thisprob->pub_data.contest);
      const char *msg = ogr_errormsg(r);
      Log("%s load failure: %s\nStub discarded.\n", contname, msg );
      return -1;
    }
    if (thisprob->priv_data.contestwork.ogr_p2.workstub.worklength > (u32)thisprob->priv_data.contestwork.ogr_p2.workstub.stub.length)
    {
      thisprob->pub_data.startkeys.hi = thisprob->priv_data.contestwork.ogr_p2.nodes.hi;
      thisprob->pub_data.startkeys.lo = thisprob->priv_data.contestwork.ogr_p2.nodes.lo;
      thisprob->pub_data.startpermille = __compute_permille( thisprob->pub_data.contest, &thisprob->priv_data.contestwork );
    }
    break;
  }
  #endif
  #if defined(HAVE_OGR_CORES)
  case OGR_NG:
  {
    int r;
    thisprob->priv_data.contestwork.ogr_ng = work->ogr_ng;
    if (thisprob->priv_data.contestwork.ogr_ng.nodes.hi != 0 || thisprob->priv_data.contestwork.ogr_ng.nodes.lo != 0)
    {
      if (thisprob->pub_data.client_cpu != expected_cputype || thisprob->pub_data.coresel != expected_corenum ||
          CLIENT_OS != expected_os || CLIENT_VERSION != expected_build)
      {
        thisprob->pub_data.was_reset = 1;
        thisprob->priv_data.contestwork.ogr_ng.workstub.worklength = thisprob->priv_data.contestwork.ogr_ng.workstub.stub.length;
        thisprob->priv_data.contestwork.ogr_ng.nodes.hi = thisprob->priv_data.contestwork.ogr_ng.nodes.lo = 0;
      }
    }

    r = (thisprob->pub_data.unit_func.ogr)->init();
    if (r == CORE_S_OK)
    {
      r = (thisprob->pub_data.unit_func.ogr)->create(&thisprob->priv_data.contestwork.ogr_ng.workstub,
                      sizeof(OgrWorkStub), thisprob->priv_data.core_membuffer, MAX_MEM_REQUIRED_BY_CORE,
                      thisprob->priv_data.contestwork.ogr_ng.workstub.collapsed);
    }
    if (r != CORE_S_OK)
    {
      /* if it got here, then the stub is truly bad or init failed and
      ** it is ok to discard the stub (and let the network recycle it)
      */
      const char *contname = CliGetContestNameFromID(thisprob->pub_data.contest);
      const char *msg = ogr_errormsg(r);
      Log("%s load failure: %s\nStub discarded.\n", contname, msg );
      return -1;
    }
    if (thisprob->priv_data.contestwork.ogr_ng.workstub.worklength > (u32)thisprob->priv_data.contestwork.ogr_ng.workstub.stub.length)
    {
      thisprob->pub_data.startkeys.hi = thisprob->priv_data.contestwork.ogr_ng.nodes.hi;
      thisprob->pub_data.startkeys.lo = thisprob->priv_data.contestwork.ogr_ng.nodes.lo;
      thisprob->pub_data.startpermille = __compute_permille( thisprob->pub_data.contest, &thisprob->priv_data.contestwork );
    }
    break;
  }
  #endif

  default:
    PROJECT_NOT_HANDLED(thisprob->pub_data.contest);
  }

  //---------------------------------------------------------------

  {
    // set timers
    struct timeval tv;
    thisprob->priv_data.loadtime_sec = 0;
    if (CliGetMonotonicClock(&tv) != 0)
    {
      if (CliGetMonotonicClock(&tv) != 0)
        thisprob->priv_data.loadtime_sec = 0xfffffffful;
    }
    if (thisprob->priv_data.loadtime_sec == 0)
    {
      thisprob->priv_data.loadtime_sec = tv.tv_sec;
      thisprob->priv_data.loadtime_usec = tv.tv_usec;
    }
    thisprob->pub_data.elapsed_time_sec = 0xfffffffful; // invalid while RESULT_WORKING
  }

  loaded_problems[thisprob->pub_data.contest]++;       /* per contest */
  loaded_problems[CONTEST_COUNT]++; /* total */
  thisprob->priv_data.last_resultcode = RESULT_WORKING;
  thisprob->priv_data.initialized = 1;

  return( 0 );
}

/* LoadState() and RetrieveState() work in pairs. A LoadState() without
   a previous RetrieveState(,,purge) will fail, and vice-versa.

   return values:  0 OK
                  -1 error -> retry
                  -2 error -> abort
*/
int ProblemLoadState( void *__thisprob,
                      const ContestWork * work, unsigned int contestid,
                      u32 _iterations, int expected_cputype,
                      int expected_corenum, int expected_os,
                      int expected_build )
{
  InternalProblem *temp_prob = __pick_probptr(__thisprob, PICKPROB_TEMP);
  InternalProblem *main_prob = __pick_probptr(__thisprob, PICKPROB_MAIN);
  int res = -1;

  if (!temp_prob || !main_prob)
  {
    return -1;
  }

  __assert_lock(__thisprob);
  __copy_internal_problem( temp_prob, main_prob ); /* copy main->temp */
  __release_lock(__thisprob);

  res = __InternalLoadState( temp_prob, work, contestid, _iterations,
                             expected_cputype, expected_corenum, expected_os,
                             expected_build );
  if (res != 0)
  {
    return (res<0)?(res):(-1);
  }

  /* success */
  __assert_lock(__thisprob);
  __copy_internal_problem( main_prob, temp_prob ); /* copy temp->main */
  __release_lock(__thisprob);
  return 0;
}

/* ------------------------------------------------------------------- */

/* LoadState() and RetrieveState() work in pairs. A LoadState() without
   a previous RetrieveState(,,purge) will fail, and vice-versa.
*/

int ProblemRetrieveState( void *__thisprob,
                          ContestWork * work, unsigned int *contestid,
                          int dopurge, int dontwait )
{
  int ret_code = 0;
  InternalProblem *thisprob = __pick_probptr(__thisprob, PICKPROB_MAIN);
  if (!thisprob)
  {
    return -1;
  }
  __assert_lock(__thisprob);

  if (!thisprob->priv_data.initialized)
  {
    //LogScreen("ProblemRetrieveState() without preceding LoadState()\n");
    ret_code = -1;
  }
  else
  {
    if (work) // store back the state information
    {
      switch (thisprob->pub_data.contest)
      {
        case RC5_72:
        {
          memcpy( (void *)work,
                  (void *)&thisprob->priv_data.contestwork,
                  sizeof(ContestWork));
          break;
        }
        #if defined(HAVE_OGR_PASS2)
        case OGR_P2:
        {
          (thisprob->pub_data.unit_func.ogr)->getresult(
                       thisprob->priv_data.core_membuffer,
                       &thisprob->priv_data.contestwork.ogr_p2.workstub,
                       sizeof(WorkStub));
          memcpy( (void *)work,
                  (void *)&thisprob->priv_data.contestwork,
                  sizeof(ContestWork));
          break;
        }
        #endif
        #if defined(HAVE_OGR_CORES)
        case OGR_NG:
        {
          (thisprob->pub_data.unit_func.ogr)->getresult(
                       thisprob->priv_data.core_membuffer,
                       &thisprob->priv_data.contestwork.ogr_ng.workstub,
                       sizeof(OgrWorkStub));
          memcpy( (void *)work,
                  (void *)&thisprob->priv_data.contestwork,
                  sizeof(ContestWork));
          break;
        }
        #endif
        default: /* cannot happen */
        {
          PROJECT_NOT_HANDLED(thisprob->pub_data.contest);
          break;
        }
      } /* switch */
    }
    if (contestid)
      *contestid = thisprob->pub_data.contest;
    if (dopurge)
    {
      thisprob->priv_data.initialized = 0;
      loaded_problems[thisprob->pub_data.contest]--;       /* per contest */
      loaded_problems[CONTEST_COUNT]--; /* total */
    }
    ret_code = thisprob->priv_data.last_resultcode;
    if (ret_code < 0)
    {
      //LogScreen("last resultcode = %d\n",ret_code);
      ret_code = -1;
    }
  }
  __release_lock(__thisprob);
  dontwait = dontwait; /* no longer neccesary since we have full locks now */
  return ret_code;
}

/* ------------------------------------------------------------- */

static int Run_OGR_P2( InternalProblem *thisprob, /* already validated */
                    u32 *iterationsP, int *resultcode)
{
#if !defined(HAVE_OGR_PASS2)
  thisprob = thisprob;
  iterationsP = iterationsP;
#else
  int r, nodes;

  nodes = (int)(*iterationsP);
  r = (thisprob->pub_data.unit_func.ogr)->cycle(
                          thisprob->priv_data.core_membuffer,
                          &nodes,
                          thisprob->pub_data.cruncher_is_time_constrained);
  /*
   * We'll calculate and return true number of core iterations for timesling.
   * This number may be NOT equal to number of actually processed OGR nodes,
   * which is returned in 'nodes'. See ogr.cpp for details about node caching.
   *
   * Following code based on kakace' rules:
   *   a) if cruncher_is_time_constrained is true, core MUST exit after
   *      processing exactly requested number of iterations; returned number
   *      of nodes is ignored due to node caching.
   *   b) if cruncher_is_time_constrained is false, core MUST NOT cache nodes
   *      and value in 'nodes' MUST be true number of iterations done.
   * If core failed to follow these rules due to coding errors or optimization,
   * things may became unpredictable.
   */
  if (!thisprob->pub_data.cruncher_is_time_constrained)
    *iterationsP = (u32)nodes; /* with t.c., iter. count not changed */

  u32 newnodeslo = thisprob->priv_data.contestwork.ogr_p2.nodes.lo + nodes;
  if (newnodeslo < thisprob->priv_data.contestwork.ogr_p2.nodes.lo) {
    thisprob->priv_data.contestwork.ogr_p2.nodes.hi++;
  }
  thisprob->priv_data.contestwork.ogr_p2.nodes.lo = newnodeslo;

  switch (r)
  {
    case CORE_S_OK:
    {
      r = (thisprob->pub_data.unit_func.ogr)->destroy(thisprob->priv_data.core_membuffer);
      if (r == CORE_S_OK)
      {
        *resultcode = RESULT_NOTHING;
        return RESULT_NOTHING;
      }
      break;
    }
    case CORE_S_CONTINUE:
    {
      *resultcode = RESULT_WORKING;
      return RESULT_WORKING;
    }
    case CORE_S_SUCCESS:
    {
      r = (thisprob->pub_data.unit_func.ogr)->getresult(thisprob->priv_data.core_membuffer,
                              &thisprob->priv_data.contestwork.ogr_p2.workstub, sizeof(WorkStub));
      if (r == CORE_S_OK)
      {
        //Log("OGR-P2 Success!\n");
        *resultcode = RESULT_FOUND;
        return RESULT_FOUND;
      }
      break;
    }
  }
  /* Something bad happened */
  if (r < 0) {
    const char *contname = CliGetContestNameFromID(thisprob->pub_data.contest);
    const char* msg = ogr_errormsg(r);
    Log("%s load failure: %s\nStub discarded.\n", contname, msg );
  }
#endif
 *resultcode = -1; /* this will cause the problem to be discarded */
 return -1;
}

/* ------------------------------------------------------------- */

static int Run_OGR_NG( InternalProblem *thisprob, /* already validated */
                    u32 *iterationsP, int *resultcode)
{
#if !defined(HAVE_OGR_CORES)
  thisprob = thisprob;
  iterationsP = iterationsP;
#else
  int r, nodes;

  nodes = (int)(*iterationsP);

  SAVE_CLIENT_OS_CONTEXT

  r = (thisprob->pub_data.unit_func.ogr)->cycle(
                          thisprob->priv_data.core_membuffer,
                          &nodes, 0);

  RESTORE_CLIENT_OS_CONTEXT

  /*
   * We'll calculate and return true number of core iterations for timesling.
   * This number may be NOT equal to number of actually processed OGR nodes,
   * which is returned in 'nodes'. See ogr.cpp for details about node caching.
   *
   * Following code based on kakace' rules:
   *   a) if cruncher_is_time_constrained is true, core MUST exit after
   *      processing exactly requested number of iterations; returned number
   *      of nodes is ignored due to node caching.
   *   b) if cruncher_is_time_constrained is false, core MUST NOT cache nodes
   *      and value in 'nodes' MUST be true number of iterations done.
   * If core failed to follow these rules due to coding errors or optimization,
   * things may became unpredictable.
   */
  if (!thisprob->pub_data.cruncher_is_time_constrained)
    *iterationsP = (u32)nodes; /* with t.c., iter. count not changed */

  u32 newnodeslo = thisprob->priv_data.contestwork.ogr_ng.nodes.lo + nodes;
  if (newnodeslo < thisprob->priv_data.contestwork.ogr_ng.nodes.lo) {
    thisprob->priv_data.contestwork.ogr_ng.nodes.hi++;
  }
  thisprob->priv_data.contestwork.ogr_ng.nodes.lo = newnodeslo;

  switch (r)
  {
    case CORE_S_OK:
    {
      r = (thisprob->pub_data.unit_func.ogr)->destroy(thisprob->priv_data.core_membuffer);
      if (r == CORE_S_OK)
      {
        *resultcode = RESULT_NOTHING;
        return RESULT_NOTHING;
      }
      break;
    }
    case CORE_S_CONTINUE:
    {
      *resultcode = RESULT_WORKING;
      return RESULT_WORKING;
    }
    case CORE_S_SUCCESS:
    {
      r = (thisprob->pub_data.unit_func.ogr)->getresult(thisprob->priv_data.core_membuffer,
                              &thisprob->priv_data.contestwork.ogr_ng.workstub, sizeof(OgrWorkStub));
      if (r == CORE_S_OK)
      {
        *resultcode = RESULT_FOUND;
        return RESULT_FOUND;
      }
      break;
    }
  }
  /* Something bad happened */
  if (r < 0) {
    const char *contname = CliGetContestNameFromID(thisprob->pub_data.contest);
    const char* msg = ogr_errormsg(r);
    Log("%s load failure: %s\nStub discarded.\n", contname, msg );
  }
#endif
 *resultcode = -1; /* this will cause the problem to be discarded */
 return -1;
}

/* ------------------------------------------------------------- */

static int Run_RC5_72(InternalProblem *thisprob, /* already validated */
                   u32 *keyscheckedP /* count of ... */, int *resultcode)
{
#ifndef HAVE_RC5_72_CORES
  thisprob = thisprob;
  *keyscheckedP = 0;
  *resultcode = -1;
  thisprob->priv_data.last_resultcode = -1;
  return -1;
#else
  s32 rescode = -1;

  /* a brace to ensure 'keystocheck' is not referenced in the common part */
  {
    u32 keystocheck = *keyscheckedP;

    // don't allow a too large of a keystocheck be used ie (>(iter-keysdone))
    // (technically not necessary, but may save some wasted time)
    if ((thisprob->priv_data.contestwork.bigcrypto.iterations.hi -
         thisprob->priv_data.contestwork.bigcrypto.keysdone.hi <= 1) &&
        (thisprob->priv_data.contestwork.bigcrypto.keysdone.lo +
         keystocheck < keystocheck))
    {
      keystocheck = thisprob->priv_data.contestwork.bigcrypto.iterations.lo -
                    thisprob->priv_data.contestwork.bigcrypto.keysdone.lo;
    }

    if (keystocheck < MINIMUM_ITERATIONS)
      keystocheck = MINIMUM_ITERATIONS;
    else if ((keystocheck % MINIMUM_ITERATIONS) != 0)
      keystocheck += (MINIMUM_ITERATIONS - (keystocheck % MINIMUM_ITERATIONS));

    #if 0
    LogScreen("align iterations: effective iterations: %lu (0x%lx),\n"
              "suggested iterations: %lu (0x%lx)\n"
              "thisprob->pub_data.pipeline_count = %lu, iterations%%thisprob->pub_data.pipeline_count = %lu\n",
              (unsigned long)keystocheck, (unsigned long)keystocheck,
              (unsigned long)(*keyscheckedP), (unsigned long)(*keyscheckedP),
              thisprob->pub_data.pipeline_count, keystocheck%thisprob->pub_data.pipeline_count );
    #endif

    if (thisprob->pub_data.use_generic_proto)
    {
      //we don't care about thisprob->pub_data.pipeline_count when using unified cores.
      //we _do_ need to care that the keystocheck and starting key are aligned.

      *keyscheckedP = keystocheck; /* Pass 'keystocheck', get back 'keyschecked'*/

#if (CLIENT_CPU == CPU_CUDA) || (CLIENT_CPU == CPU_AMD_STREAM)
      thisprob->priv_data.rc5_72unitwork.threadnum = thisprob->pub_data.threadnum;
#endif     
      SAVE_CLIENT_OS_CONTEXT

      rescode = (*(thisprob->pub_data.unit_func.gen_72))(&thisprob->priv_data.rc5_72unitwork,keyscheckedP,thisprob->priv_data.core_membuffer);

      RESTORE_CLIENT_OS_CONTEXT

      if (rescode >= 0 && thisprob->pub_data.cruncher_is_asynchronous) /* co-processor or similar */
      {
        keystocheck = *keyscheckedP; /* always so */
        /* how this works:
         - for RESULT_FOUND, we don't need to do anything, since keyscheckedP
           has the real count of iters done. If we were still using old style
           method of determining RESULT_FOUND by (keyscheckedP < keystocheck),
           then we would simply need to set 'keystocheck = 1 + *keyscheckedP'
           to make it greater.
         - for RESULT_NOTHING/RESULT_WORKING
           unlike normal cores, where RESULT_NOTHING and RESULT_WORKING
           are synonymous (RESULT_NOTHING from the core's perspective ==
           RESULT_WORKING from the client's perspective), async cores tell
           us which-is-which through the keyscheckedP pointer. As long as
           they are _WORKING, the *keyscheckedP (ie iterations_done) will be
           zero. (And of course, incrementations checks will pass as long
           as iterations_done is zero :).
        */
        //(these next 3 lines are quite useless, since the actual state
        //is set lower down, but leave them in anyway to show how it works)
        //remember: keystocheck has already been set equal to *keyscheckedP
        if (rescode != RESULT_FOUND) /* RESULT_NOTHING/RESULT_WORKING */
        {
          rescode = *resultcode = RESULT_NOTHING; //assume we know something
          if (*keyscheckedP == 0)  /* still working */
            rescode = *resultcode = RESULT_WORKING;
        }
      }
    }
    else /* old style */
    {
      /* do not write old style cores for rc5-72 ! */
      rescode = -1;
    }
  } /* brace to ensure that 'keystocheck' is not referenced beyond here */
  /* -- the code from here on down is identical to that of CSC -- */

  if (rescode < 0) /* "kiter" error */
  {
    *resultcode = -1;
    thisprob->priv_data.last_resultcode = -1;
    return -1;
  }
  *resultcode = (int)rescode;

  // Increment reference key count
  __IncrementKey(&thisprob->priv_data.refL0.hi, &thisprob->priv_data.refL0.mid, &thisprob->priv_data.refL0.lo, *keyscheckedP, thisprob->pub_data.contest);

  // Compare ref to core key incrementation
  if (((thisprob->priv_data.refL0.hi  != thisprob->priv_data.rc5_72unitwork.L0.hi)  ||
       (thisprob->priv_data.refL0.mid != thisprob->priv_data.rc5_72unitwork.L0.mid) ||
       (thisprob->priv_data.refL0.lo  != thisprob->priv_data.rc5_72unitwork.L0.lo))
      && (*resultcode != RESULT_FOUND) )
  {
    if (thisprob->priv_data.contestwork.bigcrypto.iterations.hi == 0 &&
        thisprob->priv_data.contestwork.bigcrypto.iterations.lo == 0x20000) /* test case */
    {
      Log("RC5-72 incrementation mismatch:\n"
          "Debug Information: %02x:%08x:%08x - %02x:%08x:%08x\n",
          thisprob->priv_data.rc5_72unitwork.L0.hi, thisprob->priv_data.rc5_72unitwork.L0.mid, thisprob->priv_data.rc5_72unitwork.L0.lo,
          thisprob->priv_data.refL0.hi, thisprob->priv_data.refL0.mid, thisprob->priv_data.refL0.lo);
    }
    *resultcode = -1;
    thisprob->priv_data.last_resultcode = -1;
    return -1;
  };

  // Checks passed, increment keys done count.
  thisprob->priv_data.contestwork.bigcrypto.keysdone.lo += *keyscheckedP;
  if (thisprob->priv_data.contestwork.bigcrypto.keysdone.lo < *keyscheckedP)
      thisprob->priv_data.contestwork.bigcrypto.keysdone.hi++;

  // update counter measure checks (core stores mangled key)
  thisprob->priv_data.contestwork.bigcrypto.check.count =
      thisprob->priv_data.rc5_72unitwork.check.count;
  thisprob->priv_data.contestwork.bigcrypto.check.hi =
      thisprob->priv_data.rc5_72unitwork.check.hi;
  thisprob->priv_data.contestwork.bigcrypto.check.mid =
      thisprob->priv_data.rc5_72unitwork.check.mid;
  thisprob->priv_data.contestwork.bigcrypto.check.lo =
      thisprob->priv_data.rc5_72unitwork.check.lo;
  __SwitchRC572Format(&(thisprob->priv_data.contestwork.bigcrypto.check.hi),
                      &(thisprob->priv_data.contestwork.bigcrypto.check.mid),
                      &(thisprob->priv_data.contestwork.bigcrypto.check.lo));

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)  //(*keyscheckedP < keystocheck)
  {
    // found it!
    u32 keylo  = thisprob->priv_data.contestwork.bigcrypto.key.lo;
    u32 keymid = thisprob->priv_data.contestwork.bigcrypto.key.mid;
    thisprob->priv_data.contestwork.bigcrypto.key.lo  += thisprob->priv_data.contestwork.bigcrypto.keysdone.lo;
    if (thisprob->priv_data.contestwork.bigcrypto.key.lo < keylo)
    {
      thisprob->priv_data.contestwork.bigcrypto.key.mid++; // wrap occured ?
      keymid++;
      if (thisprob->priv_data.contestwork.bigcrypto.key.mid == 0)
        thisprob->priv_data.contestwork.bigcrypto.key.hi++;
    }
    thisprob->priv_data.contestwork.bigcrypto.key.mid += thisprob->priv_data.contestwork.bigcrypto.keysdone.hi;
    if (thisprob->priv_data.contestwork.bigcrypto.key.mid < keymid)
        thisprob->priv_data.contestwork.bigcrypto.key.hi++;

    return RESULT_FOUND;
  }

  if ( ( thisprob->priv_data.contestwork.bigcrypto.keysdone.hi > thisprob->priv_data.contestwork.bigcrypto.iterations.hi ) ||
       ( ( thisprob->priv_data.contestwork.bigcrypto.keysdone.hi == thisprob->priv_data.contestwork.bigcrypto.iterations.hi ) &&
       ( thisprob->priv_data.contestwork.bigcrypto.keysdone.lo >= thisprob->priv_data.contestwork.bigcrypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  #ifdef STRESS_THREADS_AND_BUFFERS
  if (core_prob->priv_data.contestwork.bigcrypto.key.hi  ||
      core_prob->priv_data.contestwork.bigcrypto.key.mid ||
      core_prob->priv_data.contestwork.bigcrypto.key.lo) /* not bench */
  {
    core_prob->priv_data.contestwork.bigcrypto.key.hi = 0;
    core_prob->priv_data.contestwork.bigcrypto.key.mid = 0;
    core_prob->priv_data.contestwork.bigcrypto.key.lo = 0;
    core_prob->priv_data.contestwork.bigcrypto.keysdone.hi =
      core_prob->priv_data.contestwork.bigcrypto.iterations.hi;
    core_prob->priv_data.contestwork.bigcrypto.keysdone.lo =
      core_prob->priv_data.contestwork.bigcrypto.iterations.lo;
    *resultcode = RESULT_NOTHING;
  }
  #endif

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING;    // Done with this round
#endif
}

/* ------------------------------------------------------------- */

static void __compute_run_times(InternalProblem *thisprob,
                                u32 runstart_secs, u32 runstart_usecs,
                                u32 *probstart_secs, u32 *probstart_usecs,
                                int using_ptime, volatile int *s_using_ptime,
                                int core_resultcode )
{
  struct timeval clock_stop;
  int last_runtime_is_invalid = 0;
  int clock_stop_is_time_now = 0;
  u32 timehi, timelo, elapsedhi, elapsedlo;
  clock_stop.tv_sec = 0;
  clock_stop.tv_usec = 0;

  /* ++++++++++++++++++++++++++ */

  /* first compute elapsed time for this run */
  if (runstart_secs == 0xfffffffful) /* couldn't get a start time */
  {
    last_runtime_is_invalid = 1;
  }
  else if (!using_ptime)
  {
    if (CliGetMonotonicClock(&clock_stop) != 0)
    {
      if (CliGetMonotonicClock(&clock_stop) != 0)
        last_runtime_is_invalid = 1;
    }
    if (!last_runtime_is_invalid)
    {
      /* flag to say clock_stop reflects 'now' */
      clock_stop_is_time_now = 1;
    }
  }
  else if (CliGetThreadUserTime(&clock_stop) < 0)
  {
    *s_using_ptime = 0;
    last_runtime_is_invalid = 1;
  }
  if (!last_runtime_is_invalid)
  {
    timehi = runstart_secs;
    timelo = runstart_usecs;
    elapsedhi = clock_stop.tv_sec;
    elapsedlo = clock_stop.tv_usec;

    if (elapsedhi <  timehi || (elapsedhi == timehi && elapsedlo < timelo ))
    {
      /* AIEEEE - clock is whacked */
      last_runtime_is_invalid = 1;
    }
    else
    {
      last_runtime_is_invalid = 0;

      if (elapsedlo < timelo)
      {
        elapsedhi--;
        elapsedlo += 1000000UL;
      }
      elapsedhi -= timehi;
      elapsedlo -= timelo;
      thisprob->pub_data.last_runtime_sec = elapsedhi;
      thisprob->pub_data.last_runtime_usec = elapsedlo;

      elapsedhi += thisprob->pub_data.runtime_sec;
      elapsedlo += thisprob->pub_data.runtime_usec;
      if (elapsedlo >= 1000000UL)
      {
        elapsedhi++;
        elapsedlo -= 1000000UL;
      }
      thisprob->pub_data.runtime_sec  = elapsedhi;
      thisprob->pub_data.runtime_usec = elapsedlo;
    }
  }
  if (last_runtime_is_invalid)
  {
    thisprob->pub_data.last_runtime_sec = 0;
    thisprob->pub_data.last_runtime_usec = 0;
  }
  thisprob->pub_data.last_runtime_is_invalid = last_runtime_is_invalid;

  /* ++++++++++++++++++++++++++ */

  /* do we need to compute elapsed wall clock time for this packet? */
  if ( core_resultcode == RESULT_WORKING ) /* no, not yet */
  {
    if (clock_stop_is_time_now /* we have determined 'now' */
    && *probstart_secs == 0xfffffffful) /* our start time was invalid */
    {                          /* then save 'now' as our start time */
      *probstart_secs = clock_stop.tv_sec;
      *probstart_usecs = clock_stop.tv_usec;
    }
  }
  else /* _FOUND/_NOTHING. run is finished, compute elapsed wall clock time */
  {
    timehi = *probstart_secs;
    timelo = *probstart_usecs;

    if (!clock_stop_is_time_now /* we haven't determined 'now' yet */
    && timehi != 0xfffffffful) /* our start time was not invalid */
    {
      if (CliGetMonotonicClock(&clock_stop) != 0)
      {
        if (CliGetMonotonicClock(&clock_stop) != 0)
          timehi = 0xfffffffful; /* no stop time, so make start invalid */
      }
    }
    elapsedhi = clock_stop.tv_sec;
    elapsedlo = clock_stop.tv_usec;

    if (timehi == 0xfffffffful || /* start time is invalid */
        elapsedhi <  timehi || (elapsedhi == timehi && elapsedlo < timelo ))
    {
      /* either start time is invalid, or end-time < start-time */
      /* both are BadThing(TM)s - have to use the per-run total */
      elapsedhi = thisprob->pub_data.runtime_sec;
      elapsedlo = thisprob->pub_data.runtime_usec;
    }
    else /* start and 'now' time are ok */
    {
      if (elapsedlo < timelo)
      {
        elapsedlo += 1000000UL;
        elapsedhi --;
      }
      elapsedhi -= timehi;
      elapsedlo -= timelo;
    }
    thisprob->pub_data.elapsed_time_sec  = elapsedhi;
    thisprob->pub_data.elapsed_time_usec = elapsedlo;
  }

  return;
}

/* ------------------------------------------------------------- */

int __prime_run_times( u32 *runstart_secs, u32 *runstart_usecs,
                       int using_ptime )
{
  struct timeval tv;
  int err = 0;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  if (using_ptime)
  {
    if (CliGetThreadUserTime(&tv) != 0)
      using_ptime = 0;
  }
  if (!using_ptime)
  {
    if (CliGetMonotonicClock(&tv) != 0)
    {
      if (CliGetMonotonicClock(&tv) != 0)
        err = 1;
    }
  }
  if (!err)
  {
    *runstart_secs = tv.tv_sec;
    *runstart_usecs = tv.tv_usec;
  }
  else
  {
    *runstart_secs = 0xffffffff;
    *runstart_usecs = 0;
  }
  return using_ptime;
}

/* ---------------------------------------------------------------- */

int ProblemRun(void *__thisprob) /* returns RESULT_*  or -1 */
{
  int last_resultcode;
  InternalProblem *main_prob = __pick_probptr(__thisprob, PICKPROB_MAIN);
  InternalProblem *core_prob = __pick_probptr(__thisprob, PICKPROB_CORE);

  if (!main_prob || !core_prob)
  {
    return -1;
  }
  if ( !main_prob->priv_data.initialized )
  {
    return -1;
  }

  __assert_lock(__thisprob);
  main_prob->priv_data.started = 1;
  main_prob->pub_data.last_runtime_is_invalid = 1; /* haven't changed runtime fields yet */
  main_prob->pub_data.last_runtime_usec = 0;
  main_prob->pub_data.last_runtime_sec = 0;
  __copy_internal_problem( core_prob, main_prob ); /* copy main->core */
  __release_lock(__thisprob);

  last_resultcode = core_prob->priv_data.last_resultcode;
  if ( last_resultcode == RESULT_WORKING ) /* _FOUND, _NOTHING or -1 */
  {
    static volatile int s_using_ptime = -1;
    int retcode; u32 iterations, runstart_secs, runstart_usecs;
    int using_ptime = __prime_run_times( &runstart_secs, &runstart_usecs,
                                         s_using_ptime );

    /* +++++++++++++++++ */

    /*
      On return from the Run_XXX core_prob->priv_data.contestwork must be in a
      state that we can put away to disk - that is, do not expect the loader
      (probfill et al) to fiddle with iterations or key or whatever.

      The Run_XXX functions do *not* update
      problem.core_prob->priv_data.last_resultcode, they use the local
      last_resultcode instead. This is so that members of the problem object
      that are updated after the resultcode has been set will not be out of
      sync when the main thread gets it with RetrieveState().
    */

    /* Run_XXX retcode:
    ** although the value returned by Run_XXX is usually the same as
    ** the priv_data.last_resultcode it is not always the case. For instance,
    ** if post-LoadState() initialization failed, but can be deferred, Run_XXX
    ** may choose to return -1, but keep priv_data.last_resultcode at
    ** RESULT_WORKING.
    */

    retcode         = -1;
    iterations      = core_prob->pub_data.tslice;
    switch (core_prob->pub_data.contest)
    {
      case RC5_72:
        retcode = Run_RC5_72( core_prob, &iterations, &last_resultcode );
        break;
      case OGR_P2:
        retcode = Run_OGR_P2( core_prob, &iterations, &last_resultcode );
        break;
      case OGR_NG:
        retcode = Run_OGR_NG( core_prob, &iterations, &last_resultcode );
        break;
      default:
        PROJECT_NOT_HANDLED(core_prob->pub_data.contest);
        retcode = 0;
        last_resultcode = -1;
        break;
    }

    if (retcode < 0)
    {
      /* don't touch core_prob->pub_data.tslice or runtime as long as retcode < 0!!! */
      last_resultcode = core_prob->priv_data.last_resultcode;
    }
    else
    {
      __assert_lock(__thisprob);
      if (!main_prob->priv_data.started ||  /* LoadState() clears this */
          !main_prob->priv_data.initialized) /* RetrieveState() clears this */
      {
        /* whoops! RetrieveState(,,purge) [with or without a subsequent
        ** LoadState()] was called while we in core.
        */
        last_resultcode = -1;
      }
      else /* update the remaining Run() related things, and synchronize */
      {
        /* update the core's copy of the public area. It might have
        ** changed while we in core.
        */
        memcpy( &(core_prob->pub_data), &(main_prob->pub_data),
                                         sizeof(core_prob->pub_data) );

        /*
        ** make the necessary modifications to the public area
        */
        __compute_run_times( core_prob, runstart_secs, runstart_usecs,
                             &core_prob->priv_data.loadtime_sec,
                             &core_prob->priv_data.loadtime_usec,
                             using_ptime, &s_using_ptime, last_resultcode );
        core_prob->pub_data.core_run_count++;
        core_prob->pub_data.tslice = iterations;
#if (CLIENT_CPU == CPU_CUDA)
        // FIXME there could be a better way to do this
        core_prob->pub_data.tslice_increment_hint = core_prob->priv_data.rc5_72unitwork.optimal_timeslice_increment;
#endif

        /*
        ** make the necessary modifications to the private area
        */
        core_prob->priv_data.last_resultcode = last_resultcode;

        /*
        ** now blast the whole (public AND private) core_prob into main_prob.
        */
        __copy_internal_problem( main_prob, core_prob ); /* copy core->main */
      }
      __release_lock(__thisprob);
    }
  } /* if (last_resultcode == RESULT_WORKING) */

  return last_resultcode;
}

/* ----------------------------------------------------------------------- */

// Returns 1 if it is safe to load the specified contest onto a
// specified problem slot, or 0 if it is not allowed.  Core
// thread-safety and contest availability checks are used to determine
// allowability, but not contest closure.

int IsProblemLoadPermitted(long prob_index, unsigned int contest_i)
{
  DNETC_UNUSED_PARAM(prob_index);

  #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  if (prob_index == 1 && /* thread number reserved for x86 card */
     contest_i != RC5 && /* RISC OS x86 thread only supports RC5 */
     GetNumberOfDetectedProcessors() > 1) /* have x86 card */
    return 0;
  #endif
  #if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN16)
  /* Cannot run (long-running) OGR on non-preemptive OSs on low end
     hardware. OGR has significant per-call overhead which ultimately
     prevents frequent yielding no matter how small the timeslice.
     Examples (486/66, NetWare, 3c5x9 polling NIC):
     16 nodes per call: max achievable yield rate: 13-15/sec
     Server is extremely laggy.
     256 nodes per call: max achievable yield rate ALSO 13-15/sec
     For the fun of it, I then tried 1024 nodes per call: NetWare
     started dropping packets, clients disconnected, the profiler
     froze - I couldn't switch back to the console to unload the
     client and had to power-cycle.

     But that has nothing to do with RiscOS in a taskwindow, as a
     process in a taskwindow is preemptively scheduled. At least as
     fas as I know and observed.
  */
  if (contest_i == OGR_NG || contest_i == OGR_P2) /* crunchers only */
  {
    #if (CLIENT_CPU == CPU_68K)
    return 0;
    #elif (CLIENT_CPU == CPU_ARM)
    if (riscos_check_taskwindow())
      return 0;
    #else
    static int should_not_do = -1;
    if (should_not_do == -1)
    {
      long det = GetProcessorType(1);
      if (det >= 0)
      {
        switch (det & 0xff)
        {
          #if (CLIENT_CPU == CPU_X86)
          case 0x00:  // P5
          case 0x01:  // 386/486
          case 0x03:  // Cx6x86
          case 0x04:  // K5
          case 0x06:  // Cyrix 486
          case 0x0A:  // Centaur C6
          #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
          case 0x01:  // PPC 601
          #endif
                    should_not_do = +1;
                    break;
          default:  should_not_do = 0;
                    break;
        }
      }
    }
    if (should_not_do)
      return 0;
    #endif
  }
  #endif


  switch (contest_i)
  {
    case RC5_72:
    {
      #ifdef HAVE_RC5_72_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case OGR_P2:
    {
      #if defined (HAVE_OGR_PASS2)
      return 1;
      #else
      return 0;
      #endif
    }
    case OGR_NG:
    {
      #if defined (HAVE_OGR_CORES)
      return 1;
      #else
      return 0;
      #endif
    }
    default:
    {
      // PROJECT_NOT_HANDLED(contest_i);
      break;
    }
  }
  return 0;
}

/* ----------------------------------------------------------------------- */
/* support functions for ProblemGetInfo()                         */
/* ----------------------------------------------------------------------- */

static void __u64mul( u32 _ahi, u32 _alo, u32 _bhi, u32 _blo,
                      u32 *reshi, u32 *reslo )
{
  /* when modifying this keep in mind that input parameters and result
     may overlap
  */
  if (reshi || reslo)
  {
    #if (ULONG_MAX > 0xfffffffful) /* 64+ bits */
    unsigned long d,r;
    d = (((unsigned long)_ahi)<<32UL)+((unsigned long)_alo);
    r = (((unsigned long)_bhi)<<32UL)+((unsigned long)_blo);
    r *= d;
    if (reshi) *reshi = (u32)(r >> 32);
    if (reslo) *reslo = (u32)(r & 0xfffffffful);
    #elif 0 /* 32bit - using mul+polinomial add (Karatsiba/Knuth) */
    u32 rhi, rlo, ahi = _ahi, alo = _alo, bhi = _bhi, blo = _blo;
    rlo = ((alo >> 16) * (blo & 0xffff)) +
          (((alo & 0xffff) * (blo & 0xffff)) >> 16);
    rhi = (ahi * blo) + (alo * bhi) + ((rlo >> 16) + ((alo >> 16) * (blo >> 16))
          + (((rlo & 0xffff) + ((blo >> 16) * (alo & 0xffff))) >> 16));
    rlo = (alo * blo);
    if (reshi) *reshi = rhi;
    if (reslo) *reslo = rlo;
    #else /* 32bit - long multiplication using shift+add */
    u32 rhi = 0, rlo = 0, ahi = _ahi, alo = _alo, bhi = _bhi, blo = _blo;
    TRACE_U64OPS((+1,"__u64mul(%u:%u, %u:%u)\n",ahi,alo,bhi,blo));
    while (bhi || blo)
    {
      if ((blo & 1) != 0)
      {
        u32 lo = rlo + alo;
        if (lo < rlo) rhi++;
        rhi += ahi;
        rlo = lo;
      }
      ahi <<= 1; ahi |= (alo >> 31); alo <<= 1;
      blo >>= 1; blo |= (bhi << 31); bhi >>= 1;
    }
    if (reshi) *reshi = rhi;
    if (reslo) *reslo = rlo;
    TRACE_U64OPS((-1,"__u64mul() => %u:%u\n",rhi,rlo));
    #endif
  }
  return;
}

/* ----------------------------------------------------------------------- */

static void __u64div( u32 numerhi, u32 numerlo, u32 denomhi, u32 denomlo,
                      u32 *quothi, u32 *quotlo, u32 *remahi, u32 *remalo )
{
  /* when modifying this keep in mind that input parameters and result
     may overlap
  */
  if (!denomhi && !denomlo) /* treat divide by zero as "divide by an */
  {                         /* extremely small number" */
    /* we shouldn't ever have to handle this, but better safe than sorry */
    u32 qlo = numerlo, qhi = numerhi;
    if (quothi) *quothi = qhi;
    if (quotlo) *quotlo = qlo;
    if (remahi) *remahi = 0;
    if (remalo) *remalo = 0;
    return;
  }
  #if (ULONG_MAX > 0xfffffffful) /* 64 bit */
  {
    unsigned long n, d, r;
    n = (((unsigned long)numerhi)<<32UL)+((unsigned long)numerlo);
    d = (((unsigned long)denomhi)<<32UL)+((unsigned long)denomlo);
    if (quothi || quotlo)
    {
      r = n / d;
      if (quothi) *quothi = (u32)(r >> 32);
      if (quotlo) *quotlo = (u32)(r & 0xfffffffful);
    }
    if (remahi || remalo)
    {
      r = n % d;
      if (remahi) *remahi = (u32)(r >> 32);
      if (remalo) *remalo = (u32)(r & 0xfffffffful);
    }
  }
  #else /* 32bit - long division using rshift and sub */
  if ((numerhi == 0 && denomhi == 0) ||
      (denomlo == 0 && denomhi == 0)) /* catch divide by zero here */
  {
    TRACE_U64OPS((+1,"__u64div(%u:%u, %u:%u)\n",numerhi,numerlo,denomhi,denomlo));
    u32 n = numerlo, d = denomlo;
    if (remalo) *remalo = n % d;
    if (remahi) *remahi = 0;
    if (quotlo) *quotlo = n / d;
    if (quothi) *quothi = 0;
    TRACE_U64OPS((-1,"__u64div()=>0:%u [rem=0:%u]\n",n/d,n%d));
  }
  else
  {
    u32 qhi = 0, qlo = 0;
    u32 nhi = numerhi, nlo = numerlo;
    u32 dhi = denomhi, dlo = denomlo;
    int count = 0;
    TRACE_U64OPS((+1,"__u64div(%u:%u, %u:%u)\n",numerhi,numerlo,denomhi,denomlo));
    while ((dhi & 0x80000000ul) == 0)
    {
      if ((nhi < dhi) || ((nhi == dhi) && (nlo <= dlo)))
        break;
      dhi <<= 1; dhi |= (dlo >> 31);
      dlo <<= 1;
      count++;
    }
    while (count >= 0)
    {
      qhi <<= 1; qhi |= (qlo >> 31);
      qlo <<= 1;
      if ((nhi > dhi) || ((nhi == dhi) && (nlo >= dlo)))
      {
        u32 t = nlo - dlo;
        nhi -= dhi;
        if (t > nlo)
          nhi--;
        nlo = t;
        #if 0
        u32 t = ~dlo + 1;
        nhi += ~dhi;
        if (!t) nhi++;
        t += nlo;
        if (t < nlo)
          nhi++;
        nlo = t;
        #endif
        qlo |= 1;
      }
      dlo >>= 1; dlo |= (dhi << 31);
      dhi >>= 1;
      count--;
    }
    if (remahi) *remahi = nhi;
    if (remalo) *remalo = nlo;
    if (quothi) *quothi = qhi;
    if (quotlo) *quotlo = qlo;
    TRACE_U64OPS((-1,"__u64div()=>%u:%u [rem=%u:%u]\n",qhi,qlo,nhi,nlo));
  }
  #endif
}

/* ----------------------------------------------------------------------- */

char *U64stringify(char *buffer, unsigned int buflen, u32 hi, u32 lo,
                            int numstr_style, const char *numstr_suffix )
{
  /* numstring_style:
   * -1=unformatted,
   *  0=commas, magna
   *  1=0+space between magna and number (or at end if no magnitude char)
   *  2=1+numstr_suffix
  */
  TRACE_U64OPS((+1,"__u64stringify(%u:%u)\n",hi,lo));
  if (buffer && buflen)
  {
    char numbuf[32]; /* U64MAX is "18,446,744,073,709,551,615" (len=26) */
    unsigned int suffixstr_len;
    unsigned int suffix_len = 0;
    /* buffer = [buflen:digits,comma,dot][suffix_len:space,magna][suffixstr_len:keys/nodes/...][\0] */

    --buflen; // buffer[buflen] = '\0'
    if (numstr_style != 2 || !numstr_suffix)
      numstr_suffix = "";
    suffixstr_len = strlen( numstr_suffix );
    if (numstr_style == 2 && suffixstr_len == 0)
      numstr_style = 1;
    if (buflen && (numstr_style == 1 || numstr_style == 2))
    {
      ++suffix_len; /* space after number */
      --buflen;
    }
    if (buflen > suffixstr_len)
      buflen -= suffixstr_len;
    else if (buflen >= 5) /* so that it falls into next part */
      buflen = 4;

    if (buflen < 5)
    {
      strcpy( numbuf, "***" );
      suffixstr_len = 0;
      numstr_style = 0;
    }
    else
    {
      /* kilo(10**3), Mega(10**6), Giga(10**9), Tera(10**12), Peta(10**15),
         Exa(10**18), Zetta(10**21), Yotta(10**24)
      */
      static char magna_tab[]={0,'k','M','G','T','P','E','Z','Y'};
      unsigned int magna = 0, len = 0;

      #if (ULONG_MAX > 0xfffffffful) /* 64+ bit */
      {
        len = sprintf( numbuf, "%lu", (((unsigned long)hi)<<32UL)+((unsigned long)lo));
      }
      #else
      {
        u32 h = 0, m = 0, l = lo;
        if (hi)
        {
          __u64div( hi, lo, 0, 1000000000ul, &h, &m, 0, &l );
        }
        if (!h && !m)
          len = sprintf( numbuf, "%lu", (unsigned long)l );
        else if (!h)
          len = sprintf( numbuf, "%lu%09lu", (unsigned long)m, (unsigned long)l );
        else
        {
          __u64div( h, m, 0, 1000000000ul, 0,  &h, 0, &m );
          len = sprintf( numbuf, "%lu%09lu%09lu",
                         (unsigned long)h, (unsigned long)m, (unsigned long)l );
        }
      }
      #endif
      //printf("numl = %2d  ", len);
      if (numstr_style != -1 && len > 3) /* at least one comma separator */
      {
        char fmtbuf[sizeof(numbuf)];
        char *r = &numbuf[len];
        char *w = &fmtbuf[sizeof(fmtbuf)];
        len = 0;
        *--w = '\0';
        for (;;)
        {
          *--w = *--r;
          if (r == &numbuf[0])
            break;
          if (((++len) % 3)==0)
            *--w = ',';
        }
        len = strlen(strcpy( numbuf, w ));
        //printf("commal = %2d  ", len);
        if (len > buflen)
        {
          ++suffix_len; /* magna char */
          --buflen;
          len = buflen - 3;  /* rightmost location for decimal point */
          while (len > 0 && numbuf[len] != ',') /* find a spot for a dec pt */
            len--;
          if (len == 0) /* bufsz < 7 and "nnn,nn..." or "nn,nn..." */
          {
            strcpy(numbuf,"***");
            len = 3;
          }
          else
          {
            unsigned int pos = len;
            while (numbuf[pos] == ',')
            {
              magna++;
              pos += 4;
            }
            numbuf[len] = '.';
            len += 3;
            /* round the resulting number */
            if (numbuf[len] >= '5')
            {
              int carry = 1;
              r = &numbuf[len];
              w = &fmtbuf[sizeof(fmtbuf)];
              *--w = '\0';
              for (;;)
              {
                *--w = *--r;
                if (carry && '0' <= *w && *w <= '9')
                {
                  ++(*w);
                  if (*w > '9')
                    *w = '0';
                  else
                    carry = 0;
                }
                if (r == &numbuf[0])
                  break;
              }
              if (carry)
                *--w = '1';
              if (!carry) /* do not round if string length changes */
                len = strlen(strcpy( numbuf, w ));
            }
            numbuf[len] = '\0';
          }
        }
      } /* len > 3 */
      //printf("truncl = %2d  ", len);
      if (numstr_style == 1 || numstr_style == 2)
        numbuf[len++] = ' ';
      if (magna)
        numbuf[len++] = magna_tab[magna];
      numbuf[len] = '\0';
      //printf("l = %2d  bufl = %2d  sl = %2d  ss_l = %2d\n", len, buflen, suffix_len, suffixstr_len);
    } /* buflen >= 5 */
    if (strlen(numbuf) > (buflen + suffix_len))
      strcpy( numbuf, "***" );
    strncpy( buffer, numbuf, (buflen + suffix_len) );
    buffer[buflen+suffix_len] = '\0';
    if (numstr_style == 2) /* buflen has already been checked to ensure */
      strcat(buffer, numstr_suffix); /* this strcat() is ok */
  }
  TRACE_U64OPS((-1,"__u64stringify()=>'%s'\n",((buffer)?(buffer):("(null)"))));
  return buffer;
}

/* ----------------------------------------------------------------------- */

/* if secs:usecs is zero, ProblemComputeRate() just returns
   a formatted iterhi:lo
*/
const char *ProblemComputeRate( unsigned int contestid,
                                u32 secs, u32 usecs, u32 iterhi, u32 iterlo,
                                u32 *ratehi, u32 *ratelo,
                                char *ratebuf, unsigned int ratebufsz )
{
  u32 hi = iterhi, lo = iterlo;
  TRACE_U64OPS((+1,"ProblemComputeRate(%s,%u:%u,%u:%u)\n",
               CliGetContestNameFromID(contestid),secs,usecs,iterhi,iterlo));
  if (hi || lo)
  {
    u32 t, thi, tlo;
    __u64mul( 0, secs, 0, 1000, &thi, &tlo ); /* secs *= 1000 */
    t = tlo + ((usecs+499) / 1000);
    if (t < tlo) thi++;
    tlo = t;                                  /* ms = secs*1000+usecs/1000 */
    if (thi || tlo)
    {
      __u64mul( hi, lo, 0, 1000, &hi,  &lo ); /* iter *= 1000 */
      __u64div( hi, lo, thi, tlo, &hi, &lo, 0, 0 ); /* (iter*1000)/millisecs */
    }
  }
  if (ratehi) *ratehi = hi;
  if (ratelo) *ratelo = lo;
  if (ratebuf && ratebufsz)
  {
    U64stringify( ratebuf, ratebufsz, hi, lo, 2, CliGetContestUnitFromID(contestid) );
  }
  TRACE_U64OPS((-1,"ProblemComputeRate() => %u:%u\n",hi,lo));
  return ratebuf;
}

static unsigned int __compute_permille(unsigned int cont_i,
                                       const ContestWork *work)
{
  unsigned int permille = 0;
  switch (cont_i)
  {
// TODO: acidblood/trashover
#ifdef HAVE_OGR_PASS2
    case OGR_P2:
    if (work->ogr_p2.workstub.worklength > (u32)work->ogr_p2.workstub.stub.length)
    {
      // This is just a quick&dirty calculation that resembles progress.
      permille = work->ogr_p2.workstub.stub.diffs[work->ogr_p2.workstub.stub.length]*10
                +work->ogr_p2.workstub.stub.diffs[work->ogr_p2.workstub.stub.length+1]/10;
    }
    break;
#endif
#ifdef HAVE_OGR_CORES
    case OGR_NG:
    if (work->ogr_ng.workstub.worklength > (u32)work->ogr_ng.workstub.stub.length)
    {
      // This is just a quick&dirty calculation that resembles progress.
      permille = work->ogr_ng.workstub.stub.diffs[work->ogr_ng.workstub.stub.length]*10
                +work->ogr_ng.workstub.stub.diffs[work->ogr_ng.workstub.stub.length+1]/10;
    }
    break;
#endif
#ifdef HAVE_CRYPTO_V2
    case RC5_72:
    {
      if (work->bigcrypto.keysdone.lo || work->bigcrypto.keysdone.hi)
      {
        permille = 1000;
        if ((work->bigcrypto.keysdone.hi < work->bigcrypto.iterations.hi) ||
            ((work->bigcrypto.keysdone.hi== work->bigcrypto.iterations.hi) &&
            (work->bigcrypto.keysdone.lo < work->bigcrypto.iterations.lo)))
        {
          u32 hi,lo;
          __u64mul( work->bigcrypto.keysdone.hi, work->bigcrypto.keysdone.lo,
                    0, 1000, &hi, &lo );
          __u64div( hi, lo, work->bigcrypto.iterations.hi,
                            work->bigcrypto.iterations.lo, &hi, &lo, 0, 0);
          if (lo > 1000)
            lo = 1000;
          permille = lo;
        }
      }
    }
    break;
#endif

    default:
    PROJECT_NOT_HANDLED(cont_i);
    break;
  }
  return permille;
}


int WorkGetSWUCount( const ContestWork *work,
                     int rescode, unsigned int contestid,
                     unsigned int *swucount )
{
  if (rescode != RESULT_WORKING && rescode != RESULT_FOUND &&
      rescode != RESULT_NOTHING)
  {
    rescode = -1;
  }
  else
  {
    unsigned int units = 0;
    switch (contestid)
    {

#ifdef HAVE_OGR_PASS2
      case OGR_P2:
      {
        if (swucount && rescode != RESULT_WORKING)
        {
          u32 lo, tcountlo = work->ogr_p2.nodes.lo + 5000000ul;
          u32 hi, tcounthi = work->ogr_p2.nodes.hi + (tcountlo<work->ogr_p2.nodes.lo ? 1 : 0);
          /* ogr stats unit is Gnodes */
          __u64div( tcounthi, tcountlo, 0, 1000000000ul, 0, &hi, 0, &lo);
          units = (unsigned int)(hi * 100)+(lo / 10000000ul);
        }
      } /* OGR-P2 */
      break;
#endif
#ifdef HAVE_OGR_CORES
      case OGR_NG:
      {
        if (swucount && rescode != RESULT_WORKING)
        {
          u32 lo, tcountlo = work->ogr_ng.nodes.lo + 5000000ul;
          u32 hi, tcounthi = work->ogr_ng.nodes.hi + (tcountlo<work->ogr_ng.nodes.lo ? 1 : 0);
          /* ogr stats unit is Gnodes */
          __u64div( tcounthi, tcountlo, 0, 1000000000ul, 0, &hi, 0, &lo);
          units = (unsigned int)(hi * 100)+(lo / 10000000ul);
        }
      } /* OGR-NG */
      break;
#endif
#ifdef HAVE_CRYPTO_V2
      case RC5_72:
      {
        u32 tcounthi = work->bigcrypto.iterations.hi;
        /* note that we return zero for test packets */
        units = 100 * tcounthi;
      }
      break;
#endif

      default:
        PROJECT_NOT_HANDLED(contestid);
        break;
    } /* switch() */

    if (swucount)
      *swucount = units;
  } /* if (swucount) */

  return rescode;
}

// thisprob may be 0, gracefully exits
// info may not be 0, it must be a valid address of a ProblemInfo struct
// see problem.h for flag values and descriptions of ProblemInfo member vars
int ProblemGetInfo(void *__thisprob, ProblemInfo *info, long flags)
{
  int rescode = -1;
  ContestWork work;
  unsigned int contestid;
  InternalProblem *thisprob = __pick_probptr(__thisprob, PICKPROB_MAIN);

  if (thisprob)
  {
    rescode = ProblemRetrieveState( thisprob, &work, &contestid, 0, 0 );

    if (rescode >= 0)
    {
      u32 e_sec = 0, e_usec = 0;

      info->name = CliGetContestNameFromID(contestid);
      info->unit = CliGetContestUnitFromID(contestid);
#ifdef HAVE_RC5_72_CORES
      info->is_test_packet = contestid == RC5_72 &&
                             work.bigcrypto.iterations.lo == 0x00100000 &&
                             work.bigcrypto.iterations.hi == 0;
#else
      info->is_test_packet = 0;
#endif
// FIXME: hmmm is this correct ??????
      info->stats_units_are_integer = (contestid != OGR_P2 && contestid != OGR_NG);
      info->show_exact_iterations_done = (contestid == OGR_P2 || contestid == OGR_NG);

      if (flags & (P_INFO_E_TIME | P_INFO_RATE | P_INFO_RATEBUF))
      {
        if (thisprob->pub_data.elapsed_time_sec != 0xfffffffful)
        {
          // problem finished, elapsed time has already calculated by Run()
          e_sec  = thisprob->pub_data.elapsed_time_sec;
          e_usec = thisprob->pub_data.elapsed_time_usec;
        }
        else /* compute elapsed wall clock time since loadtime */
        {
          u32 start_sec  = thisprob->priv_data.loadtime_sec;
          u32 start_usec = thisprob->priv_data.loadtime_usec;

          if (start_sec != 0xfffffffful) /* our start time was not invalid */
          {
            struct timeval clock_now;
            if (CliGetMonotonicClock(&clock_now) != 0)
            {
              if (CliGetMonotonicClock(&clock_now) != 0)
                start_sec = 0xfffffffful; /* no current time, so make start invalid */
            }
            if (start_sec != 0xfffffffful)
            {
              e_sec  = clock_now.tv_sec;
              e_usec = clock_now.tv_usec;
            }
          }
          if (start_sec == 0xfffffffful || /* start time is invalid */
              e_sec <  start_sec || (e_sec == start_sec && e_usec < start_usec ))
          {
            /* either start time is invalid, or current-time < start-time */
            /* both are BadThing(TM)s - have to use the per-run total */
            e_sec  = thisprob->pub_data.runtime_sec;
            e_usec = thisprob->pub_data.runtime_usec;
          }
          else /* start and 'now' time are ok */
          {
            if (e_usec < start_usec)
            {
              e_usec += 1000000UL;
              e_sec  --;
            }
            e_sec  -= start_sec;
            e_usec -= start_usec;
          }
        }
        if (flags & P_INFO_E_TIME)
        {
          info->elapsed_secs = e_sec;
          info->elapsed_usecs = e_usec;
        }
        if (thisprob->pub_data.is_benchmark && (thisprob->pub_data.runtime_sec || thisprob->pub_data.runtime_usec))
        {
          e_sec = thisprob->pub_data.runtime_sec;
          e_usec = thisprob->pub_data.runtime_usec;
        }
      } // if (flags & (P_INFO_E_TIME | P_INFO_RATE | P_INFO_RATEBUF))
      if ( flags & (P_INFO_C_PERMIL | P_INFO_S_PERMIL | P_INFO_RATE   |
                    P_INFO_RATEBUF  | P_INFO_SIGBUF   | P_INFO_CWPBUF |
                    P_INFO_SWUCOUNT | P_INFO_TCOUNT   | P_INFO_CCOUNT |
                    P_INFO_DCOUNT) )
      {
          u32 hi, lo;
          u32 tcounthi=0, tcountlo=0; /*total 'iter' (n/a if not finished)*/
          u32 ccounthi=0, ccountlo=0; /*'iter' done (so far, this start) */
          u32 dcounthi=0, dcountlo=0; /*'iter' done (so far, all starts) */

          switch (contestid)
          {
   #ifdef HAVE_OGR_PASS2
            case OGR_P2:
            {
              dcounthi = work.ogr_p2.nodes.hi;
              dcountlo = work.ogr_p2.nodes.lo;
              if (rescode == RESULT_NOTHING || rescode == RESULT_FOUND)
              {
                tcounthi = dcounthi;
                tcountlo = dcountlo;
              }
              /* current = donecount - startpos */
              ccountlo = dcountlo - thisprob->pub_data.startkeys.lo;
              ccounthi = dcounthi - thisprob->pub_data.startkeys.hi;
              if (ccountlo > dcountlo)
                ccounthi--;

              if (flags & P_INFO_SIGBUF)
              {
                ogr_stubstr_r( &work.ogr_p2.workstub.stub, info->sigbuf, sizeof(info->sigbuf), 0);
              }
              if (flags & P_INFO_CWPBUF)
              {
                ogr_stubstr_r( &work.ogr_p2.workstub.stub, info->cwpbuf, sizeof(info->sigbuf), work.ogr_p2.workstub.worklength);
              }
              if ((flags & P_INFO_SWUCOUNT) && (tcounthi || tcountlo)) /* only if finished */
              {
                u32 nodeshi, nodeslo;
                /* ogr stats unit is Gnodes, rounded to 0.01 Gnodes */
                nodeslo = tcountlo + 5000000ul;
                nodeshi = tcounthi + (nodeslo<tcountlo ? 1 : 0);
                __u64div( nodeshi, nodeslo, 0, 1000000000ul, 0, &hi, 0, &lo);
                info->swucount = (hi * 100)+(lo / 10000000ul);
              }
              if (flags & P_INFO_EXACT_PE)
              {
                if (flags & P_INFO_C_PERMIL)
                  info->c_permille = 0;
                if (flags & P_INFO_S_PERMIL)
                  info->s_permille = 0;
                /* do not do inexact permille calculation for OGR */
                flags &= ~(P_INFO_C_PERMIL | P_INFO_S_PERMIL);
              }
            } /* OGR-P2 */
            break;
  #endif
  #ifdef HAVE_OGR_CORES
            case OGR_NG:
            {
              dcounthi = work.ogr_ng.nodes.hi;
              dcountlo = work.ogr_ng.nodes.lo;
              if (rescode == RESULT_NOTHING || rescode == RESULT_FOUND)
              {
                tcounthi = dcounthi;
                tcountlo = dcountlo;
              }
              /* current = donecount - startpos */
              ccountlo = dcountlo - thisprob->pub_data.startkeys.lo;
              ccounthi = dcounthi - thisprob->pub_data.startkeys.hi;
              if (ccountlo > dcountlo)
                ccounthi--;

              if (flags & P_INFO_SIGBUF)
              {
                ogrng_stubstr_r(&work.ogr_ng.workstub, info->sigbuf, sizeof(info->sigbuf), 0);
              }
              if (flags & P_INFO_CWPBUF)
              {
                ogrng_stubstr_r(&work.ogr_ng.workstub, info->cwpbuf, sizeof(info->sigbuf), 10);
              }
              if ((flags & P_INFO_SWUCOUNT) && (tcounthi || tcountlo)) /* only if finished */
              {
                u32 nodeshi, nodeslo;
                /* ogr stats unit is Gnodes, rounded to 0.01 Gnodes */
                nodeslo = tcountlo + 5000000ul;
                nodeshi = tcounthi + (nodeslo<tcountlo ? 1 : 0);
                __u64div( nodeshi, nodeslo, 0, 1000000000ul, 0, &hi, 0, &lo);
                info->swucount = (hi * 100)+(lo / 10000000ul);
              }
              if (flags & P_INFO_EXACT_PE)
              {
                if (flags & P_INFO_C_PERMIL)
                  info->c_permille = 0;
                if (flags & P_INFO_S_PERMIL)
                  info->s_permille = 0;
                /* do not do inexact permille calculation for OGR */
                flags &= ~(P_INFO_C_PERMIL | P_INFO_S_PERMIL);
              }
            } /* OGR-NG */
            break;
  #endif
  #ifdef HAVE_CRYPTO_V2
            case RC5_72:
            {
              unsigned int units, twoxx;

              ccounthi = thisprob->pub_data.startkeys.hi;
              ccountlo = thisprob->pub_data.startkeys.lo;
              tcounthi = work.bigcrypto.iterations.hi;
              tcountlo = work.bigcrypto.iterations.lo;
              dcounthi = work.bigcrypto.keysdone.hi;
              dcountlo = work.bigcrypto.keysdone.lo;
              /* current = donecount - startpos */
              ccountlo = dcountlo - ccountlo;
              ccounthi = dcounthi - ccounthi;
              if (ccountlo > dcountlo)
                ccounthi--;

              units = tcounthi;
              twoxx = 32;
              if (!units) /* less than 2^32 packet (eg test) */
              {
                units = tcountlo >> 20;
                twoxx = 20;
              }
              if (rescode != RESULT_NOTHING && rescode != RESULT_FOUND)
              {
                tcounthi = 0;
                tcountlo = 0;
              }
              if (flags & P_INFO_SIGBUF)
              {
                sprintf( info->sigbuf, "%02lX:%08lX:%08lX:%u*2^%u",
                         (unsigned long) ( work.bigcrypto.key.hi ),
                         (unsigned long) ( work.bigcrypto.key.mid ),
                         (unsigned long) ( work.bigcrypto.key.lo ),
                         units, twoxx );
              }
              if (flags & P_INFO_CWPBUF)
              {
                // ToDo: do something different here - any ideas for a cwp for crypto packets?
                sprintf( info->cwpbuf, "%02lX:%08lX:%08lX:%u*2^%u",
                         (unsigned long) ( work.bigcrypto.key.hi ),
                         (unsigned long) ( work.bigcrypto.key.mid ),
                         (unsigned long) ( work.bigcrypto.key.lo ),
                         units, twoxx );
              }
              if ((flags & P_INFO_SWUCOUNT) && (tcounthi || tcountlo)) /* only if finished */
              {
                /* note that we return zero for test packets */
                info->swucount = tcounthi*100;
              }
            } /* case: crypto */
            break;
  #endif /* HAVE_CRYPTO_V2 */

            default:
            PROJECT_NOT_HANDLED(contestid);
            break;
          } /* switch() */

          if (flags & (P_INFO_RATE | P_INFO_RATEBUF))
          {
            if (!(flags & P_INFO_RATEBUF))
            {
              info->rate.ratebuf = 0;
              info->rate.size = 0;
            }
            ProblemComputeRate( contestid, e_sec, e_usec, ccounthi, ccountlo,
                                &hi, &lo, info->rate.ratebuf, info->rate.size );
            ProjectSetSpeed(contestid, hi, lo);
            info->ratehi = hi;
            info->ratelo = lo;
          }
          if (flags & P_INFO_C_PERMIL)
          {
            if (!thisprob->priv_data.started)
              info->c_permille = thisprob->pub_data.startpermille;
            else if (rescode != RESULT_WORKING) /* _FOUND or _NOTHING */
              info->c_permille = 1000;
            else
              info->c_permille = __compute_permille(contestid, &work);
          }
          if (flags & P_INFO_S_PERMIL)
            info->s_permille = thisprob->pub_data.startpermille;
          if (flags & P_INFO_DCOUNT)
          {
            info->dcounthi = dcounthi;
            info->dcountlo = dcountlo;
          }
          if (flags & P_INFO_CCOUNT)
          {
            info->ccounthi = ccounthi;
            info->ccountlo = ccountlo;
          }
          if (flags & P_INFO_TCOUNT)
          {
            info->tcounthi = tcounthi;
            info->tcountlo = tcountlo;
          }
      } /* if (sigbuf || ... ) */
    } /* if (rescode >= 0) */
  } /* if thisprob != 0 */
  return rescode;
}

// this is a bit misplaced here, but we need u64-ops
// speed is in units per second
int ProjectSetSpeed(int projectid, u32 speedhi, u32 speedlo)
{
  TRACE_OUT((0, "ProjectSetSpeed(prj=%d, hi=%d, lo=%d)\n", projectid, speedhi, speedlo));
  u32 wusizehi = 0, wusizelo = 0;
  u32 sechi = 0, seclo = 0; // we calculate seconds per work unit
  switch(projectid) {
    case OGR_NG:
    case OGR_P2:
      /* unknown */
      wusizehi = wusizelo = 0;
      break;
    case RC5_72:
      /* 2^32 */
      wusizehi = 1;
      wusizelo = 0;
      break;
    default:
      PROJECT_NOT_HANDLED(projectid);
      return -1;
  }
  if ((speedhi|speedlo) == 0)
    speedlo = 1;
  __u64div( wusizehi, wusizelo, speedhi, speedlo, &sechi, &seclo, 0, 0);
  if (seclo && !sechi)
    return CliSetContestWorkUnitSpeed( projectid, seclo );
  return 0;
}
