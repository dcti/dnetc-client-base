/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
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
return "@(#)$Id: problem.cpp,v 1.108.2.108.2.7 2001/03/22 22:41:24 sampo Exp $"; }

//#define TRACE
#define TRACE_U64OPS(x) TRACE_OUT(x)

#include "cputypes.h"
#include "baseincs.h"
#include "version.h"  //CLIENT_BUILD_FRAC
#include "client.h"   //CONTEST_COUNT
#include "clitime.h"  //CliClock()
#include "logstuff.h" //LogScreen()
#include "probman.h"  //GetProblemPointerFromIndex()
#include "random.h"   //Random()
#include "rsadata.h"  //Get cipher/etc for random blocks
#include "clicdata.h" //CliSetContestWorkUnitSpeed()
#include "selcore.h"  //selcoreGetSelectedCoreForContest()
#include "util.h"     //trace
#include "cpucheck.h" //hardware detection
#include "console.h"  //ConOutErr
#include "triggers.h" //RaiseExitRequestTrigger()
#include "clisync.h"  //synchronisation primitives
#include "problem.h"  //ourselves

//#define STRESS_THREADS_AND_BUFFERS /* !be careful with this! */

#ifndef MINIMUM_ITERATIONS
#define MINIMUM_ITERATIONS 24
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

#if !defined(MIPSpro) 
  #if (SIZEOF_LONG == 8)  /* SIZEOF_LONG is defined in cputypes.h */
    #pragma pack(8)
  #else    
    #pragma pack(4)
  #endif    
#endif

typedef struct
{
  struct problem_publics pub_data; /* public members - must come first */
  struct
  {
    /* the following must be protected for thread safety */
    /* --------------------------------------------------------------- */
    RC5UnitWork rc5unitwork; /* MUST BE longword (64bit) aligned */
    struct {u32 hi,lo;} refL0;
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
  } priv_data;
} InternalProblem;

#ifndef MIPSpro
#pragma pack()
#endif

/* ======================================================================= */

#ifndef MIPSpro
#pragma pack(1)
#endif

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
  InternalProblem twist_and_shout[3]; 
  #define PICKPROB_MAIN 0 /* MAIN must be first */
  #define PICKPROB_CORE 1
  #define PICKPROB_TEMP 2 /* temporary copy used by load */
  mutex_t copy_lock; /* locked when a sync is in progress */
} SuperProblem;  

#ifndef MIPSpro
#pragma pack()
#endif

unsigned int ProblemGetSize(void)
{ /* needed by IPC/shmem */
  return sizeof(SuperProblem);
}  

void ProblemFree(void *__thisprob)
{
  SuperProblem *thisprob = (SuperProblem *)__thisprob;
  if (thisprob)
  {
    // sorry, this is needed for mutex emulation
    #if (CLIENT_OS == OS_MACOS) && (CLIENT_CPU == CPU_POWERPC)
    if (MPLibraryIsLoaded())
      MPDeleteCriticalRegion(thisprob->copy_lock.MPregion);
    #endif
    
    memset( thisprob, 0, sizeof(SuperProblem) );
    __problem_counter--;
    free((void *)thisprob);
  }
  return;
}

Problem *ProblemAlloc(void)
{
  char *p;
  SuperProblem *thisprob = (SuperProblem *)0;
  int err = 0;

  #ifdef STRESS_THREADS_AND_BUFFERS
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
    thisprob = (SuperProblem *)malloc(sizeof(SuperProblem));
    if (!thisprob)
    {
      Log("Insufficient memory to allocate problem data\n");
      err = 1;
    }
  }    
    
  if (thisprob && !err)
  {
    p = (char *)&(thisprob->twist_and_shout[PICKPROB_CORE].priv_data.rc5unitwork);
    if ((((unsigned long)p) & (sizeof(void *)-1)) != 0)
    {
      /* Ensure that the core data is going to be aligned */
      Log("priv_data.rc5unitwork for problem %d is misaligned!\n", __problem_counter);
      err = 1;
    }
    else 
    {
      /* Ensure that what we return as 'Problem *' is valid */
      if ( ((Problem *)thisprob) != 
        ((Problem *)&(thisprob->twist_and_shout[PICKPROB_MAIN].pub_data)) )
      {
        Log("Ack! Phui! Problem != Problem\n");
        err = 1;
      }  
    }
  }      

  if (thisprob && !err)
  {
    mutex_t initmux = DEFAULTMUTEX; /* {0} or whatever */

    memset( thisprob, 0, sizeof(SuperProblem) );
    memcpy( &(thisprob->copy_lock), &initmux, sizeof(mutex_t));
    
    // sorry, this is needed for mutex emulation
    #if (CLIENT_OS == OS_MACOS) && (CLIENT_CPU == CPU_POWERPC)
    if (MPLibraryIsLoaded())
      MPCreateCriticalRegion(&(thisprob->copy_lock.MPregion));
    #endif
    
    thisprob->twist_and_shout[PICKPROB_CORE].priv_data.threadindex = 
    thisprob->twist_and_shout[PICKPROB_MAIN].priv_data.threadindex = 
    thisprob->twist_and_shout[PICKPROB_TEMP].priv_data.threadindex = 
                                              __problem_counter++;

    //align core_membuffer to 16byte boundary
    p = &(thisprob->twist_and_shout[PICKPROB_CORE].priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->twist_and_shout[PICKPROB_CORE].priv_data.core_membuffer = p;

    p = &(thisprob->twist_and_shout[PICKPROB_MAIN].priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->twist_and_shout[PICKPROB_MAIN].priv_data.core_membuffer = p;

    p = &(thisprob->twist_and_shout[PICKPROB_TEMP].priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->twist_and_shout[PICKPROB_TEMP].priv_data.core_membuffer = p;
  }
  
  if (thisprob && err)
  {
    free((void *)thisprob);
    thisprob = (SuperProblem *)0;
  }
  return (Problem *)thisprob;
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
    
    return &(p->twist_and_shout[which]);
  }    
  return (InternalProblem *)0;
}

static inline void __assert_lock( void *__thisprob )
{
  SuperProblem *p = (SuperProblem *)__thisprob;
  mutex_lock(&(p->copy_lock));
}

static inline void __release_lock( void *__thisprob )
{
  SuperProblem *p = (SuperProblem *)__thisprob;
  mutex_unlock(&(p->copy_lock));
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

// for some odd reasons, the RC5 algorithm requires keys in reversed order
//         key.hi   key.lo
// ie key 01234567:89ABCDEF is sent to rc5_pub_data.unit_func like that :
//        EFCDAB89:67452301
// This function switches from one format to the other.
//
// [Even if it looks like a little/big endian problem, it isn't. Whatever
//  endianess the underlying system has, we must swap every byte in the key
//  before sending it to rc5_pub_data.unit_func()]
//
// Note that DES has a similiar but far more complex system, but everything
// is handled by des_pub_data.unit_func().

static void  __SwitchRC5Format(u32 *hi, u32 *lo)
{
    register u32 tempkeylo = *hi; /* note: we switch the order */
    register u32 tempkeyhi = *lo;

    *lo =
      ((tempkeylo >> 24) & 0x000000FFL) |
      ((tempkeylo >>  8) & 0x0000FF00L) |
      ((tempkeylo <<  8) & 0x00FF0000L) |
      ((tempkeylo << 24) & 0xFF000000L);
    *hi =
      ((tempkeyhi >> 24) & 0x000000FFL) |
      ((tempkeyhi >>  8) & 0x0000FF00L) |
      ((tempkeyhi <<  8) & 0x00FF0000L) |
      ((tempkeyhi << 24) & 0xFF000000L);
}

/* ------------------------------------------------------------------- */

// Input:  - an RC5 key in 'mangled' (reversed) format or a DES key
//         - an incrementation count
//         - a contest identifier (0==RC5 1==DES 2==OGR 3==CSC)
//
// Output: the key incremented

static void __IncrementKey(u32 *keyhi, u32 *keylo, u32 iters, int contest)
{
  switch (contest)
  {
    case RC5:
      __SwitchRC5Format(keyhi,keylo);
      *keylo = *keylo + iters;
      if (*keylo < iters) *keyhi = *keyhi + 1;
      __SwitchRC5Format (keyhi,keylo);
      break;
    case DES:
    case CSC:
      *keylo = *keylo + iters;
      if (*keylo < iters) *keyhi = *keyhi + 1; /* Account for carry */
      break;
    case OGR:
      /* This should never be called for OGR */
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
    case RC5:
    case DES:
    case CSC:
    {
      work->crypto.key.lo = ( 0 );
      work->crypto.key.hi = ( 0 );
      work->crypto.iv.lo = ( 0 );
      work->crypto.iv.hi = ( 0 );
      work->crypto.plain.lo = ( 0 );
      work->crypto.plain.hi = ( 0 );
      work->crypto.cypher.lo = ( 0 );
      work->crypto.cypher.hi = ( 0 );
      work->crypto.keysdone.lo = ( 0 );
      work->crypto.keysdone.hi = ( 0 );
      work->crypto.iterations.lo = ( 0 );
      work->crypto.iterations.hi = ( 1 );
      return contestid;
    }
    #if defined(HAVE_OGR_CORES)
    case OGR:
    {
      //24/2-22-32-21-5-1-12
      //25/6-9-30-14-10-11
      work->ogr.workstub.stub.marks = 25;    //24;
      work->ogr.workstub.worklength = 6;     //7;
      work->ogr.workstub.stub.length = 6;    //7;
      work->ogr.workstub.stub.diffs[0] = 6;  //2;
      work->ogr.workstub.stub.diffs[1] = 9;  //22;
      work->ogr.workstub.stub.diffs[2] = 30;  //32;
      work->ogr.workstub.stub.diffs[3] = 14; //21;
      work->ogr.workstub.stub.diffs[4] = 10;  //5;
      work->ogr.workstub.stub.diffs[5] = 11;  //1;
      work->ogr.workstub.stub.diffs[6] = 0;  //12;
      work->ogr.nodes.lo = 0;
      work->ogr.nodes.hi = 0;
      return contestid;
    }
    #endif
    default:
      break;
  }
  return -1;
}

/* ------------------------------------------------------------------- */

static int last_rc5_prefix = -1;

static int __gen_random_work(unsigned int contestid, ContestWork * work)
{
  // the random prefix is updated by LoadState() for every RC5 block loaded
  // that is >= 2^28 (thus excludes test blocks)
  // make one up in the event that no block was every loaded.

  u32 rnd = Random(NULL,0);
  u32 randomprefix = last_rc5_prefix;
  if (last_rc5_prefix == -1) /* no random prefix determined yet */
    last_rc5_prefix = randomprefix = 100+(rnd % (0xff-100));

  contestid = RC5; 
  work->crypto.key.lo   = (rnd & 0xF0000000L);
  work->crypto.key.hi   = (rnd & 0x00FFFFFFL) + (last_rc5_prefix<<24);
  //constants are in rsadata.h
  work->crypto.iv.lo     = ( RC564_IVLO );     //( 0xD5D5CE79L );
  work->crypto.iv.hi     = ( RC564_IVHI );     //( 0xFCEA7550L );
  work->crypto.cypher.lo = ( RC564_CYPHERLO ); //( 0x550155BFL );
  work->crypto.cypher.hi = ( RC564_CYPHERHI ); //( 0x4BF226DCL );
  work->crypto.plain.lo  = ( RC564_PLAINLO );  //( 0x20656854L );
  work->crypto.plain.hi  = ( RC564_PLAINHI );  //( 0x6E6B6E75L );
  work->crypto.keysdone.lo = 0;
  work->crypto.keysdone.hi = 0;
  work->crypto.iterations.lo = 1L<<28;
  work->crypto.iterations.hi = 0;
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
*/
static inline int __InternalLoadState( InternalProblem *thisprob,
                      const ContestWork * work, unsigned int contestid,
                      u32 _iterations, int expected_cputype,
                      int expected_corenum, int expected_os,
                      int expected_buildfrac )
{
  ContestWork for_magic;
  int genned_random = 0, genned_benchmark = 0;
  struct selcore selinfo; int coresel;

  //has to be done before anything else
  if (work == CONTESTWORK_MAGIC_RANDOM) /* ((const ContestWork *)0) */
  {
    contestid = __gen_random_work(contestid, &for_magic);
    work = &for_magic;
    genned_random = 1;
  }
  else if (work == CONTESTWORK_MAGIC_BENCHMARK) /* ((const ContestWork *)1) */
  {
    contestid = __gen_benchmark_work(contestid, &for_magic);
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
    return -1;
  }
  if (contestid == RC5 && (MINIMUM_ITERATIONS % selinfo.pipeline_count) != 0)
  {
    LogScreen("(MINIMUM_ITERATIONS %% thisprob->pub_data.pipeline_count) != 0)\n");
    return -1;
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
    if (riscos_check_taskwindow() && thisprob->pub_data.client_cpu!=CPU_X86)
      thisprob->pub_data.cruncher_is_time_constrained = 1;  
    #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)  
    if (winGetVersion() < 400)
      thisprob->pub_data.cruncher_is_time_constrained = 1;  
    #elif (CLIENT_OS == OS_MACOS)
      thisprob->pub_data.cruncher_is_time_constrained = 1;
    #elif (CLIENT_OS == OS_NETWARE)  
      thisprob->pub_data.cruncher_is_time_constrained = 1;
    #endif  
  }

  //----------------------------------------------------------------

  if (thisprob->pub_data.contest == RC5)
  {
    if (!thisprob->pub_data.is_random &&
       (work->crypto.iterations.hi || work->crypto.iterations.lo >= (1L<<28)))
    {
      last_rc5_prefix = (int)(work->crypto.key.hi >> 24);
    }    
  }
  if (thisprob->pub_data.contest == RC5
   || thisprob->pub_data.contest == DES
   || thisprob->pub_data.contest == CSC)
  {
    // copy over the state information
    thisprob->priv_data.contestwork.crypto.key.hi = ( work->crypto.key.hi );
    thisprob->priv_data.contestwork.crypto.key.lo = ( work->crypto.key.lo );
    thisprob->priv_data.contestwork.crypto.iv.hi = ( work->crypto.iv.hi );
    thisprob->priv_data.contestwork.crypto.iv.lo = ( work->crypto.iv.lo );
    thisprob->priv_data.contestwork.crypto.plain.hi = ( work->crypto.plain.hi );
    thisprob->priv_data.contestwork.crypto.plain.lo = ( work->crypto.plain.lo );
    thisprob->priv_data.contestwork.crypto.cypher.hi = ( work->crypto.cypher.hi );
    thisprob->priv_data.contestwork.crypto.cypher.lo = ( work->crypto.cypher.lo );
    thisprob->priv_data.contestwork.crypto.keysdone.hi = ( work->crypto.keysdone.hi );
    thisprob->priv_data.contestwork.crypto.keysdone.lo = ( work->crypto.keysdone.lo );
    thisprob->priv_data.contestwork.crypto.iterations.hi = ( work->crypto.iterations.hi );
    thisprob->priv_data.contestwork.crypto.iterations.lo = ( work->crypto.iterations.lo );

    if (thisprob->priv_data.contestwork.crypto.keysdone.lo || thisprob->priv_data.contestwork.crypto.keysdone.hi)
    {
      if (thisprob->pub_data.client_cpu != expected_cputype || thisprob->pub_data.coresel != expected_corenum ||
          CLIENT_OS != expected_os || CLIENT_BUILD_FRAC!=expected_buildfrac)
      {
        thisprob->priv_data.contestwork.crypto.keysdone.lo = thisprob->priv_data.contestwork.crypto.keysdone.hi = 0;
        thisprob->pub_data.was_reset = 1;
      }
    }
    //determine starting key number. accounts for carryover & highend of keysdone
    thisprob->priv_data.rc5unitwork.L0.hi = thisprob->priv_data.contestwork.crypto.key.hi + thisprob->priv_data.contestwork.crypto.keysdone.hi +
       ((((thisprob->priv_data.contestwork.crypto.key.lo & 0xffff) + (thisprob->priv_data.contestwork.crypto.keysdone.lo & 0xffff)) +
         ((thisprob->priv_data.contestwork.crypto.key.lo >> 16) + (thisprob->priv_data.contestwork.crypto.keysdone.lo >> 16))) >> 16);
    thisprob->priv_data.rc5unitwork.L0.lo = thisprob->priv_data.contestwork.crypto.key.lo + thisprob->priv_data.contestwork.crypto.keysdone.lo;
    if (thisprob->pub_data.contest == RC5)
      __SwitchRC5Format(&(thisprob->priv_data.rc5unitwork.L0.hi), &(thisprob->priv_data.rc5unitwork.L0.lo));
    thisprob->priv_data.refL0.lo = thisprob->priv_data.rc5unitwork.L0.lo;
    thisprob->priv_data.refL0.hi = thisprob->priv_data.rc5unitwork.L0.hi;
    // set up the unitwork structure
    thisprob->priv_data.rc5unitwork.plain.hi = thisprob->priv_data.contestwork.crypto.plain.hi ^ thisprob->priv_data.contestwork.crypto.iv.hi;
    thisprob->priv_data.rc5unitwork.plain.lo = thisprob->priv_data.contestwork.crypto.plain.lo ^ thisprob->priv_data.contestwork.crypto.iv.lo;
    thisprob->priv_data.rc5unitwork.cypher.hi = thisprob->priv_data.contestwork.crypto.cypher.hi;
    thisprob->priv_data.rc5unitwork.cypher.lo = thisprob->priv_data.contestwork.crypto.cypher.lo;

    thisprob->pub_data.startkeys.hi = thisprob->priv_data.contestwork.crypto.keysdone.hi;
    thisprob->pub_data.startkeys.lo = thisprob->priv_data.contestwork.crypto.keysdone.lo;
    thisprob->pub_data.startpermille = __compute_permille( thisprob->pub_data.contest, &thisprob->priv_data.contestwork );
  }
  #if defined(HAVE_OGR_CORES)
  else if (thisprob->pub_data.contest == OGR)
  {
    int r;
    thisprob->priv_data.contestwork.ogr = work->ogr;
    if (thisprob->priv_data.contestwork.ogr.nodes.hi != 0 || thisprob->priv_data.contestwork.ogr.nodes.lo != 0)
    {
      if (thisprob->pub_data.client_cpu != expected_cputype || thisprob->pub_data.coresel != expected_corenum ||
          CLIENT_OS != expected_os || CLIENT_BUILD_FRAC!=expected_buildfrac)
      {
        thisprob->pub_data.was_reset = 1;
        thisprob->priv_data.contestwork.ogr.workstub.worklength = thisprob->priv_data.contestwork.ogr.workstub.stub.length;
        thisprob->priv_data.contestwork.ogr.nodes.hi = thisprob->priv_data.contestwork.ogr.nodes.lo = 0;
      }
    }

    r = (thisprob->pub_data.unit_func.ogr)->init();
    if (r == CORE_S_OK)
    {
      r = (thisprob->pub_data.unit_func.ogr)->create(&thisprob->priv_data.contestwork.ogr.workstub,
                      sizeof(WorkStub), thisprob->priv_data.core_membuffer, MAX_MEM_REQUIRED_BY_CORE);
    }
    if (r != CORE_S_OK)
    {
      /* if it got here, then the stub is truly bad or init failed and
      ** it is ok to discard the stub (and let the network recycle it)
      */
      const char *msg = "Unknown error";
      if      (r == CORE_E_MEMORY)  msg = "CORE_E_MEMORY: Insufficient memory";
      else if (r == CORE_E_FORMAT)  msg = "CORE_E_FORMAT: Format or range error";
      Log("OGR load failure: %s\nStub discarded.\n", msg );
      return -1;
    }
    if (thisprob->priv_data.contestwork.ogr.workstub.worklength > (u32)thisprob->priv_data.contestwork.ogr.workstub.stub.length)
    {
      thisprob->pub_data.startkeys.hi = thisprob->priv_data.contestwork.ogr.nodes.hi;
      thisprob->pub_data.startkeys.lo = thisprob->priv_data.contestwork.ogr.nodes.lo;
      thisprob->pub_data.startpermille = __compute_permille( thisprob->pub_data.contest, &thisprob->priv_data.contestwork );
    }
  }
  #endif

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
*/
int ProblemLoadState( void *__thisprob,
                      const ContestWork * work, unsigned int contestid,
                      u32 _iterations, int expected_cputype,
                      int expected_corenum, int expected_os,
                      int expected_buildfrac )
{
  InternalProblem *temp_prob = __pick_probptr(__thisprob, PICKPROB_TEMP);
  InternalProblem *main_prob = __pick_probptr(__thisprob, PICKPROB_MAIN);
  if (!temp_prob || !main_prob)
  {
    return -1;
  }

  __assert_lock(__thisprob);
  __copy_internal_problem( temp_prob, main_prob ); /* copy main->temp */
  __release_lock(__thisprob);

  if (__InternalLoadState( temp_prob, work, contestid, _iterations, 
                           expected_cputype, expected_corenum, expected_os,
                           expected_buildfrac ) != 0)
  {
    return -1;
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
        case RC5:
        case DES:
        case CSC:
        {
          memcpy( (void *)work, 
                  (void *)&thisprob->priv_data.contestwork, 
                  sizeof(ContestWork));
          break;
        }
        #if defined(HAVE_OGR_CORES)
        case OGR:
        {
          (thisprob->pub_data.unit_func.ogr)->getresult(
                       thisprob->priv_data.core_membuffer, 
                       &thisprob->priv_data.contestwork.ogr.workstub, 
                       sizeof(WorkStub));
          memcpy( (void *)work, 
                  (void *)&thisprob->priv_data.contestwork, 
                  sizeof(ContestWork));

          /* is the stub invalid? */
          if (thisprob->priv_data.last_resultcode == RESULT_NOTHING &&
              work->ogr.nodes.hi == 0 && work->ogr.nodes.lo == 0)
          {
            #if defined(STUB_E_GOLOMB) /* newer ansi core */
            if (!thisprob->pub_data.was_truncated)
            {
              unsigned int r = work->ogr.workstub.worklength;
              const char *reason = "STUB_E_*: Undefined core error";
              if      (r == STUB_E_MARKS)  reason = "STUB_E_MARKS: Stub is not supported by this client";
              else if (r == STUB_E_GOLOMB) reason = "STUB_E_GOLOMB: Stub is not golomb";
              else if (r == STUB_E_LIMIT)  reason = "STUB_E_LIMIT: Stub is obsolete";
              thisprob->pub_data.was_truncated = reason;
            }
            #endif
          }    
          break;
        } 
        #endif
        default: /* cannot happen */  
        {
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

static int Run_RC5(InternalProblem *thisprob, /* already validated */
                   u32 *keyscheckedP /* count of ... */, int *resultcode)
{
  s32 rescode = -1;

  /* a brace to ensure 'keystocheck' is not referenced in the common part */
  {
    u32 keystocheck = *keyscheckedP;
    // don't allow a too large of a keystocheck be used ie (>(iter-keysdone))
    // (technically not necessary, but may save some wasted time)
    if (thisprob->priv_data.contestwork.crypto.keysdone.hi == thisprob->priv_data.contestwork.crypto.iterations.hi)
    {
      u32 todo = thisprob->priv_data.contestwork.crypto.iterations.lo-thisprob->priv_data.contestwork.crypto.keysdone.lo;
      if (todo < keystocheck)
        keystocheck = todo;
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
      rescode = (*(thisprob->pub_data.unit_func.gen))(&thisprob->priv_data.rc5unitwork,keyscheckedP,thisprob->priv_data.core_membuffer);

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
      *keyscheckedP = (*(thisprob->pub_data.unit_func.rc5))(&thisprob->priv_data.rc5unitwork,(keystocheck/thisprob->pub_data.pipeline_count));
      //don't use the next few lines as a guide for conversion to unified
      //prototypes!  look at the end of rc5/ansi/rc5ansi_2-rg.cpp instead.
      if (*keyscheckedP < keystocheck)
        rescode = RESULT_FOUND;
      else if (*keyscheckedP == keystocheck)
        rescode = RESULT_WORKING; /* synonymous with RESULT_NOTHING */
      else
        rescode = -1;
    }
  } /* brace to ensure that 'keystocheck' is not referenced beyond here */
  /* -- the code from here on down is identical to that of CSC -- */

  if (rescode < 0) /* "kiter" error */
  {
    *resultcode = -1;
    return -1;
  }
  *resultcode = (int)rescode;

  // Increment reference key count
  __IncrementKey(&thisprob->priv_data.refL0.hi, &thisprob->priv_data.refL0.lo, *keyscheckedP, thisprob->pub_data.contest);

  // Compare ref to core key incrementation
  if (((thisprob->priv_data.refL0.hi != thisprob->priv_data.rc5unitwork.L0.hi) || (thisprob->priv_data.refL0.lo != thisprob->priv_data.rc5unitwork.L0.lo))
      && (*resultcode != RESULT_FOUND) )
  {
    if (thisprob->priv_data.contestwork.crypto.iterations.hi == 0 &&
        thisprob->priv_data.contestwork.crypto.iterations.lo == 0x20000) /* test case */
    {
      Log("RC5 incrementation mismatch:\n"
          "Debug Information: %08x:%08x - %08x:%08x\n",
          thisprob->priv_data.rc5unitwork.L0.hi, thisprob->priv_data.rc5unitwork.L0.lo, thisprob->priv_data.refL0.hi, thisprob->priv_data.refL0.lo);
    }
    *resultcode = -1;
    return -1;
  };

  // Checks passed, increment keys done count.
  thisprob->priv_data.contestwork.crypto.keysdone.lo += *keyscheckedP;
  if (thisprob->priv_data.contestwork.crypto.keysdone.lo < *keyscheckedP)
    thisprob->priv_data.contestwork.crypto.keysdone.hi++;

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)  //(*keyscheckedP < keystocheck)
  {
    // found it!
    u32 keylo = thisprob->priv_data.contestwork.crypto.key.lo;
    thisprob->priv_data.contestwork.crypto.key.lo += thisprob->priv_data.contestwork.crypto.keysdone.lo;
    thisprob->priv_data.contestwork.crypto.key.hi += thisprob->priv_data.contestwork.crypto.keysdone.hi;
    if (thisprob->priv_data.contestwork.crypto.key.lo < keylo)
      thisprob->priv_data.contestwork.crypto.key.hi++; // wrap occured ?
    return RESULT_FOUND;
  }

  if ( ( thisprob->priv_data.contestwork.crypto.keysdone.hi > thisprob->priv_data.contestwork.crypto.iterations.hi ) ||
       ( ( thisprob->priv_data.contestwork.crypto.keysdone.hi == thisprob->priv_data.contestwork.crypto.iterations.hi ) &&
       ( thisprob->priv_data.contestwork.crypto.keysdone.lo >= thisprob->priv_data.contestwork.crypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  #ifdef STRESS_THREADS_AND_BUFFERS
  if (core_prob->priv_data.contestwork.crypto.key.hi ||
      core_prob->priv_data.contestwork.crypto.key.lo) /* not bench */
  {
    core_prob->priv_data.contestwork.crypto.key.hi = 0;
    core_prob->priv_data.contestwork.crypto.key.lo = 0;
    core_prob->priv_data.contestwork.crypto.keysdone.hi = 
      core_prob->priv_data.contestwork.crypto.iterations.hi;
    core_prob->priv_data.contestwork.crypto.keysdone.lo = 
      core_prob->priv_data.contestwork.crypto.iterations.lo;
    *resultcode = RESULT_NOTHING;
  }
  #endif

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING;    // Done with this round
}

/* ------------------------------------------------------------- */

static int Run_CSC(InternalProblem *thisprob, /* already validated */
                   u32 *iterationsP, int *resultcode)
{
#ifndef HAVE_CSC_CORES
  thisprob = thisprob;
  *iterationsP = 0;
  *resultcode = -1;
  return -1;
#else
  s32 rescode = (*(thisprob->pub_data.unit_func.gen))( 
                   &(thisprob->priv_data.rc5unitwork), 
                   iterationsP, thisprob->priv_data.core_membuffer );

  if (rescode < 0) /* "kiter" error */
  {
    *resultcode = -1;
    return -1;
  }
  *resultcode = (int)rescode;

  // Increment reference key count
  __IncrementKey (&thisprob->priv_data.refL0.hi, &thisprob->priv_data.refL0.lo, *iterationsP, thisprob->pub_data.contest);

  // Compare ref to core key incrementation
  if ((thisprob->priv_data.refL0.hi != thisprob->priv_data.rc5unitwork.L0.hi) || (thisprob->priv_data.refL0.lo != thisprob->priv_data.rc5unitwork.L0.lo))
  {
    if (thisprob->priv_data.contestwork.crypto.iterations.hi == 0 &&
        thisprob->priv_data.contestwork.crypto.iterations.lo == 0x20000) /* test case */
    {
      Log("CSC incrementation mismatch:\n"
          "expected %08x:%08x, got %08x:%08x\n",
          thisprob->priv_data.refL0.lo, thisprob->priv_data.refL0.hi, thisprob->priv_data.rc5unitwork.L0.lo, thisprob->priv_data.rc5unitwork.L0.hi );
    }
    *resultcode = -1;
    return -1;
  }

  // Checks passed, increment keys done count.
  thisprob->priv_data.contestwork.crypto.keysdone.lo += *iterationsP;
  if (thisprob->priv_data.contestwork.crypto.keysdone.lo < *iterationsP)
    thisprob->priv_data.contestwork.crypto.keysdone.hi++;

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)
  {
    u32 keylo = thisprob->priv_data.contestwork.crypto.key.lo;
    thisprob->priv_data.contestwork.crypto.key.lo += thisprob->priv_data.contestwork.crypto.keysdone.lo;
    thisprob->priv_data.contestwork.crypto.key.hi += thisprob->priv_data.contestwork.crypto.keysdone.hi;
    if (thisprob->priv_data.contestwork.crypto.key.lo < keylo)
      thisprob->priv_data.contestwork.crypto.key.hi++; // wrap occured ?
    return RESULT_FOUND;
  }

  if ( ( thisprob->priv_data.contestwork.crypto.keysdone.hi > thisprob->priv_data.contestwork.crypto.iterations.hi ) ||
       ( ( thisprob->priv_data.contestwork.crypto.keysdone.hi == thisprob->priv_data.contestwork.crypto.iterations.hi ) &&
       ( thisprob->priv_data.contestwork.crypto.keysdone.lo >= thisprob->priv_data.contestwork.crypto.iterations.lo ) ) )
  {
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }
  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING; // Done with this round
#endif
}

/* ------------------------------------------------------------- */

static int Run_DES(InternalProblem *thisprob, /* already validated */
                   u32 *iterationsP, int *resultcode)
{
#ifndef HAVE_DES_CORES
  thisprob = thisprob;
  *iterationsP = 0;  /* no keys done */
  *resultcode = -1; /* core error */
  return -1;
#else

  //iterationsP == in: suggested iterations, out: effective iterations
  u32 kiter = (*(thisprob->pub_data.unit_func.des))( &thisprob->priv_data.rc5unitwork, iterationsP, (char *)thisprob->priv_data.core_membuffer );

  __IncrementKey ( &thisprob->priv_data.refL0.hi, &thisprob->priv_data.refL0.lo, *iterationsP, thisprob->pub_data.contest);
  // Increment reference key count

  if (((thisprob->priv_data.refL0.hi != thisprob->priv_data.rc5unitwork.L0.hi) ||  // Compare ref to core
      (thisprob->priv_data.refL0.lo != thisprob->priv_data.rc5unitwork.L0.lo)) &&  // key incrementation
      (kiter == *iterationsP))
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #23: Please contact help@distributed.net\n"
        "Debug Information: %08x:%08x - %08x:%08x\n",
        thisprob->priv_data.rc5unitwork.L0.lo, thisprob->priv_data.rc5unitwork.L0.hi, thisprob->priv_data.refL0.lo, thisprob->priv_data.refL0.hi);
    #endif
    *resultcode = -1;
    return -1;
  };

  thisprob->priv_data.contestwork.crypto.keysdone.lo += kiter;
  if (thisprob->priv_data.contestwork.crypto.keysdone.lo < kiter)
    thisprob->priv_data.contestwork.crypto.keysdone.hi++;
    // Checks passed, increment keys done count.

  // Update data returned to caller
  if (kiter < *iterationsP)
  {
    // found it!
    u32 keylo = thisprob->priv_data.contestwork.crypto.key.lo;
    thisprob->priv_data.contestwork.crypto.key.lo += thisprob->priv_data.contestwork.crypto.keysdone.lo;
    thisprob->priv_data.contestwork.crypto.key.hi += thisprob->priv_data.contestwork.crypto.keysdone.hi;
    if (thisprob->priv_data.contestwork.crypto.key.lo < keylo)
      thisprob->priv_data.contestwork.crypto.key.hi++; // wrap occured ?
    *resultcode = RESULT_FOUND;
    return RESULT_FOUND;
  }
  else if (kiter != *iterationsP)
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #24: Please contact help@distributed.net\n"
        "Debug Information: k: %x t: %x\n"
        "Debug Information: %08x:%08x - %08x:%08x\n", kiter, *iterationsP,
        thisprob->priv_data.rc5unitwork.L0.lo, thisprob->priv_data.rc5unitwork.L0.hi, thisprob->priv_data.refL0.lo, thisprob->priv_data.refL0.hi);
    #endif
    *resultcode = -1; /* core error */
    return -1;
  };

  if ( ( thisprob->priv_data.contestwork.crypto.keysdone.hi > thisprob->priv_data.contestwork.crypto.iterations.hi ) ||
     ( ( thisprob->priv_data.contestwork.crypto.keysdone.hi == thisprob->priv_data.contestwork.crypto.iterations.hi ) &&
     ( thisprob->priv_data.contestwork.crypto.keysdone.lo >= thisprob->priv_data.contestwork.crypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING; // Done with this round
#endif /* #ifdef HAVE_DES_CORES */
}

/* ------------------------------------------------------------- */

static int Run_OGR( InternalProblem *thisprob, /* already validated */
                    u32 *iterationsP, int *resultcode)
{
#if !defined(HAVE_OGR_CORES)
  thisprob = thisprob;
  iterationsP = iterationsP;
#else
  int r, nodes;

  nodes = (int)(*iterationsP);
  r = (thisprob->pub_data.unit_func.ogr)->cycle(
                          thisprob->priv_data.core_membuffer, 
                          &nodes,
                          thisprob->pub_data.cruncher_is_time_constrained);
  *iterationsP = (u32)nodes;

  u32 newnodeslo = thisprob->priv_data.contestwork.ogr.nodes.lo + nodes;
  if (newnodeslo < thisprob->priv_data.contestwork.ogr.nodes.lo) {
    thisprob->priv_data.contestwork.ogr.nodes.hi++;
  }
  thisprob->priv_data.contestwork.ogr.nodes.lo = newnodeslo;

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
      if ((thisprob->pub_data.unit_func.ogr)->getresult(thisprob->priv_data.core_membuffer, &thisprob->priv_data.contestwork.ogr.workstub, sizeof(WorkStub)) == CORE_S_OK)
      {
        //Log("OGR Success!\n");
        thisprob->priv_data.contestwork.ogr.workstub.stub.length =
                  (u16)(thisprob->priv_data.contestwork.ogr.workstub.worklength);
        *resultcode = RESULT_FOUND;
        return RESULT_FOUND;
      }
      break;
    }
  }
  /* Something bad happened */
#endif
 *resultcode = -1; /* this will cause the problem to be discarded */
 return -1;
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
    ** if post-LoadState() initialization  failed, but can be deferred, Run_XXX
    ** may choose to return -1, but keep priv_data.last_resultcode at 
    ** RESULT_WORKING.
    */

    retcode         = -1;
    iterations      = core_prob->pub_data.tslice;
    switch (core_prob->pub_data.contest)
    {
      case RC5: retcode = Run_RC5( core_prob, &iterations, &last_resultcode );
                break;
      case DES: retcode = Run_DES( core_prob, &iterations, &last_resultcode );
                break;
      case OGR: retcode = Run_OGR( core_prob, &iterations, &last_resultcode );
                break;
      case CSC: retcode = Run_CSC( core_prob, &iterations, &last_resultcode );
                break;
      default:  retcode = 0; last_resultcode = -1;
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
  prob_index = prob_index; /* possibly unused */

  #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
  if (prob_index == 1 && /* thread number reserved for x86 card */
     contest_i != RC5 && /* RISC OS x86 thread only supports RC5 */
     GetNumberOfDetectedProcessors() > 1) /* have x86 card */
    return 0;
  #endif
  #if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_MACOS) || \
      (CLIENT_OS == OS_WIN16) /*|| (CLIENT_OS == OS_RISCOS)*/
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
  if (contest_i == OGR /* && prob_index >= 0 */) /* crunchers only */
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
          #elif (CLIENT_CPU == CPU_POWERPC)
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
    case RC5:
    {
      return 1;
    }
    case DES:
    {
      #ifdef HAVE_DES_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case OGR:
    {
      #ifdef HAVE_OGR_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case CSC:
    {
      #ifdef HAVE_CSC_CORES
      return 1;
      #else
      return 0;
      #endif
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
    if (numstr_style == 1 || numstr_style == 2) 
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
    case RC5:
    case DES:
    case CSC:
    {
      if (work->crypto.keysdone.lo || work->crypto.keysdone.hi)
      {
        permille = 1000;
        if ((work->crypto.keysdone.hi < work->crypto.iterations.hi) ||
            ((work->crypto.keysdone.hi== work->crypto.iterations.hi) &&
            (work->crypto.keysdone.lo < work->crypto.iterations.lo)))
        {
          u32 hi,lo;
          __u64mul( work->crypto.keysdone.hi, work->crypto.keysdone.lo,
                    0, 1000, &hi, &lo );   
          __u64div( hi, lo, work->crypto.iterations.hi,
                            work->crypto.iterations.lo, &hi, &lo, 0, 0);
          if (lo > 1000)
            lo = 1000;
          permille = lo;   
        }
      }
    }
    break;
#ifdef HAVE_OGR_CORES
    case OGR:
    if (work->ogr.workstub.worklength > (u32)work->ogr.workstub.stub.length)
    {
      // This is just a quick&dirty calculation that resembles progress.
      permille = work->ogr.workstub.stub.diffs[work->ogr.workstub.stub.length]*10
                +work->ogr.workstub.stub.diffs[work->ogr.workstub.stub.length+1]/10;
    }
    break;
#endif
    default:
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
      case RC5:
      case DES:
      case CSC:
      { 
        u32 tcounthi = work->crypto.iterations.hi;
        u32 tcountlo = work->crypto.iterations.lo;
        if (contestid == DES)
        {
          tcounthi <<= 1; tcounthi |= (tcountlo >> 31); tcountlo <<= 1; 
        }
        /* note that we return zero for test packets */
        units = 100 * ((tcountlo >> 28)+(tcounthi << 4)); 

        // if this is a completed packet and not a test one (other random
        // packets are ok), then remember its prefix for random prefix.
        if (contestid == RC5 && rescode != RESULT_WORKING &&
            ((tcounthi != 0) || (tcounthi == 0 && tcountlo != 0x00100000UL)))
        {   
          last_rc5_prefix = ((work->crypto.key.hi >> 24) & 0xFF);
        }
      }
      break;
#ifdef HAVE_OGR_CORES
      case OGR:
      {
        if (swucount && rescode != RESULT_WORKING)
        {
          u32 hi, tcounthi = work->ogr.nodes.hi;
          u32 lo, tcountlo = work->ogr.nodes.lo;
          /* ogr stats unit is Gnodes */
          __u64div( tcounthi, tcountlo, 0, 1000000000ul, 0, &hi, 0, &lo);
          units = (unsigned int)(hi * 100)+(lo / 10000000ul);
        }
      } /* OGR */      
      break;
#endif /* HAVE_OGR_CORES */
      default:
        break;  
    } /* switch() */

    if (swucount)
      *swucount = units;
  } /* if (swucount) */

  return rescode;
}

// thisprob may be 0, gracefully exits
// info may not be zero, it must be a valid address of a ProblemInfo struct
// see problem.h for flag values
int ProblemGetInfo(void *__thisprob, ProblemInfo *info, u32 flags)
{
  int rescode = -1;
  ContestWork work;
  u32 contestid;
  InternalProblem *thisprob = __pick_probptr(__thisprob, PICKPROB_MAIN);
  
  if (thisprob)
  {  
    rescode = ProblemRetrieveState( thisprob, &work, &contestid, 0, 0 );
  }
  if (rescode >= 0)
  {
    u32 e_sec = 0, e_usec = 0;

    info->is_test_packet = contestid == RC5 && 
                           work.crypto.iterations.lo == 0x00100000 &&
                           work.crypto.iterations.hi == 0;
    //info->stats_units_are_integer = (contestid != OGR);
    info->show_exact_iterations_done = (contestid == OGR);

    if (flags & (P_INFO_E_TIME | P_INFO_RATE | P_INFO_RATEBUF))
    {
      if (thisprob->pub_data.elapsed_time_sec != 0xfffffffful)
      {
        /* problem finished, elapsed time has already calculated by Run() */
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
        unsigned long rate2wuspeed = 0;

        switch (contestid)
        {
          case RC5:
          case DES:
          case CSC:
          { 
            unsigned int units, twoxx;
            rate2wuspeed = 1UL<<28;

            ccounthi = thisprob->pub_data.startkeys.hi;
            ccountlo = thisprob->pub_data.startkeys.lo;
            tcounthi = work.crypto.iterations.hi;
            tcountlo = work.crypto.iterations.lo;
            dcounthi = work.crypto.keysdone.hi;
            dcountlo = work.crypto.keysdone.lo;
            if (contestid == DES)
            {
              tcounthi <<= 1; tcounthi |= (tcountlo >> 31); tcountlo <<= 1; 
              dcounthi <<= 1; dcounthi |= (dcountlo >> 31); dcountlo <<= 1; 
              ccounthi <<= 1; ccounthi |= (ccountlo >> 31); ccountlo <<= 1;
            }
            /* current = donecount - startpos */
            ccountlo = dcountlo - ccountlo;
            ccounthi = dcounthi - ccounthi;
            if (ccountlo > dcountlo)
              ccounthi--;

            units = ((tcountlo >> 28)+(tcounthi << 4)); 
            twoxx = 28;
            if (!units) /* less than 2^28 packet (eg test) */
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
              char scratch[32];
              sprintf( scratch, "%08lX:%08lX:%u*2^%u", 
                       (unsigned long) ( work.crypto.key.hi ),
                       (unsigned long) ( work.crypto.key.lo ),
                       units, twoxx );
              strncpy( info->sigbuf, scratch, info->sigbufsz );
              info->sigbuf[info->sigbufsz-1] = '\0';
            }
            if (flags & P_INFO_CWPBUF)
            {
              // ToDo: do something different here - any ideas for a cwp for crypto packets?
              char scratch[32];
              sprintf( scratch, "%08lX:%08lX:%u*2^%u", 
                       (unsigned long) ( work.crypto.key.hi ),
                       (unsigned long) ( work.crypto.key.lo ),
                       units, twoxx );
              strncpy( info->cwpbuf, scratch, info->cwpbufsz );
              info->cwpbuf[info->cwpbufsz-1] = '\0';
            }
            if ((flags & P_INFO_SWUCOUNT) && (tcounthi || tcountlo)) /* only if finished */
            {
              /* note that we return zero for test packets */
              info->swucount = ((tcountlo >> 28)+(tcounthi << 4))*100;
            }
          } /* case: crypto */
          break;
#ifdef HAVE_OGR_CORES
          case OGR:
          {
            rate2wuspeed = 0;
            dcounthi = work.ogr.nodes.hi;
            dcountlo = work.ogr.nodes.lo;
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
              ogr_stubstr_r( &work.ogr.workstub.stub, info->sigbuf, info->sigbufsz, 0);
            }
            if (flags & P_INFO_CWPBUF)
            {
              ogr_stubstr_r( &work.ogr.workstub.stub, info->cwpbuf, info->cwpbufsz, work.ogr.workstub.worklength);
            }
            if ((flags & P_INFO_SWUCOUNT) && (tcounthi || tcountlo)) /* only if finished */
            {
              /* ogr stats unit is Gnodes */
              __u64div( tcounthi, tcountlo, 0, 1000000000ul, 0, &hi, 0, &lo);
              info->swucount = (hi * 100)+(lo / 10000000ul);
            }
            if (info->permille_only_if_exact)
            {
              if (flags & P_INFO_C_PERMIL)
                info->c_permille = 0;
              if (flags & P_INFO_S_PERMIL)
                info->s_permille = 0;
            }
          } /* OGR */      
          break;
#endif /* HAVE_OGR_CORES */
          default:
          break;  
        } /* switch() */

        if (flags & (P_INFO_RATE | P_INFO_RATEBUF))
        {
          char * _ratebuf = 0;
          u32 _ratebufsz = 0;
          
          if(flags & P_INFO_RATEBUF)
          {
            _ratebuf = info->ratebuf;
            _ratebufsz = info->ratebufsz;
          }
          ProblemComputeRate( contestid, e_sec, e_usec, ccounthi, ccountlo,
                              &hi, &lo, _ratebuf, _ratebufsz );
          if (rate2wuspeed && lo)
          {
            CliSetContestWorkUnitSpeed( contestid, (unsigned int)
                                        ((1+rate2wuspeed) / lo) );     
          }
          if (flags & P_INFO_RATE)
          {
            info->ratehi = hi;
            info->ratelo = lo;
          }
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
  return rescode;
}
