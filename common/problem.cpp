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
return "@(#)$Id: problem.cpp,v 1.108.2.87 2001/01/03 19:38:27 cyp Exp $"; }

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
#include "sleepdef.h" //sleep()
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
    char __core_membuffer_space[(MAX_MEM_REQUIRED_BY_CORE+(1UL<<CORE_MEM_ALIGNMENT)-1)];
    void *core_membuffer; /* aligned pointer to __core_membuffer_space */
    /* --------------------------------------------------------------- */
    u32 loadtime_sec, loadtime_usec; /* LoadState() time */
    int last_resultcode; /* the rescode the last time contestwork was stable */
    int started;
    int initialized;
    unsigned int threadindex; /* 0-n (globally unique identifier) */
    volatile int running; /* RetrieveState(,,purge) has to wait while Run()ning */
  } priv_data;
} InternalProblem;

#ifndef MIPSpro
#pragma pack()
#endif

static InternalProblem *__validate_probptr(void *thisprob)
{
   /* nothing yet */
  return (InternalProblem *)thisprob;
}

/* ------------------------------------------------------------------- */

void ProblemFree(void *__thisprob)
{
  InternalProblem *thisprob = __validate_probptr(__thisprob);
  if (thisprob)
  {
    memset( thisprob, 0, sizeof(InternalProblem) );
    __problem_counter--;
    free((void *)thisprob);
  }
  return;
}

int ProblemIsInitialized(void *__thisprob)
{ 
  InternalProblem *thisprob = __validate_probptr(__thisprob);
  if (thisprob)
  {
    int init = thisprob->priv_data.initialized;
    int rescode = thisprob->priv_data.last_resultcode;
    if (init)
    {
      if (rescode <= 0) /* <0 = error, 0 = RESULT_WORKING */
        return -1;
      return rescode; /* 1==RESULT_NOTHING, 2==RESULT_FOUND */
    } 
  }
  return 0;
}

unsigned int ProblemGetSize(void)
{ /* needed by IPC/shmem */
  return sizeof(InternalProblem);
}  

Problem *ProblemAlloc(void)
{
  char *p; unsigned long ww;
  InternalProblem *thisprob = (InternalProblem *)0;
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
    thisprob = (InternalProblem *)malloc(sizeof(InternalProblem));
    if (!thisprob)
    {
      Log("Insufficient memory to allocate problem data\n");
      err = 1;
    }
  }    
    
  if (thisprob && !err)
  {
    p = (char *)&(thisprob->priv_data.rc5unitwork);
    ww = ((unsigned long)p);

    #if (CLIENT_CPU == CPU_ALPHA) /* sizeof(long) can be either 4 or 8 */
    ww &= 0x7; /* (sizeof(longword)-1); */
    #else
    ww &= (sizeof(int)-1); /* int alignment */
    #endif
    if (ww)
    {
      Log("priv_data.rc5unitwork for problem %d is misaligned!\n", __problem_counter);
      err = 1;
    }
  }      

  if (thisprob && !err)
  {
    memset( thisprob, 0, sizeof(InternalProblem) );
    thisprob->priv_data.threadindex = __problem_counter++;

    //align core_membuffer to 16byte boundary
    p = &(thisprob->priv_data.__core_membuffer_space[0]);
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    thisprob->priv_data.core_membuffer = p;
  }
  
  if (thisprob && err)
  {
    free((void *)thisprob);
    thisprob = (InternalProblem *)0;
  }
  return (Problem *)thisprob;
}

/* ------------------------------------------------------------------- */

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

/* forward reference */
static unsigned int __compute_permille(unsigned int cont_i, const ContestWork *work);

/* LoadState() and RetrieveState() work in pairs. A LoadState() without
   a previous RetrieveState(,,purge) will fail, and vice-versa.
*/
int ProblemLoadState( void *__thisprob,
                      const ContestWork * work, unsigned int contestid,
                      u32 _iterations, int expected_cputype,
                      int expected_corenum, int expected_os,
                      int expected_buildfrac )
{
  ContestWork for_magic;
  int genned_random = 0, genned_benchmark = 0;
  InternalProblem *thisprob = __validate_probptr(__thisprob);
  struct selcore selinfo; int coresel;

  if (!thisprob)
  {
    return -1;
  }
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
  thisprob->pub_data.is_random = genned_random;
  thisprob->pub_data.is_benchmark = genned_benchmark;

  thisprob->pub_data.coresel = coresel;
  thisprob->pub_data.client_cpu = selinfo.client_cpu;
  thisprob->pub_data.pipeline_count = selinfo.pipeline_count;
  thisprob->pub_data.use_generic_proto = selinfo.use_generic_proto;
  thisprob->pub_data.cruncher_is_asynchronous = selinfo.cruncher_is_asynchronous;
  memcpy( (void *)&(thisprob->pub_data.unit_func), 
          &selinfo.unit_func, sizeof(thisprob->pub_data.unit_func));

  //----------------------------------------------------------------

  switch (thisprob->pub_data.contest)
  {
    case RC5:
    if (!thisprob->pub_data.is_random &&
       (work->crypto.iterations.hi || work->crypto.iterations.lo >= (1L<<28)))
    {
      last_rc5_prefix = (int)(work->crypto.key.hi >> 24);
    }    
    /* fallthrough */
    #if defined(HAVE_DES_CORES)
    case DES:
    #endif
    #if defined(HAVE_CSC_CORES)
    case CSC: // HAVE_CSC_CORES
    #endif
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
      break;
    }
    #if defined(HAVE_OGR_CORES)
    case OGR:
    {
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
      //thisprob->pub_data.unit_func.ogr = [xxx_]ogr_get_dispatch_table(); was done by selcore
      int r = (thisprob->pub_data.unit_func.ogr)->init();
      if (r != CORE_S_OK)
        return -1;
      r = (thisprob->pub_data.unit_func.ogr)->create(&thisprob->priv_data.contestwork.ogr.workstub,
                      sizeof(WorkStub), thisprob->priv_data.core_membuffer, MAX_MEM_REQUIRED_BY_CORE);
      if (r != CORE_S_OK)
        return -1;
      if (thisprob->priv_data.contestwork.ogr.workstub.worklength > (u32)thisprob->priv_data.contestwork.ogr.workstub.stub.length)
      {
        thisprob->pub_data.startkeys.hi = thisprob->priv_data.contestwork.ogr.nodes.hi;
        thisprob->pub_data.startkeys.lo = thisprob->priv_data.contestwork.ogr.nodes.lo;
        thisprob->pub_data.startpermille = __compute_permille( thisprob->pub_data.contest, &thisprob->priv_data.contestwork );
      }
      break;
    }
    #endif
    default:
      return -1;
  }

  //---------------------------------------------------------------

  {
    // set timers
    thisprob->priv_data.loadtime_sec = 0;
    struct timeval tv;
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



/* ------------------------------------------------------------------- */

/* LoadState() and RetrieveState() work in pairs. A LoadState() without
   a previous RetrieveState(,,purge) will fail, and vice-versa.
*/
int ProblemRetrieveState( void *__thisprob,
                          ContestWork * work, unsigned int *contestid, 
                          int dopurge, int dontwait )
{
  InternalProblem *thisprob = __validate_probptr(__thisprob);
  if (!thisprob)
  {
    return -1;
  }    
  if (!thisprob->priv_data.initialized)
  {
    //LogScreen("ProblemRetrieveState() without preceding LoadState()\n");
    return -1;
  }    
  if (work) // store back the state information
  {
    switch (thisprob->pub_data.contest) {
      case RC5:
      case DES:
      case CSC:
        // nothing special needs to be done here
        break;
      #if defined(HAVE_OGR_CORES)
      case OGR:
        (thisprob->pub_data.unit_func.ogr)->getresult(thisprob->priv_data.core_membuffer, &thisprob->priv_data.contestwork.ogr.workstub, sizeof(WorkStub));
        break;
      #endif
    }
    memcpy( (void *)work, (void *)&thisprob->priv_data.contestwork, sizeof(ContestWork));
  }
  if (contestid)
    *contestid = thisprob->pub_data.contest;
  if (dopurge)
  {
    thisprob->priv_data.initialized = 0;
    if (!dontwait) /* normal state is to wait. But we can't wait when aborting */
    {
      while (thisprob->priv_data.running) /* need to guarantee that no Run() will occur on a */
      {
        usleep(1000); /* purged problem. */
      }    
    }
    loaded_problems[thisprob->pub_data.contest]--;       /* per contest */  
    loaded_problems[CONTEST_COUNT]--; /* total */
  }
  if (thisprob->priv_data.last_resultcode < 0)
  {
    //LogScreen("last resultcode = %d\n",thisprob->priv_data.last_resultcode);
    return -1;
  }    
  return ( thisprob->priv_data.last_resultcode );
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

  if (*iterationsP > 0x100000UL && !thisprob->pub_data.is_benchmark)
    *iterationsP = 0x100000UL;

  nodes = (int)(*iterationsP);
  r = (thisprob->pub_data.unit_func.ogr)->cycle(thisprob->priv_data.core_membuffer, &nodes);
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

/* ---------------------------------------------------------------- */

int ProblemRun(void *__thisprob) /* returns RESULT_*  or -1 */
{
  static volatile int s_using_ptime = -1;
  struct timeval tv;
  int retcode, core_resultcode, using_ptime;
  u32 iterations, runstart_secs, runstart_usecs;
  InternalProblem *thisprob = __validate_probptr(__thisprob);

  if (!thisprob)
  {
    return -1;
  }

  thisprob->pub_data.last_runtime_is_invalid = 1; /* haven't changed runtime fields yet */

  if ( !thisprob->priv_data.initialized )
  {
    return ( -1 );
  }
  if ((++thisprob->priv_data.running) > 1)
  {
    --thisprob->priv_data.running;
    return -1;
  }

#ifdef STRESS_THREADS_AND_BUFFERS
  if (thisprob->pub_data.contest == RC5 && !thisprob->priv_data.started)
  {
    thisprob->priv_data.contestwork.crypto.key.hi = thisprob->priv_data.contestwork.crypto.key.lo = 0;
    thisprob->priv_data.contestwork.crypto.keysdone.hi = thisprob->priv_data.contestwork.crypto.iterations.hi;
    thisprob->priv_data.contestwork.crypto.keysdone.lo = thisprob->priv_data.contestwork.crypto.iterations.lo;
    thisprob->pub_data.runtime_usec = 1; /* ~1Tkeys for a 2^20 packet */
    thisprob->pub_data.elapsed_time_usec = 1;
    thisprob->priv_data.last_resultcode = RESULT_NOTHING;
    thisprob->priv_data.started = 1;
  }
#endif

  if ( thisprob->priv_data.last_resultcode != RESULT_WORKING ) /* _FOUND, _NOTHING or -1 */
  {
    thisprob->priv_data.running--;
    return ( thisprob->priv_data.last_resultcode );
  }

  /*
    On return from the Run_XXX thisprob->priv_data.contestwork must be in a state that we
    can put away to disk - that is, do not expect the loader (probfill
    et al) to fiddle with iterations or key or whatever.

    The Run_XXX functions do *not* update problem.thisprob->priv_data.last_resultcode, they use
    core_resultcode instead. This is so that members of the problem object
    that are updated after the resultcode has been set will not be out of
    sync when the main thread gets it with RetrieveState().

    note: although the value returned by Run_XXX is usually the same as
    the core_resultcode it is not always the case. For instance, if
    post-LoadState() initialization  failed, but can be deferred, Run_XXX
    may choose to return -1, but keep core_resultcode at RESULT_WORKING.
  */

  thisprob->priv_data.started = 1;
  thisprob->pub_data.last_runtime_usec = thisprob->pub_data.last_runtime_sec = 0;
  runstart_secs = 0xfffffffful;
  using_ptime = s_using_ptime;
  if (using_ptime)
  {
    if (CliGetThreadUserTime(&tv) != 0)
      using_ptime = 0;
    else
      runstart_secs = 0;
  }
  if (!using_ptime)
  {
    runstart_secs = 0;
    if (CliGetMonotonicClock(&tv) != 0)
    {
      if (CliGetMonotonicClock(&tv) != 0)
        runstart_secs = 0xfffffffful;
    }
  }
  runstart_usecs = 0; /* shaddup compiler */
  if (runstart_secs == 0)
  {
    runstart_secs = tv.tv_sec;
    runstart_usecs = tv.tv_usec;
  }
  iterations = thisprob->pub_data.tslice;
  core_resultcode = thisprob->priv_data.last_resultcode;
  retcode = -1;

  switch (thisprob->pub_data.contest)
  {
    case RC5: retcode = Run_RC5( thisprob, &iterations, &core_resultcode );
              break;
    case DES: retcode = Run_DES( thisprob, &iterations, &core_resultcode );
              break;
    case OGR: retcode = Run_OGR( thisprob, &iterations, &core_resultcode );
              break;
    case CSC: retcode = Run_CSC( thisprob, &iterations, &core_resultcode );
              break;
    default: retcode = core_resultcode = thisprob->priv_data.last_resultcode = -1;
       break;
  }

  if (retcode < 0) /* don't touch thisprob->pub_data.tslice or runtime as long as < 0!!! */
  {
    thisprob->priv_data.running--;
    return -1;
  }
  if (!thisprob->priv_data.started || !thisprob->priv_data.initialized) /* RetrieveState(,,purge) has been called */
  {
    core_resultcode = -1; // "Discarded (core error)": discard the purged block
  }

  thisprob->pub_data.core_run_count++;
  __compute_run_times( thisprob, runstart_secs, runstart_usecs, 
                       &thisprob->priv_data.loadtime_sec, &thisprob->priv_data.loadtime_usec,
                       using_ptime, &s_using_ptime, core_resultcode );
  thisprob->pub_data.tslice = iterations;
  thisprob->priv_data.last_resultcode = core_resultcode;
  thisprob->priv_data.running--;
  return thisprob->priv_data.last_resultcode;
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
     thisprob->priv_data.started dropping packets, clients disconnected, the profiler
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

static char *__u64stringify(char *buffer, unsigned int buflen, u32 hi, u32 lo,
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
      char magna_tab[]={0,'k','M','G','T','P','E','Z','Y'};
      unsigned int magna = 0, len = 0;

      #if (ULONG_MAX > 0xfffffffful) /* 64+ bit */
      {
        len = sprintf( numbuf, "%lu", (((unsigned long)hi)<<32UL)+((unsigned long)lo);
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
    const char *unitname = ""; 
    switch (contestid)
    {
      case RC5:
      case DES:
      case CSC: unitname = "keys"; break;
      case OGR: unitname = "nodes"; break;
      default:  break;
    }
    __u64stringify( ratebuf, ratebufsz, hi, lo, 2, unitname );
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


int ProblemGetSWUCount( const ContestWork *work,
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


/* more info than you ever wanted. :) any/all params can be NULL/0
 * tcount = total_number_of_iterations_to_do
 * ccount = number_of_iterations_done_thistime.
 * dcount = number_of_iterations_done_ever
 * counts are unbiased (adjustment for DES etc already done)
 * numstring_style: -1=unformatted, 0=commas, 
 * 1=0+space between magna and number (or at end), 2=1+"nodes"/"keys"
*/
int ProblemGetInfo(void *__thisprob,
                   unsigned int *cont_id, const char **cont_name, 
                   u32 *elapsed_secsP, u32 *elapsed_usecsP, 
                   unsigned int *swucount, int numstring_style,
                   const char **unit_name, 
                   unsigned int *c_permille, unsigned int *s_permille,
                   int permille_only_if_exact,
                   char *sigbuf, unsigned int sigbufsz,
                   u32 *ratehi, u32 *ratelo, 
                   char *ratebuf, unsigned int ratebufsz,
                   u32 *ubtcounthi, u32 *ubtcountlo, 
                   char *tcountbuf, unsigned int tcountbufsz,
                   u32 *ubccounthi, u32 *ubccountlo, 
                   char *ccountbuf, unsigned int ccountbufsz,
                   u32 *ubdcounthi, u32 *ubdcountlo,
                   char *dcountbuf, unsigned int dcountbufsz)
{
  int rescode = -1;
  InternalProblem *thisprob = __validate_probptr(__thisprob);
  permille_only_if_exact = permille_only_if_exact; /* possibly unused */

  if (thisprob)
  {  
    if (thisprob->priv_data.initialized)
      rescode = thisprob->priv_data.last_resultcode;
  }
  if (rescode >= 0)
  {
    u32 e_sec = 0, e_usec = 0;

    if (cont_id)
    {
      *cont_id = thisprob->pub_data.contest;
    }
    if (cont_name)
    {
      switch (thisprob->pub_data.contest)
      {
        case RC5: *cont_name = "RC5"; break;
        case DES: *cont_name = "DES"; break;
        case OGR: *cont_name = "OGR"; break;
        case CSC: *cont_name = "CSC"; break;
        default:  *cont_name = "???"; break;
      }
    }
    if (unit_name)
    {
      switch (thisprob->pub_data.contest)
      {
        case RC5:
        case DES:
        case CSC: *unit_name = "keys"; break; 
        case OGR: *unit_name = "nodes"; break;
        default:  *unit_name = "???"; break;
      }
    }
    if (sigbuf)
    {
      if (sigbufsz)
        *sigbuf = '\0';
      if (sigbufsz < 2)
        sigbuf = (char *)0;
    }
    if (ratebuf)
    {
      if (ratebufsz)
        *ratebuf = '\0';
      if (ratebufsz < 2)
        ratebuf = (char *)0;
    }
    if (tcountbuf)
    {
      if (tcountbufsz)
        *tcountbuf = '\0';
      if (tcountbufsz < 2)
        tcountbuf = (char *)0;
    }
    if (ccountbuf)
    {
      if (ccountbufsz)
        *ccountbuf = '\0';
      if (ccountbufsz < 2)
        ccountbuf = (char *)0;
    }
    if (elapsed_secsP || elapsed_usecsP || ratehi || ratelo || ratebuf)
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
      if (elapsed_secsP)
        *elapsed_secsP = e_sec;
      if (elapsed_usecsP)
        *elapsed_usecsP = e_usec;
      if (thisprob->pub_data.is_benchmark && (thisprob->pub_data.runtime_sec || thisprob->pub_data.runtime_usec))
      {
        e_sec = thisprob->pub_data.runtime_sec;
        e_usec = thisprob->pub_data.runtime_usec;
      }
    } /* if (elapsed || rate || ratebuf) */    
    if ( sigbuf     || c_permille || s_permille ||
         ratehi     || ratelo     || ratebuf   ||  
         ubtcounthi || ubtcountlo || tcountbuf ||
         ubccounthi || ubccountlo || ccountbuf ||
         ubdcounthi || ubdcountlo || dcountbuf )
    { 
      ContestWork work;
      unsigned int contestid = 0;
      int rescode = ProblemRetrieveState( thisprob, &work, &contestid, 0, 0 );

      if (rescode >= 0) /* hmm! */
      {
        u32 hi, lo;
        u32 tcounthi=0, tcountlo=0; /*total 'iter' (n/a if not finished)*/
        u32 ccounthi=0, ccountlo=0; /*'iter' done (so far, this start) */
        u32 dcounthi=0, dcountlo=0; /*'iter' done (so far, all starts) */
        const char *numstr_suffix = "";
        unsigned long rate2wuspeed = 0;

        switch (contestid)
        {
          case RC5:
          case DES:
          case CSC:
          { 
            unsigned int units, twoxx;
            numstr_suffix = "keys";
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
            if (sigbuf)
            {
              char scratch[32];
              sprintf( scratch, "%08lX:%08lX:%u*2^%u", 
                       (unsigned long) ( work.crypto.key.hi ),
                       (unsigned long) ( work.crypto.key.lo ),
                       units, twoxx );
              strncpy( sigbuf, scratch, sigbufsz );
              sigbuf[sigbufsz-1] = '\0';
            }
            if (swucount && (tcounthi || tcountlo)) /* only if finished */
            {
              /* note that we return zero for test packets */
              *swucount = ((tcountlo >> 28)+(tcounthi << 4))*100;
            }
          } /* case: crypto */
          break;
#ifdef HAVE_OGR_CORES
          case OGR:
          {
            rate2wuspeed = 0;
            numstr_suffix = "nodes";
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

            if (sigbuf)
            {
              ogr_stubstr_r( &work.ogr.workstub.stub, sigbuf, sigbufsz );
            }
            if (swucount && (tcounthi || tcountlo)) /* only if finished */
            {
              /* ogr stats unit is Gnodes */
              __u64div( tcounthi, tcountlo, 0, 1000000000ul, 0, &hi, 0, &lo);
              *swucount = (hi * 100)+(lo / 10000000ul);
            }
            if (permille_only_if_exact)
            {
              if (c_permille) *c_permille = 0; c_permille = (unsigned int *)0;
              if (s_permille) *s_permille = 0; s_permille = (unsigned int *)0; 
            }
          } /* OGR */      
          break;
#endif /* HAVE_OGR_CORES */
          default:
          break;  
        } /* switch() */

        if (tcountbuf && (tcounthi || tcountlo)) /* only if finished */
        {
          __u64stringify( tcountbuf, tcountbufsz, tcounthi, tcountlo, 
                          numstring_style, numstr_suffix);
        }
        if (ccountbuf) /* count - this time */
        {
          __u64stringify( ccountbuf, ccountbufsz, ccounthi, ccountlo,
                          numstring_style, numstr_suffix);
        }
        if (dcountbuf) /* count - all times */
        {
          __u64stringify( dcountbuf, dcountbufsz, dcounthi, dcountlo,
                          numstring_style, numstr_suffix);
        }
        if (ratehi || ratelo || ratebuf)
        {
          ProblemComputeRate( contestid, e_sec, e_usec, ccounthi, ccountlo,
                              &hi, &lo, ratebuf, ratebufsz );
          if (rate2wuspeed && lo)
          {
            CliSetContestWorkUnitSpeed( contestid, (unsigned int)
                                        ((1+rate2wuspeed) / lo) );     
          }
          if (ratehi) *ratehi = hi;
          if (ratelo) *ratelo = lo;
        }
        if (c_permille)
        {
          if (!thisprob->priv_data.started)
            *c_permille = thisprob->pub_data.startpermille;
          else if (rescode != RESULT_WORKING) /* _FOUND or _NOTHING */
            *c_permille = 1000;
          else
            *c_permille = __compute_permille(contestid, &work); 
        }
        if (s_permille)
          *s_permille = thisprob->pub_data.startpermille;
        if (ubdcounthi)
          *ubdcounthi = dcounthi;
        if (ubdcountlo)
          *ubdcountlo = dcountlo;
        if (ubccounthi)
          *ubccounthi = ccounthi;
        if (ubccountlo)
          *ubccountlo = ccountlo;
        if (ubtcounthi)
          *ubtcounthi = tcounthi;
        if (ubtcountlo)
          *ubtcountlo = tcountlo;
      } /* if (rescode >= 0) */
    } /* if (sigbuf || ... ) */
  } /* if (rescode >= 0) */
  return rescode;
}

