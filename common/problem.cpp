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
return "@(#)$Id: problem.cpp,v 1.108.2.77 2000/11/01 19:58:18 cyp Exp $"; }

/* ------------------------------------------------------------- */

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
   pipeline_counts currently in use (1,2,3,4[,6?])
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

Problem::~Problem()
{
  __problem_counter--;
  initialized = started = 0;
  running = 0;
}

/* ------------------------------------------------------------------- */

Problem::Problem(void)
{
  threadindex = __problem_counter++;
  initialized = 0;
  started = 0;
  running = 0;

  //align core_membuffer to 16byte boundary
  {
    char *p = &__core_membuffer_space[0];
    while ((((unsigned long)p) & ((1UL << CORE_MEM_ALIGNMENT) - 1)) != 0)
      p++;
    core_membuffer = p;
  }

  {
    unsigned int sz = sizeof(int);
    if (sz < sizeof(u32)) /* need to do it this way to suppress compiler warnings. */
    {
      LogScreen("FATAL: sizeof(int) < sizeof(u32)\n");
      //#error "everything assumes a 32bit CPU..."
      RaiseExitRequestTrigger();
      return;
    }
    else
    {
      /*
       this next part is essential for alpha, but is probably beneficial to
       all platforms. If it fails for your os/cpu, we may need to redesign
       how objects are allocated/how rc5unitwork is addressed, so let me know.
                                                         -cyp Jun 14 1999
      */
      RC5UnitWork *w = &rc5unitwork;
      unsigned long ww = ((unsigned long)w);

      #if (CLIENT_CPU == CPU_ALPHA) /* sizeof(long) can be either 4 or 8 */
      ww &= 0x7; /* (sizeof(longword)-1); */
      #else
      ww &= (sizeof(int)-1); /* int alignment */
      #endif
      if (ww)
      {
        Log("rc5unitwork for problem %d is misaligned!\n", threadindex);
        RaiseExitRequestTrigger();
        return;
      }
    }
  }

//LogScreen("Problem created. threadindex=%u\n", threadindex);

  #ifdef STRESS_THREADS_AND_BUFFERS
  {
    static int runlevel = 0;
    if (runlevel != -12345)
    {
      if ((++runlevel) != 1)
      {
        --runlevel;
        return;
      }
      RaisePauseRequestTrigger();
      LogScreen("Warning! STRESS_THREADS_AND_BUFFERS is defined.\n"
                "Are you sure that the client is pointing at\n"
                "a test proxy? If so, type 'yes': ");
      char getyes[10];
      ConInStr(getyes,4,0);
      ClearPauseRequestTrigger();
      if (strcmp( getyes, "yes" ) == 0)  FIXME: exit if entered 'yes' ?!
      {
        runlevel = +12345;
        RaiseExitRequestTrigger();
        return;
      }
      runlevel = -12345;
    }
  }
  #endif
}

/* ------------------------------------------------------------------- */

// for some odd reasons, the RC5 algorithm requires keys in reversed order
//         key.hi   key.lo
// ie key 01234567:89ABCDEF is sent to rc5_unit_func like that :
//        EFCDAB89:67452301
// This function switches from one format to the other.
//
// [Even if it looks like a little/big endian problem, it isn't. Whatever
//  endianess the underlying system has, we must swap every byte in the key
//  before sending it to rc5_unit_func()]
//
// Note that DES has a similiar but far more complex system, but everything
// is handled by des_unit_func().

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

/* generate a contestwork for benchmarking (should be 
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
    case OGR:
    {
      //24/2-22-32-21-5-1-12
      //25/2-8-6-20-4-9
      work->ogr.workstub.stub.marks = 25;    //24;
      work->ogr.workstub.worklength = 6;     //7;
      work->ogr.workstub.stub.length = 6;    //7;
      work->ogr.workstub.stub.diffs[0] = 2;  //2;
      work->ogr.workstub.stub.diffs[1] = 8;  //22;
      work->ogr.workstub.stub.diffs[2] = 6;  //32;
      work->ogr.workstub.stub.diffs[3] = 20; //21;
      work->ogr.workstub.stub.diffs[4] = 4;  //5;
      work->ogr.workstub.stub.diffs[5] = 9;  //1;
      work->ogr.workstub.stub.diffs[6] = 0;  //12;
      work->ogr.nodes.lo = 0;
      work->ogr.nodes.hi = 0;
      return contestid;
    }
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
static unsigned int __compute_permille(unsigned int cont_i, ContestWork *work);

/* LoadState() and RetrieveState() work in pairs. A LoadState() without
   a previous RetrieveState(,,purge) will fail, and vice-versa.
*/
int Problem::LoadState( const ContestWork * work, unsigned int contestid,
              u32 _iterations, int expected_cputype,
              int expected_corenum, int expected_os,
              int expected_buildfrac )
{
  ContestWork for_magic;

  if (initialized)
  {
    /* This can only happen if RetrieveState(,,purge) was not called */
    Log("BUG! LoadState() without previous RetrieveState(,,purge)!\n");
    return -1;
  }

  last_resultcode = -1;
  started = initialized = 0;
  loadtime_sec = loadtime_usec = 0;
  elapsed_time_sec = elapsed_time_usec = 0;
  runtime_sec = runtime_usec = 0;
  last_runtime_sec = last_runtime_usec = 0;
  last_runtime_is_invalid = 1;
  memset((void *)&profiling, 0, sizeof(profiling));
  startpermille = 0;
// unused:  permille = 0;
  startkeys.lo = startkeys.hi = 0;
  loaderflags = 0;
  contest = contestid;
  tslice = _iterations;
  was_reset = 0;
  is_random = 0;
  is_benchmark = 0;

  //has to be done before selcore 
  if (work == CONTESTWORK_MAGIC_RANDOM) /* ((const ContestWork *)0) */
  {
    contestid = __gen_random_work(contestid, &for_magic);
    work = &for_magic;
    is_random = 1;
  }
  else if (work == CONTESTWORK_MAGIC_BENCHMARK) /* ((const ContestWork *)1) */
  {
    contestid = __gen_benchmark_work(contestid, &for_magic);
    work = &for_magic;
    is_benchmark = 1;
  }

  if (!IsProblemLoadPermitted(threadindex, contestid))
    return -1;

  client_cpu = CLIENT_CPU; /* usual case */
  use_generic_proto = 0; /* assume RC5/DES unit_func is _not_ generic form */
  cruncher_is_asynchronous = 0; /* not a co-processor */
  memset( &unit_func, 0, sizeof(unit_func) );
  coresel = selcoreSelectCore( contestid, threadindex, &client_cpu, this );
  if (coresel < 0)
    return -1;

  //----------------------------------------------------------------

  switch (contest)
  {
    case RC5:
    if (!is_random &&
       (work->crypto.iterations.hi || work->crypto.iterations.lo >= (1L<<28)))
    {
      last_rc5_prefix = (int)(work->crypto.key.hi >> 24);
    }    
    if ((MINIMUM_ITERATIONS % pipeline_count) != 0)
    {
      LogScreen("(MINIMUM_ITERATIONS %% pipeline_count) != 0)\n");
      return -1;
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
      contestwork.crypto.key.hi = ( work->crypto.key.hi );
      contestwork.crypto.key.lo = ( work->crypto.key.lo );
      contestwork.crypto.iv.hi = ( work->crypto.iv.hi );
      contestwork.crypto.iv.lo = ( work->crypto.iv.lo );
      contestwork.crypto.plain.hi = ( work->crypto.plain.hi );
      contestwork.crypto.plain.lo = ( work->crypto.plain.lo );
      contestwork.crypto.cypher.hi = ( work->crypto.cypher.hi );
      contestwork.crypto.cypher.lo = ( work->crypto.cypher.lo );
      contestwork.crypto.keysdone.hi = ( work->crypto.keysdone.hi );
      contestwork.crypto.keysdone.lo = ( work->crypto.keysdone.lo );
      contestwork.crypto.iterations.hi = ( work->crypto.iterations.hi );
      contestwork.crypto.iterations.lo = ( work->crypto.iterations.lo );

      if (contestwork.crypto.keysdone.lo || contestwork.crypto.keysdone.hi)
      {
        if (client_cpu != expected_cputype || coresel != expected_corenum ||
            CLIENT_OS != expected_os || CLIENT_BUILD_FRAC!=expected_buildfrac)
        {
          contestwork.crypto.keysdone.lo = contestwork.crypto.keysdone.hi = 0;
          was_reset = 1;
        }
      }

      //determine starting key number. accounts for carryover & highend of keysdone
      rc5unitwork.L0.hi = contestwork.crypto.key.hi + contestwork.crypto.keysdone.hi +
         ((((contestwork.crypto.key.lo & 0xffff) + (contestwork.crypto.keysdone.lo & 0xffff)) +
           ((contestwork.crypto.key.lo >> 16) + (contestwork.crypto.keysdone.lo >> 16))) >> 16);
      rc5unitwork.L0.lo = contestwork.crypto.key.lo + contestwork.crypto.keysdone.lo;
      if (contest == RC5)
        __SwitchRC5Format(&(rc5unitwork.L0.hi), &(rc5unitwork.L0.lo));
      refL0.lo = rc5unitwork.L0.lo;
      refL0.hi = rc5unitwork.L0.hi;

      // set up the unitwork structure
      rc5unitwork.plain.hi = contestwork.crypto.plain.hi ^ contestwork.crypto.iv.hi;
      rc5unitwork.plain.lo = contestwork.crypto.plain.lo ^ contestwork.crypto.iv.lo;
      rc5unitwork.cypher.hi = contestwork.crypto.cypher.hi;
      rc5unitwork.cypher.lo = contestwork.crypto.cypher.lo;

      startkeys.hi = contestwork.crypto.keysdone.hi;
      startkeys.lo = contestwork.crypto.keysdone.lo;
      startpermille = __compute_permille( contest, &contestwork );
      break;
    }
    #if defined(HAVE_OGR_CORES)
    case OGR:
    {
      contestwork.ogr = work->ogr;
      if (contestwork.ogr.nodes.hi != 0 || contestwork.ogr.nodes.lo != 0)
      {
        if (client_cpu != expected_cputype || coresel != expected_corenum ||
            CLIENT_OS != expected_os || CLIENT_BUILD_FRAC!=expected_buildfrac)
        {
          was_reset = 1;
          contestwork.ogr.workstub.worklength = contestwork.ogr.workstub.stub.length;
          contestwork.ogr.nodes.hi = contestwork.ogr.nodes.lo = 0;
        }
      }
      //unit_func.ogr = [xxx_]ogr_get_dispatch_table(); was done by selcore
      int r = (unit_func.ogr)->init();
      if (r != CORE_S_OK)
        return -1;
      r = (unit_func.ogr)->create(&contestwork.ogr.workstub,
                      sizeof(WorkStub), core_membuffer, MAX_MEM_REQUIRED_BY_CORE);
      if (r != CORE_S_OK)
        return -1;
      if (contestwork.ogr.workstub.worklength > (u32)contestwork.ogr.workstub.stub.length)
      {
        startkeys.hi = contestwork.ogr.nodes.hi;
        startkeys.lo = contestwork.ogr.nodes.lo;
        startpermille = __compute_permille( contest, &contestwork );
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
    loadtime_sec = 0;
    struct timeval tv;
    if (CliGetMonotonicClock(&tv) != 0)
    {
      if (CliGetMonotonicClock(&tv) != 0)
        loadtime_sec = 0xfffffffful;
    }
    if (loadtime_sec == 0)
    {
      loadtime_sec = tv.tv_sec;
      loadtime_usec = tv.tv_usec;
    }
    elapsed_time_sec = 0xfffffffful; // invalid while RESULT_WORKING
  }

  loaded_problems[contest]++;       /* per contest */  
  loaded_problems[CONTEST_COUNT]++; /* total */
  last_resultcode = RESULT_WORKING;
  initialized = 1;
  return( 0 );
}



/* ------------------------------------------------------------------- */

/* LoadState() and RetrieveState() work in pairs. A LoadState() without
   a previous RetrieveState(,,purge) will fail, and vice-versa.
*/
int Problem::RetrieveState( ContestWork * work, unsigned int *contestid, 
                            int dopurge, int dontwait )
{
  if (!initialized)
    return -1;
  if (work) // store back the state information
  {
    switch (contest) {
      case RC5:
      case DES:
      case CSC:
        // nothing special needs to be done here
        break;
      #if defined(HAVE_OGR_CORES)
      case OGR:
        (unit_func.ogr)->getresult(core_membuffer, &contestwork.ogr.workstub, sizeof(WorkStub));
        break;
      #endif
    }
    memcpy( (void *)work, (void *)&contestwork, sizeof(ContestWork));
  }
  if (contestid)
    *contestid = contest;
  if (dopurge)
  {
    initialized = 0;
    if (!dontwait) /* normal state is to wait. But we can't wait when aborting */
    {
      while (running) /* need to guarantee that no Run() will occur on a */
        usleep(1000); /* purged problem. */
    }
    loaded_problems[contest]--;       /* per contest */  
    loaded_problems[CONTEST_COUNT]--; /* total */
  }
  if (last_resultcode < 0)
    return -1;
  return ( last_resultcode );
}

/* ------------------------------------------------------------- */

int Problem::Run_RC5(u32 *keyscheckedP /* count of ... */, int *resultcode)
{
  s32 rescode = -1;

  /* a brace to ensure 'keystocheck' is not referenced in the common part */
  {
    u32 keystocheck = *keyscheckedP;
    // don't allow a too large of a keystocheck be used ie (>(iter-keysdone))
    // (technically not necessary, but may save some wasted time)
    if (contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi)
    {
      u32 todo = contestwork.crypto.iterations.lo-contestwork.crypto.keysdone.lo;
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
              "pipeline_count = %lu, iterations%%pipeline_count = %lu\n",
              (unsigned long)keystocheck, (unsigned long)keystocheck,
              (unsigned long)(*keyscheckedP), (unsigned long)(*keyscheckedP),
              pipeline_count, keystocheck%pipeline_count );
    #endif

    if (use_generic_proto)
    {
      //we don't care about pipeline_count when using unified cores.
      //we _do_ need to care that the keystocheck and starting key are aligned.

      *keyscheckedP = keystocheck; /* Pass 'keystocheck', get back 'keyschecked'*/
      rescode = (*(unit_func.gen))(&rc5unitwork,keyscheckedP,core_membuffer);

      if (rescode >= 0 && cruncher_is_asynchronous) /* co-processor or similar */
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
      *keyscheckedP = (*(unit_func.rc5))(&rc5unitwork,(keystocheck/pipeline_count));
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
  __IncrementKey(&refL0.hi, &refL0.lo, *keyscheckedP, contest);

  // Compare ref to core key incrementation
  if (((refL0.hi != rc5unitwork.L0.hi) || (refL0.lo != rc5unitwork.L0.lo))
      && (*resultcode != RESULT_FOUND) )
  {
    if (contestwork.crypto.iterations.hi == 0 &&
        contestwork.crypto.iterations.lo == 0x20000) /* test case */
    {
      Log("RC5 incrementation mismatch:\n"
          "Debug Information: %08x:%08x - %08x:%08x\n",
          rc5unitwork.L0.hi, rc5unitwork.L0.lo, refL0.hi, refL0.lo);
    }
    *resultcode = -1;
    return -1;
  };

  // Checks passed, increment keys done count.
  contestwork.crypto.keysdone.lo += *keyscheckedP;
  if (contestwork.crypto.keysdone.lo < *keyscheckedP)
    contestwork.crypto.keysdone.hi++;

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)  //(*keyscheckedP < keystocheck)
  {
    // found it!
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo)
      contestwork.crypto.key.hi++; // wrap occured ?
    return RESULT_FOUND;
  }

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
       ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
       ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
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

int Problem::Run_CSC(u32 *iterationsP, int *resultcode)
{
#ifndef HAVE_CSC_CORES
  *iterationsP = 0;
  *resultcode = -1;
  return -1;
#else
  s32 rescode = (*(unit_func.gen))( &rc5unitwork, iterationsP, core_membuffer );

  if (rescode < 0) /* "kiter" error */
  {
    *resultcode = -1;
    return -1;
  }
  *resultcode = (int)rescode;

  // Increment reference key count
  __IncrementKey (&refL0.hi, &refL0.lo, *iterationsP, contest);

  // Compare ref to core key incrementation
  if ((refL0.hi != rc5unitwork.L0.hi) || (refL0.lo != rc5unitwork.L0.lo))
  {
    if (contestwork.crypto.iterations.hi == 0 &&
        contestwork.crypto.iterations.lo == 0x20000) /* test case */
    {
      Log("CSC incrementation mismatch:\n"
          "expected %08x:%08x, got %08x:%08x\n",
          refL0.lo, refL0.hi, rc5unitwork.L0.lo, rc5unitwork.L0.hi );
    }
    *resultcode = -1;
    return -1;
  }

  // Checks passed, increment keys done count.
  contestwork.crypto.keysdone.lo += *iterationsP;
  if (contestwork.crypto.keysdone.lo < *iterationsP)
    contestwork.crypto.keysdone.hi++;

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)
  {
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo)
      contestwork.crypto.key.hi++; // wrap occured ?
    return RESULT_FOUND;
  }

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
       ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
       ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
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

int Problem::Run_DES(u32 *iterationsP, int *resultcode)
{
#ifndef HAVE_DES_CORES
  *iterationsP = 0;  /* no keys done */
  *resultcode = -1; /* core error */
  return -1;
#else

  //iterationsP == in: suggested iterations, out: effective iterations
  u32 kiter = (*(unit_func.des))( &rc5unitwork, iterationsP, (char *)core_membuffer );

  __IncrementKey ( &refL0.hi, &refL0.lo, *iterationsP, contest);
  // Increment reference key count

  if (((refL0.hi != rc5unitwork.L0.hi) ||  // Compare ref to core
      (refL0.lo != rc5unitwork.L0.lo)) &&  // key incrementation
      (kiter == *iterationsP))
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #23: Please contact help@distributed.net\n"
        "Debug Information: %08x:%08x - %08x:%08x\n",
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1;
    return -1;
  };

  contestwork.crypto.keysdone.lo += kiter;
  if (contestwork.crypto.keysdone.lo < kiter)
    contestwork.crypto.keysdone.hi++;
    // Checks passed, increment keys done count.

  // Update data returned to caller
  if (kiter < *iterationsP)
  {
    // found it!
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo)
      contestwork.crypto.key.hi++; // wrap occured ?
    *resultcode = RESULT_FOUND;
    return RESULT_FOUND;
  }
  else if (kiter != *iterationsP)
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #24: Please contact help@distributed.net\n"
        "Debug Information: k: %x t: %x\n"
        "Debug Information: %08x:%08x - %08x:%08x\n", kiter, *iterationsP,
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1; /* core error */
    return -1;
  };

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
     ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
     ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
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

int Problem::Run_OGR(u32 *iterationsP, int *resultcode)
{
#if !defined(HAVE_OGR_CORES)
  iterationsP = iterationsP;
#else
  int r, nodes;

  if (*iterationsP > 0x100000UL && !is_benchmark)
    *iterationsP = 0x100000UL;

  nodes = (int)(*iterationsP);
  r = (unit_func.ogr)->cycle(core_membuffer, &nodes);
  *iterationsP = (u32)nodes;

  u32 newnodeslo = contestwork.ogr.nodes.lo + nodes;
  if (newnodeslo < contestwork.ogr.nodes.lo) {
    contestwork.ogr.nodes.hi++;
  }
  contestwork.ogr.nodes.lo = newnodeslo;

  switch (r)
  {
    case CORE_S_OK:
    {
      r = (unit_func.ogr)->destroy(core_membuffer);
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
      if ((unit_func.ogr)->getresult(core_membuffer, &contestwork.ogr.workstub, sizeof(WorkStub)) == CORE_S_OK)
      {
        //Log("OGR Success!\n");
        contestwork.ogr.workstub.stub.length =
                  (u16)(contestwork.ogr.workstub.worklength);
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

static void __compute_run_times(Problem *problem,
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
      problem->last_runtime_sec = elapsedhi;
      problem->last_runtime_usec = elapsedlo;

      elapsedhi += problem->runtime_sec;
      elapsedlo += problem->runtime_usec;
      if (elapsedlo >= 1000000UL)
      {
        elapsedhi++;
        elapsedlo -= 1000000UL;
      }
      problem->runtime_sec  = elapsedhi;
      problem->runtime_usec = elapsedlo;
    }
  }
  if (last_runtime_is_invalid)
  {
    problem->last_runtime_sec = 0;
    problem->last_runtime_usec = 0;
  }
  problem->last_runtime_is_invalid = last_runtime_is_invalid;

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
      elapsedhi = problem->runtime_sec;
      elapsedlo = problem->runtime_usec;
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
    problem->elapsed_time_sec  = elapsedhi;
    problem->elapsed_time_usec = elapsedlo;
  }

  return;
}

/* ---------------------------------------------------------------- */

int Problem::Run(void) /* returns RESULT_*  or -1 */
{
  static volatile int s_using_ptime = -1;
  struct timeval tv;
  int retcode, core_resultcode;
  u32 iterations, runstart_secs, runstart_usecs;
  int using_ptime;

  last_runtime_is_invalid = 1; /* haven't changed runtime fields yet */

  if ( !initialized )
  {
    return ( -1 );
  }
  if ((++running) > 1)
  {
    --running;
    return -1;
  }

#ifdef STRESS_THREADS_AND_BUFFERS
  if (contest == RC5 && !started)
  {
    contestwork.crypto.key.hi = contestwork.crypto.key.lo = 0;
    contestwork.crypto.keysdone.hi = contestwork.crypto.iterations.hi;
    contestwork.crypto.keysdone.lo = contestwork.crypto.iterations.lo;
    runtime_usec = 1; /* ~1Tkeys for a 2^20 packet */
    elapsed_time_usec = 1;
    last_resultcode = RESULT_NOTHING;
    started = 1;
  }
#endif

  if ( last_resultcode != RESULT_WORKING ) /* _FOUND, _NOTHING or -1 */
  {
    running--;
    return ( last_resultcode );
  }

  /*
    On return from the Run_XXX contestwork must be in a state that we
    can put away to disk - that is, do not expect the loader (probfill
    et al) to fiddle with iterations or key or whatever.

    The Run_XXX functions do *not* update problem.last_resultcode, they use
    core_resultcode instead. This is so that members of the problem object
    that are updated after the resultcode has been set will not be out of
    sync when the main thread gets it with RetrieveState().

    note: although the value returned by Run_XXX is usually the same as
    the core_resultcode it is not always the case. For instance, if
    post-LoadState() initialization  failed, but can be deferred, Run_XXX
    may choose to return -1, but keep core_resultcode at RESULT_WORKING.
  */

  started = 1;
  last_runtime_usec = last_runtime_sec = 0;
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
  iterations = tslice;
  core_resultcode = last_resultcode;
  retcode = -1;

  switch (contest)
  {
    case RC5: retcode = Run_RC5( &iterations, &core_resultcode );
              break;
    case DES: retcode = Run_DES( &iterations, &core_resultcode );
              break;
    case OGR: retcode = Run_OGR( &iterations, &core_resultcode );
              break;
    case CSC: retcode = Run_CSC( &iterations, &core_resultcode );
              break;
    default: retcode = core_resultcode = last_resultcode = -1;
       break;
  }

  if (retcode < 0) /* don't touch tslice or runtime as long as < 0!!! */
  {
    running--;
    return -1;
  }
  if (!started || !initialized) /* RetrieveState(,,purge) has been called */
  {
    core_resultcode = -1; // "Discarded (core error)": discard the purged block
  }

  core_run_count++;
  __compute_run_times( this, runstart_secs, runstart_usecs, &loadtime_sec, &loadtime_usec,
                       using_ptime, &s_using_ptime, core_resultcode );
  tslice = iterations;
  last_resultcode = core_resultcode;
  running--;
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
      (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_RISCOS)
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
/* support functions for Problem::GetProblemInfo()                         */
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
  #if (ULONG_MAX > 0xfffffffful) /* 64 bit */
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
  #else /* 32bit - long division using rshift and sub */
  if (numerhi == 0 && denomhi == 0) 
  {
    u32 n = numerlo, d = denomlo;
    if (remalo) *remalo = n % d;
    if (remahi) *remahi = 0;
    if (quotlo) *quotlo = n / d;
    if (quothi) *quothi = 0;
  } 
  else 
  {
    u32 qhi = 0, qlo = 0;
    u32 nhi = numerhi, nlo = numerlo;
    u32 dhi = denomhi, dlo = denomlo;
    int count = 0;
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
  }
  #endif
}

/* ----------------------------------------------------------------------- */

static char *__u64stringify(char *buffer, unsigned int buflen, u32 hi, u32 lo,
                            int numstr_style, const char *numstr_suffix )
{
  /* numstring_style: 
   * -1=unformatted, 
   *  0=commas, 
   *  1=0+space between magna and number (or at end if no magnitude char)
   *  2=1+numstr_suffix
  */
  if (buffer && buflen)
  {
    char numbuf[32]; /* U64MAX is "18,446,744,073,709,551,615" (len=26) */
    unsigned int suffix_len;

    if (numstr_style != 2 || !numstr_suffix)
      numstr_suffix = ""; 
    suffix_len = strlen( numstr_suffix );
    if (numstr_style == 2 && suffix_len == 0)
      numstr_style = 1;
    if (numstr_style == 1 || numstr_style == 2) 
      suffix_len++; 
    if (buflen > suffix_len)
      buflen -= suffix_len;
    else if (buflen >= 5) /* so that it falls into next part */
      buflen = 4;

    if (buflen < 5)
    {
      strcpy( numbuf, "***" );
      suffix_len = 0;
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
      if (numstr_style != -1 && len > 3) /* at least one comma separator */
      {
        char fmtbuf[sizeof(numbuf)];
        char *r = &numbuf[len];
        char *w = &fmtbuf[sizeof(fmtbuf)];
        unsigned int pos = 0;
        *--w = '\0';
        for (;;)
        {
          *--w = *--r;
          if (r == &numbuf[0])
            break;
          if (((++pos) % 3)==0)
            *--w = ',';
        }
        len = strlen(strcpy( numbuf, w )); 
        if (len >= buflen)
        {
          pos = buflen-4; /* "00X\0" */
          while (len > pos || numbuf[len] != ',')
          {
            len--;
            if (numbuf[len] == ',')
              magna++;
          }
          numbuf[len] = '.';
          len += 3;
          numbuf[len] = '\0';
        }
      } /* len > 3 */
      if (numstr_style == 1 || numstr_style == 2)
        numbuf[len++] = ' ';
      if (magna)
        numbuf[len++] = magna_tab[magna];
      numbuf[len] = '\0';
    } /* buflen >= 5 */
    strncpy( buffer, numbuf, buflen ); 
    buffer[buflen-1] = '\0';
    if (numstr_style == 2) /* buflen has already been checked to ensure */
      strcat(buffer, numstr_suffix); /* this strcat() is ok */
  }
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
  if ((hi || lo) && (secs || usecs))
  {
    u32 t, thi, tlo;
    __u64mul( 0, secs, 0, 1000, &thi, &tlo ); /* secs *= 1000 */
    t = tlo + (usecs / 1000);
    if (t < tlo) thi++;
    tlo = t;                                  /* ms = secs*1000+usecs/1000 */
    __u64mul( hi, lo, 0, 1000, &hi,  &lo ); /* iter *= 1000 */
    __u64div( hi, lo, thi, tlo, &hi, &lo, 0, 0 ); /* (iter*1000)/millisecs */
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
  return ratebuf;
}

static unsigned int __compute_permille(unsigned int cont_i, ContestWork *work)
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

/* more info than you ever wanted. :) any/all params can be NULL/0
 * tcount = total_number_of_iterations_to_do
 * ccount = number_of_iterations_done_thistime.
 * dcount = number_of_iterations_done_ever
 * counts are unbiased (adjustment for DES etc already done)
 * numstring_style: -1=unformatted, 0=commas, 
 * 1=0+space between magna and number (or at end), 2=1+"nodes"/"keys"
*/
int Problem::GetProblemInfo(unsigned int *cont_id, const char **cont_name, 
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
  int rescode = last_resultcode;
  if (initialized && rescode >= 0)
  {
    u32 e_sec = 0, e_usec = 0;

    if (cont_id)
    {
      *cont_id = contest;
    }
    if (cont_name)
    {
      switch (contest)
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
      switch (contest)
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
      if (elapsed_time_sec != 0xfffffffful)
      {
        /* problem finished, elapsed time has already calculated by Run() */
        e_sec  = elapsed_time_sec;
        e_usec = elapsed_time_usec;
      }     
      else /* compute elapsed wall clock time since loadtime */
      {
        u32 start_sec  = loadtime_sec;
        u32 start_usec = loadtime_usec;

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
          e_sec  = runtime_sec;
          e_usec = runtime_usec;
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
      if (is_benchmark && (runtime_sec || runtime_usec))
      {
        e_sec = runtime_sec;
        e_usec = runtime_usec;
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
      int rescode = RetrieveState( &work, &contestid, 0, 0 );

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

            ccounthi = startkeys.hi;
            ccountlo = startkeys.lo;
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
            ccountlo = dcountlo - startkeys.lo;
            ccounthi = dcounthi - startkeys.hi;
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
          if (!started)
            *c_permille = startpermille;
          else if (rescode != RESULT_WORKING) /* _FOUND or _NOTHING */
            *c_permille = 1000;
          else
            *c_permille = __compute_permille(contestid, &work); 
        }
        if (s_permille)
          *s_permille = startpermille;
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
  } /* if (initialized && last_resultcode >= 0) */
  return rescode;
}

