/* 
 * Copyright distributed.net 1998-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Written August 1998 by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * -------------------------------------------------------------------
 * program (pro'-gram) [vi]: To engage in a pastime similar to banging
 * one's head against a wall but with fewer opportunities for reward.
 * -------------------------------------------------------------------
*/
const char *selcore_cpp(void) {
return "@(#)$Id: selcore.cpp,v 1.131 2015/10/22 19:45:54 stream Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // problem class
#include "cpucheck.h"  // cpu selection, GetTimesliceBaseline()
#include "logstuff.h"  // Log()/LogScreen()
#include "clicdata.h"  // GetContestNameFromID()
#include "bench.h"     // TBenchmark()
#include "selftest.h"  // SelfTest()
#include "selcore.h"   // keep prototypes in sync
#include "probman.h"   // GetManagedProblemCount()
#include "triggers.h"  // CheckExitRequestTriggerNoIO()
#include "util.h"      // TRACE_OUT

/* ======================================================================== */


const char **corenames_for_contest( unsigned int cont_i )
{
  switch (cont_i)
    {
#ifdef HAVE_RC5_72_CORES
    case RC5_72:
      return corenames_for_contest_rc572();
#endif
#ifdef HAVE_OGR_CORES
    case OGR_NG:
      return corenames_for_contest_ogr_ng();
#endif
#ifdef HAVE_OGR_PASS2
    case OGR_P2:
      return corenames_for_contest_ogr();
#endif
    default:
      return NULL;
    }
}

/* -------------------------------------------------------------------- */

/* 
** Apply substition according to the same rules enforced by
** selcoreSelectCore() [ie, return the cindex of the core actually used
** after applying appropriate OS/architecture/#define limitations to
** ensure the client doesn't crash]
**
** This is necessary when the list of cores is a superset of the
** cores supported by a particular build. For example, all x86 clients
** display the same core list for RC5, but as not all cores may be 
** available in a particular client/build/environment, this function maps 
** between the ones that aren't available to the next best ones that are.
**
** Note that we intentionally don't do very intensive validation here. Thats
** selcoreGetSelectedCoreForContest()'s job when the user chooses to let
** the client auto-select. If the user has explicitely specified a core #, 
** they have to live with the possibility that the choice will at some point
** no longer be optimal.
*/
static int apply_selcore_substitution_rules(unsigned int contestid, int cindex, int device)
{
  switch (contestid)
    {
#ifdef HAVE_RC5_72_CORES
    case RC5_72:
      return apply_selcore_substitution_rules_rc572(cindex, device);
#endif
#ifdef HAVE_OGR_CORES
    case OGR_NG:
      return apply_selcore_substitution_rules_ogr_ng(cindex);
#endif
#ifdef HAVE_OGR_PASS2
    case OGR_P2:
      return apply_selcore_substitution_rules_ogr(cindex);
#endif
    default:
      return cindex;
    }
}

/* -------------------------------------------------------------------- */

unsigned int corecount_for_contest( unsigned int cont_i )
{
  const char **cnames = corenames_for_contest( cont_i );
  if (cnames != NULL)
  {
    unsigned int count_i = 0;
    while (cnames[count_i] != NULL)
      count_i++;
    return count_i;
  }
  return 0;
}

/* -------------------------------------------------------------------- */

unsigned int nominal_rate_for_contest( unsigned int cont_i)
{
  switch (cont_i)
  {
#ifdef HAVE_RC5_72_CORES
    case RC5_72:
      return estimate_nominal_rate_rc572();
#endif
#ifdef HAVE_OGR_CORES
    case OGR_NG:
      return 0;
#endif
#ifdef HAVE_OGR_PASS2
    case OGR_P2:
      return estimate_nominal_rate_ogr();
#endif
    default:
      return 0;
  }
}

/* ===================================================================== */

void selcoreEnumerateWide( int (*enumcoresproc)(
                            const char **corenames, int idx, void *udata ),
                       void *userdata )
{
  if (enumcoresproc)
  {
    unsigned int corenum;
    for (corenum = 0;;corenum++)
    {
      const char *carray[CONTEST_COUNT];
      int have_one = 0;
      unsigned int cont_i;
      for (cont_i = 0; cont_i < CONTEST_COUNT;cont_i++)
      {
        carray[cont_i] = (const char *)0;
        if (corenum < corecount_for_contest( cont_i ))
        {
          const char **names = corenames_for_contest( cont_i );
          carray[cont_i] = names[corenum];
          have_one++;
        }
      }
      if (!have_one)
        break;
      if (! ((*enumcoresproc)( &carray[0], (int)corenum, userdata )) )
        break;
    }
  }
  return;
}

/* ---------------------------------------------------------------------- */

void selcoreEnumerate( int (*enumcoresproc)(unsigned int cont, 
                            const char *corename, int idx, void *udata, Client *client ),
                       void *userdata, Client *client )
{
  if (enumcoresproc)
  {
    int stoploop = 0;
    unsigned int cont_i;
    for (cont_i = 0; !stoploop && cont_i < CONTEST_COUNT; cont_i++)
    {
      unsigned int corecount = corecount_for_contest( cont_i );
      if (corecount)
      {
        unsigned int coreindex;
        const char **corenames = corenames_for_contest(cont_i);
        for (coreindex = 0; !stoploop && coreindex < corecount; coreindex++)
          stoploop = (! ((*enumcoresproc)(cont_i, 
                      corenames[coreindex], (int)coreindex, userdata, client )) );
      }
    }
  }
  return;
}

/* --------------------------------------------------------------------- */

int selcoreValidateCoreIndex( unsigned int cont_i, int idx, Client *client )
{
  if (idx >= 0 && idx < ((int)corecount_for_contest( cont_i )))
  {
    int device = hackGetUsedDeviceIndex(client, 0); // FIXME: without -devicenum, validates against GPU0

    if (idx == apply_selcore_substitution_rules(cont_i, idx, device))
      return idx;
  }
  return -1;
}

/* --------------------------------------------------------------------- */

const char *selcoreGetDisplayName( unsigned int cont_i, int idx )
{
  if (idx >= 0 && idx < ((int)corecount_for_contest( cont_i )))
  {
     const char **names = corenames_for_contest( cont_i );
     return names[idx];
  }
  return "";
}

/* ===================================================================== */

static struct
{
  int user_cputype[CONTEST_COUNT]; /* what the user has in the ini */
  int corenum[CONTEST_COUNT]; /* what we map it to */
} selcorestatics;
static int selcore_initlev = -1; /* not initialized yet */


int DeinitializeCoreTable( void )  /* ClientMain calls this */
{
  if (selcore_initlev <= 0)
  {
    Log("ACK! DeinitializeCoreTable() called for uninitialized table\n");
    return -1;
  }


  #ifdef HAVE_RC5_72_CORES
  DeinitializeCoreTable_rc572();
  #endif
  #if defined(HAVE_OGR_CORES)
  DeinitializeCoreTable_ogr_ng();
  #endif
  #if defined(HAVE_OGR_PASS2)
  DeinitializeCoreTable_ogr();
  #endif

  selcore_initlev--;
  return 0;
}

/* ---------------------------------------------------------------------- */

int InitializeCoreTable( int *coretypes ) /* ClientMain calls this */
{
  int first_time = 0;
  unsigned int cont_i;

  if (selcore_initlev > 0)
  {
    Log("ACK! InitializeCoreTable() called more than once!\n");
    return -1;
  }
  if (selcore_initlev < 0)
  {
    first_time = 1;
    selcore_initlev = 0;
  }

#if (CLIENT_OS == OS_AIX)
  if (first_time) /* we only want to do this once */
  {
    long detected_type = GetProcessorType(1);

    if (detected_type > 0) {
      if ((detected_type & (1L<<24)) != 0 ) {
# if (CLIENT_CPU == CPU_POWER)
        /* we're running on PowerPC */
        Log("PANIC::This is a Power client running on PowerPC - please\n"
	    "get the correct client for your platform.\n");
# else /* CPU_POWERPC */
        /* we're running on Power */
        Log("PANIC::This is a PowerPC client running on Power - please\n"
            "get the correct client for your platform.\n");
# endif
        return -1;
      }
    }
  }
#endif


  #ifdef HAVE_RC5_72_CORES
  if (InitializeCoreTable_rc572(first_time) < 0) return -1;
  #endif
  #if defined(HAVE_OGR_CORES)
  if (InitializeCoreTable_ogr_ng(first_time) < 0) return -1;
  #endif
  #if defined(HAVE_OGR_PASS2)
  if (InitializeCoreTable_ogr(first_time) < 0) return -1;
  #endif


  if (first_time) /* we only want to do this once */
  {
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      selcorestatics.user_cputype[cont_i] = -1;
      selcorestatics.corenum[cont_i] = -1;
    }
  }
  if (coretypes)
  {
    for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    {
      int idx = 0;
      if (corecount_for_contest( cont_i ) > 0)
      {
        idx = coretypes[cont_i];
        if (idx < 0 || idx >= ((int)corecount_for_contest( cont_i )))
          idx = -1;
      }
      if (first_time || idx != selcorestatics.user_cputype[cont_i])
        selcorestatics.corenum[cont_i] = -1; /* got change */
      selcorestatics.user_cputype[cont_i] = idx;
    }
  }

  selcore_initlev++;
  return 0;
}

/* ---------------------------------------------------------------------- */

static long __bench_or_test( Client *client, int which, 
                            unsigned int cont_i, unsigned int benchsecs, int in_corenum )
{
  long rc = -1;
  /* FIXME: without -devicenum, test/bench will be run on GPU 0 only */
  int device = hackGetUsedDeviceIndex(client, 0);

  if (selcore_initlev > 0                  /* core table is initialized? */
      && cont_i < CONTEST_COUNT)           /* valid contest id? */
  {
    /* save current state */
    int user_cputype = selcorestatics.user_cputype[cont_i];
    int corenum = selcorestatics.corenum[cont_i];
    int coreidx, corecount = corecount_for_contest( cont_i );
    int fastest = -1;
    int hardcoded = selcoreGetPreselectedCoreForProject(cont_i, device);
    u32 bestrate_hi = 0, bestrate_lo = 0, refrate_hi = 0, refrate_lo = 0;

    rc = 0; /* assume nothing done */
    for (coreidx = 0; coreidx < corecount; coreidx++)
    {
      /* only bench/test cores that won't be automatically substituted */
      if (apply_selcore_substitution_rules(cont_i, coreidx, device) == coreidx)
      {
        if (in_corenum < 0)
            selcorestatics.user_cputype[cont_i] = coreidx; /* as if user set it */
        else
        {
          if( in_corenum < corecount )
          {
            selcorestatics.user_cputype[cont_i] = in_corenum;
            coreidx = corecount;
          }
          else  /* invalid core selection, test them all */
          {
            selcorestatics.user_cputype[cont_i] = coreidx;
            in_corenum = -1;
          }
        }
        selcorestatics.corenum[cont_i] = -1; /* reset to show name */

        if (which == 't') /* selftest */
          rc = SelfTest( client, cont_i );
        else if (which == 's') /* stresstest */
          rc = StressTest( client, cont_i );
        else {
          u32 temprate_hi, temprate_lo;
          rc = TBenchmark( client, cont_i, benchsecs, 0, &temprate_hi, &temprate_lo );
          if (rc > 0 && selcorestatics.corenum[cont_i] == hardcoded) {
            refrate_hi = temprate_hi;
            refrate_lo = temprate_lo;
          }
          if (rc > 0 && (temprate_hi > bestrate_hi || (temprate_hi == bestrate_hi && temprate_lo > bestrate_lo))) {
            bestrate_hi = temprate_hi;
            bestrate_lo = temprate_lo;
            fastest     = selcorestatics.corenum[cont_i];
          }
        }
        #if (CLIENT_OS != OS_WIN32 || !defined(SMC))
        if (rc <= 0) /* failed (<0) or not supported (0) */
          break; /* stop */
        #else
        // HACK! to ignore failed benchmark for x86 rc5 smc core #7 if
        // started from menu and another cruncher is active in background.
        if (rc <= 0) /* failed (<0) or not supported (0) */
        {
          if ( which == 'b' &&  cont_i == RC5 && coreidx == 7 )
            ; /* continue */
          else
            break; /* stop */
        }
        #endif
      }
    } /* for (coreidx = 0; coreidx < corecount; coreidx++) */

    selcorestatics.user_cputype[cont_i] = user_cputype;
    selcorestatics.corenum[cont_i] = corenum;

    /* Summarize the results if multiple cores have been benchmarked (#4108) */
#if (CLIENT_CPU != CPU_CELLBE)
    /* Not applicable for Cell due to PPU/SPU core selection hacks */
    if (in_corenum < 0 && fastest >= 0 && (bestrate_hi != 0 || bestrate_lo != 0)) {
      double percent = 100.0 * ((double)refrate_hi  * 4294967296.0 + (double)refrate_lo) /
                               ((double)bestrate_hi * 4294967296.0 + (double)bestrate_lo);
      char bestrate_str[32], refrate_str[32];

      U64stringify(bestrate_str, sizeof(bestrate_str), bestrate_hi, bestrate_lo, 2, CliGetContestUnitFromID(cont_i));
      U64stringify(refrate_str,  sizeof(refrate_str),  refrate_hi,  refrate_lo,  2, CliGetContestUnitFromID(cont_i));

      Log("%s benchmark summary :\n"
          "Default core : #%d (%s) %s/sec\n"
          "Fastest core : #%d (%s) %s/sec\n",
          CliGetContestNameFromID(cont_i), hardcoded,
          (hardcoded >= 0 ? selcoreGetDisplayName(cont_i, hardcoded) : "undefined"),
          refrate_str,
          fastest, selcoreGetDisplayName(cont_i, fastest), bestrate_str);
          
      if (percent < 100 && hardcoded >= 0 && hardcoded != fastest) {
        if (percent >= 97) {
          Log("Core #%d is marginally faster than the default core.\n"
              "Testing variability might lead to pick one or the other.\n",
              fastest);
        }
        else {
          Log("Core #%d is significantly faster than the default core.\n"
#if (CLIENT_CPU != CPU_CUDA && CLIENT_CPU != CPU_ATI_STREAM && CLIENT_CPU != CPU_OPENCL)
              "Please file a bug report along with the output of\n-cpuinfo.\n"
#else
              "The GPU core selection has been made as a tradeoff between core speed\n"
              "and responsiveness of the graphical desktop.\n"
              "Please file a bug report along with the output of -gpuinfo\n"
              "only if the the faster core selection does not degrade graphics performance.\n"
#endif
              "Changes in cores and selection are frequently made,\n"
              "so be sure to test with the latest client version,\n"
              "typically a pre-release, before filing a bug report.\n",
              fastest);
        }
      }
    }
#endif // CPU_CELLBE

#if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
    if (rc > 0 && cont_i == RC5 && 
          GetNumberOfDetectedProcessors() > 1) /* have x86 card */
    {
      Problem *prob = ProblemAlloc(); /* so bench/test gets threadnum+1 */
      rc = -1; /* assume alloc failed */
      if (prob)
      {
        Log("RC5: using x86 core.\n" );
        if (which != 's') /* bench */
          rc = TBenchmark( client, cont_i, benchsecs, 0 );
        else
          rc = SelfTest( client, cont_i );
        ProblemFree(prob);
      }
    }
    #endif

  } /* if (cont_i < CONTEST_COUNT) */

  return rc;
}

long selcoreBenchmark( Client *client, unsigned int cont_i, unsigned int secs, int corenum )
{
  return __bench_or_test( client, 'b', cont_i, secs, corenum );
}

long selcoreSelfTest( Client *client, unsigned int cont_i, int corenum)
{
  return __bench_or_test( client, 't', cont_i, 0, corenum );
}

long selcoreStressTest( Client *client, unsigned int cont_i, int corenum)
{
  return __bench_or_test( client, 's', cont_i, 0, corenum );
}

/* ---------------------------------------------------------------------- */

int selcoreGetPreselectedCoreForProject(unsigned int projectid, int device)
{
  switch (projectid)
    {
#ifdef HAVE_RC5_72_CORES
    case RC5_72:
      return selcoreGetPreselectedCoreForProject_rc572(device);
#endif
#ifdef HAVE_OGR_CORES
    case OGR_NG:
      return selcoreGetPreselectedCoreForProject_ogr_ng();
#endif
#ifdef HAVE_OGR_PASS2
    case OGR_P2:
      return selcoreGetPreselectedCoreForProject_ogr();
#endif
    default:
      return -1;
    }
}

/* ---------------------------------------------------------------------- */

/* this is called from Problem::LoadState() */
int selcoreGetSelectedCoreForContest( Client *client, unsigned int contestid )
{
  TRACE_OUT((0, "selcoreGetSelectedCoreForContest project=%d\n", contestid));
  int corename_printed = 0;
  static long detected_type = -123;
  const char *contname = CliGetContestNameFromID(contestid);
  /* FIXME: without -devicenum, core always selected for GPU 0 */
  int device = hackGetUsedDeviceIndex(client, 0);

  if (!contname) /* no such contest */
    return -1;
  if (!IsProblemLoadPermitted(-1 /*any thread*/, contestid))
    return -1; /* no cores available */
  if (selcore_initlev <= 0)                /* ACK! selcoreInitialize() */
    return -1;                             /* hasn't been called */

  if (corecount_for_contest(contestid) == 1) /* only one core? */
    return 0;
  if (selcorestatics.corenum[contestid] >= 0) /* already selected one? */
    return selcorestatics.corenum[contestid];

  TRACE_OUT((+1, "selcoreGetSelectedCoreForContest project=%d\n", contestid));
  if (detected_type == -123) /* haven't autodetected yet? */
  {
    int quietly = 1;
    unsigned int cont_i;
    for (cont_i = 0; quietly && cont_i < CONTEST_COUNT; cont_i++)
    {
      if (corecount_for_contest(cont_i) < 2)
        ; /* nothing */
      else if (selcorestatics.user_cputype[cont_i] < 0)
        quietly = 0;
    }
    detected_type = GetProcessorType(quietly, device);
    if (detected_type < 0)
      detected_type = -1;
  }

  selcorestatics.corenum[contestid] = selcorestatics.user_cputype[contestid];
  if (selcorestatics.corenum[contestid] < 0)
    selcorestatics.corenum[contestid] = selcoreGetPreselectedCoreForProject(contestid, device);

  TRACE_OUT((0, "cpu/arch preselection done: %d\n", selcorestatics.corenum[contestid]));

  if (selcorestatics.corenum[contestid] < 0)
    selcorestatics.corenum[contestid] = selcorestatics.user_cputype[contestid];

  if (selcorestatics.corenum[contestid] < 0) /* ok, bench it then */
  {
    TRACE_OUT((0, "do benchmark\n"));
    int corecount = (int)corecount_for_contest(contestid);
    selcorestatics.corenum[contestid] = 0;
    if (corecount > 0)
    {
      int whichcrunch, saidmsg = 0, fastestcrunch = -1;
      u32 fasttime_hi = 0, fasttime_lo = 0;

      for (whichcrunch = 0; whichcrunch < corecount; whichcrunch++)
      {
        /* test only if not substituted */
        if (whichcrunch == apply_selcore_substitution_rules(contestid, whichcrunch, device))
        {
          u32 rate_hi, rate_lo;
          selcorestatics.corenum[contestid] = whichcrunch;
          if (!saidmsg)
          {
            LogScreen("%s: Running micro-bench to select fastest core...\n", 
                      contname);
            saidmsg = 1;
          }
          if (CheckExitRequestTriggerNoIO())
            break;
          if (TBenchmark( client, contestid, 2, TBENCHMARK_QUIET | TBENCHMARK_IGNBRK, &rate_hi, &rate_lo ) > 0)
          {
#ifdef DEBUG
            LogScreen("%s Core %d: %d:%d keys/sec\n", contname,whichcrunch,rate_hi,rate_lo);
#endif
            if (fastestcrunch < 0 || (rate_hi > fasttime_hi || (rate_hi == fasttime_hi && rate_lo > fasttime_lo)))
            {
              fasttime_hi = rate_hi;
              fasttime_lo = rate_lo;
              fastestcrunch = whichcrunch;
            }
          }
        }
      }

      if (fastestcrunch < 0) /* all failed */
        fastestcrunch = 0; /* don't bench again */
      selcorestatics.corenum[contestid] = fastestcrunch;
    }
  }

  if (selcorestatics.corenum[contestid] >= 0) /* didn't fail */
  {
    /*
    ** substitution according to real selcoreSelectCore() rules
    ** Returns original if no substitution occurred.
    */
    int override = apply_selcore_substitution_rules(contestid, selcorestatics.corenum[contestid], device);
    if (!corename_printed)
    {
      if (override != selcorestatics.corenum[contestid])
      {
        Log("%s: selected core #%d (%s)\n"
            "     is not supported by this client/OS/architecture.\n"
            "     Using core #%d (%s) instead.\n", contname, 
                 selcorestatics.corenum[contestid], 
                 selcoreGetDisplayName(contestid, selcorestatics.corenum[contestid]),
                 override, selcoreGetDisplayName(contestid, override) );
      }
      else
      {
       Log("%s: using core #%d (%s).\n", contname, 
           selcorestatics.corenum[contestid], 
           selcoreGetDisplayName(contestid, selcorestatics.corenum[contestid]) );
      }
    }
    selcorestatics.corenum[contestid] = override;
  }

  TRACE_OUT((-1, "selcoreGetSelectedCoreForContest(%d) => %d\n", contestid, selcorestatics.corenum[contestid]));
  return selcorestatics.corenum[contestid];
}

/* ---------------------------------------------------------------------- */

int selcoreSelectCore( Client *client, unsigned int contestid, unsigned int threadindex,
                       int *client_cpuP, struct selcore *selinfo )
{
  switch (contestid)
    {
#ifdef HAVE_RC5_72_CORES
    case RC5_72:
      return selcoreSelectCore_rc572( client, threadindex, client_cpuP, selinfo );
#endif
#ifdef HAVE_OGR_PASS2
    case OGR_P2:
      return selcoreSelectCore_ogr( client, threadindex, client_cpuP, selinfo, contestid );
#endif
#ifdef HAVE_OGR_CORES
    case OGR_NG:
      return selcoreSelectCore_ogr_ng( client, threadindex, client_cpuP, selinfo, contestid );
#endif
    default:
      return -1; /* core selection failed */
    }
}

/* ------------------------------------------------------------- */

// Get GPU device index from threadindex
// (same as threadindex without -devicenum, otherwise forced by -devicenum)

int hackGetUsedDeviceIndex( Client *client, unsigned threadindex )
{
  int device = threadindex;
  /* TODO: really support separate cores for multiple types of devices */
  #if (CLIENT_CPU == CPU_CUDA) || (CLIENT_CPU == CPU_ATI_STREAM) || (CLIENT_CPU == CPU_OPENCL)
  if (client->devicenum >= 0)
    device = client->devicenum;
  #else
  DNETC_UNUSED_PARAM(client);
  #endif
  return device;
}

/* ------------------------------------------------------------- */
