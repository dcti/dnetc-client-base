/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
const char *clirun_cpp(void) {
return "@(#)$Id: clirun.cpp,v 1.93 1999/04/23 06:18:36 gregh Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
//#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "triggers.h"  // [Check|Raise][Pause|Exit]RequestTrigger()
#include "sleepdef.h"  // sleep(), usleep()
#include "pollsys.h"   // NonPolledSleep(), RegPollingProcedure() etc
#include "setprio.h"   // SetThreadPriority(), SetGlobalPriority()
#include "lurk.h"      // dialup object
#include "buffupd.h"   // BUFFERUPDATE_* constants
#include "clitime.h"   // CliTimer(), Time()/(CliGetTimeString(NULL,1))
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clicdata.h"  // CliGetContestNameFromID()
#include "checkpt.h"   // CHECKPOINT_[OPEN|CLOSE|REFRESH|_FREQ_[SECS|PERC]DIFF]
#include "cpucheck.h"  // GetTimesliceBaseline(), GetNumberOfSupportedProcessors()
#include "probman.h"   // GetProblemPointerFromIndex()
#include "probfill.h"  // LoadSaveProblems(), FILEENTRY_xxx macros
#include "modereq.h"   // ModeReq[Set|IsSet|Run]()
#include "clievent.h"  // ClientEventSyncPost() and constants

// --------------------------------------------------------------------------

#define OGR_TIMESLICE_MSEC 200
#define OGR_TIMESLICE_MAX  0x100000 // in units of nodes

// --------------------------------------------------------------------------

static struct
{
  int nonmt_ran;
  unsigned long yield_run_count;
  volatile int refillneeded;
  volatile u32 ogr_tslice;
} runstatics = {0,0,0,0x1000};

// --------------------------------------------------------------------------

static int checkifbetaexpired(void)
{
#if defined(BETA) || defined(BETA_PERIOD)
  timeval expirationtime;

  #ifndef BETA_PERIOD
  #define BETA_PERIOD (7L*24L*60L*60L) /* one week from build date */
  #endif    /* where "build date" is time of newest module in ./common/ */
  expirationtime.tv_sec = CliTimeGetBuildDate() + (time_t)BETA_PERIOD;
  expirationtime.tv_usec= 0;

  if ((CliTimer(NULL)->tv_sec) > expirationtime.tv_sec)
  {
    Log("This beta release expired on %s. Please\n"
        "download a newer beta, or run a standard-release client.\n",
        CliGetTimeString(&expirationtime,1) );
    return 1;
  }
#endif
  return 0;
}

// ----------------------------------------------------------------------

struct thread_param_block
{
  #if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32)
    unsigned long threadID;
  #elif (CLIENT_OS == OS_NETWARE)
    int threadID;
  #elif (CLIENT_OS == OS_BEOS)
    thread_id threadID;
  #elif (CLIENT_OS == OS_MACOS)
    MPTaskID threadID;
  #elif (defined(_POSIX_THREADS_SUPPORTED)) //cputypes.h
    pthread_t threadID;
  #else
    int threadID;
  #endif
  unsigned int threadnum;
  unsigned int numthreads;
  int realthread;
  unsigned int priority;
  int do_suspend;
  int do_refresh;
  int is_suspended;
  unsigned long thread_data1;
  unsigned long thread_data2;
  struct thread_param_block *next;
};

// ----------------------------------------------------------------------

#if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_RISCOS) || \
    (CLIENT_OS == OS_WIN16)

#define NON_PREEMPTIVE_OS

#define NON_PREEMPTIVE_OS_PROFILING /* undef this to use non_preemptive_os
                                 code, but use your own timeslicing method */

// TIMER_GRANULARITY       250000 /* time unit in usecs */
// MIN_RUNS_PER_TIME_GRAIN 250    /* if yield runs less often than this in the
//                                   period specified by TIMER_GRANULARITY
//                                   then we adjust the timeslice downwards */
// MAX_RUNS_PER_TIME_GRAIN 500   /* we adjust the timeslice upwards if the
//                               number of times the process yields > this. */
// INITIAL_TIMESLICE_RC5   128   /* we initialize with this */
// INITIAL_TIMESLICE_DES   256
//                              /* runaway and overflow protection */
// MIN_SANE_TIMESLICE_RC5    256
// MIN_SANE_TIMESLICE_DES    512
// MAX_SANE_TIMESLICE_RC5  16384
// MAX_SANE_TIMESLICE_DES  16384

#if (CLIENT_OS == OS_NETWARE)
  #define TIMER_GRANULARITY       125000
  #define MIN_RUNS_PER_TIME_GRAIN     75 //10, 30, 50
  #define MAX_RUNS_PER_TIME_GRAIN    125 //100
  #define INITIAL_TIMESLICE_RC5      (GetTimesliceBaseline())
  #define INITIAL_TIMESLICE_DES      (GetTimesliceBaseline()<<2)
  #define MIN_SANE_TIMESLICE_RC5     (GetTimesliceBaseline()>>1)
  #define MIN_SANE_TIMESLICE_DES     (GetTimesliceBaseline())
  #define MAX_SANE_TIMESLICE_RC5   16384
  #define MAX_SANE_TIMESLICE_DES   16384
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  #define TIMER_GRANULARITY       125000
  #define MIN_RUNS_PER_TIME_GRAIN     75 //10, 30, 50
  #define MAX_RUNS_PER_TIME_GRAIN    125 //100
  #define INITIAL_TIMESLICE_RC5      (GetTimesliceBaseline())
  #define INITIAL_TIMESLICE_DES      (GetTimesliceBaseline()<<2)
  #define MIN_SANE_TIMESLICE_RC5     (GetTimesliceBaseline()>>1)
  #define MIN_SANE_TIMESLICE_DES     (GetTimesliceBaseline())
  #define MAX_SANE_TIMESLICE_RC5   16384
  #define MAX_SANE_TIMESLICE_DES   16384
#elif (CLIENT_OS == OS_DOS) /* ineffective - used by cyp for testing */
  #define TIMER_GRANULARITY       500000 /* has horrible timer resolution */
  #define MIN_RUNS_PER_TIME_GRAIN     3 // 9 /* 18 times/sec */
  #define MAX_RUNS_PER_TIME_GRAIN     5 //18
  #define INITIAL_TIMESLICE_RC5      GetTimesliceBaseline()
  #define INITIAL_TIMESLICE_DES      (GetTimesliceBaseline()<<2)
  #define MIN_SANE_TIMESLICE_RC5      25
  #define MIN_SANE_TIMESLICE_DES      75
  #define MAX_SANE_TIMESLICE_RC5   16384
  #define MAX_SANE_TIMESLICE_DES   16384
#elif (CLIENT_OS == OS_RISCOS)
  #define TIMER_GRANULARITY       1000000
  #define MIN_RUNS_PER_TIME_GRAIN     10
  #define MAX_RUNS_PER_TIME_GRAIN     30
  #define INITIAL_TIMESLICE_RC5    65536
  #define INITIAL_TIMESLICE_DES    131072
  #define MIN_SANE_TIMESLICE_RC5    256
  #define MIN_SANE_TIMESLICE_DES    256
  #define MAX_SANE_TIMESLICE_RC5   1048576
  #define MAX_SANE_TIMESLICE_DES   1048576
#else
  #error "Unknown OS. Please check timer granularity and timeslice constants"
  #undef NON_PREEMPTIVE_OS_PROFILING  //or undef to do your own profiling
#endif

#endif /* CLIENT_OS == netware, macos, riscos, win16, win32s */

// ----------------------------------------------------------------------

static void yield_pump( void *tv_p )
{
  static int pumps_without_run = 0;
  runstatics.yield_run_count++;

  #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    thr_yield();
  #elif (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_BSDI)
    #if defined(__ELF__)
    sched_yield();
    #else // a.out
    NonPolledUSleep( 0 ); /* yield */
    #endif
  #elif (CLIENT_OS == OS_OS2)
    DosSleep(0);
  #elif (CLIENT_OS == OS_IRIX)
    sginap(0);
  #elif (CLIENT_OS == OS_WIN32)
    w32Yield(); //Sleep(0);
  #elif (CLIENT_OS == OS_DOS)
    dosCliYield(); //dpmi yield
  #elif (CLIENT_OS == OS_NETWARE)
    nwCliThreadSwitchLowPriority();
  #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    w32Yield();
  #elif (CLIENT_OS == OS_RISCOS)
    if (riscos_in_taskwindow)
      { riscos_upcall_6(); }
  #elif (CLIENT_OS == OS_LINUX)
    #if defined(__ELF__)
    sched_yield();
    #else // a.out libc4
    NonPolledUSleep( 0 ); /* yield */
    #endif
  #elif (CLIENT_OS == OS_MACOS)
    // Mac non-MP code yields in problem.cpp because it needs
    // to know the contest
    // MP code yields here because it can do only pure computing
    // (no toolbox or mixed-mode calls)
    tick_sleep(0); /* yield */
  #elif (CLIENT_OS == OS_BEOS)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_OPENBSD)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_NETBSD)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_QNX)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_AIX)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_ULTRIX)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_HPUX)
    sched_yield();;
  #elif (CLIENT_OS == OS_DEC_UNIX)
   #if defined(MULTITHREAD)
     sched_yield();
   #else
     NonPolledUSleep(0);
   #endif
  #else
    #error where is your yield function?
    NonPolledUSleep( 0 ); /* yield */
  #endif

  // used in conjunction with non-threaded go_mt
  if (tv_p)
  {
    if (runstatics.nonmt_ran)
      pumps_without_run = 0;
    #ifdef NON_PREEMPTIVE_OS_PROFILING
    else if ((++pumps_without_run) > 5)
    {
      pumps_without_run = 0;
      LogScreen("Yielding too fast. Doubled pump interval.\n");
      struct timeval *tv = (struct timeval *)tv_p;
      tv->tv_usec<<=1; tv->tv_sec<<=1;
      if (tv->tv_usec>=1000000)
          { tv->tv_sec+=tv->tv_usec/1000000; tv->tv_usec%=1000000; }
    }
    #endif
    if (RegPolledProcedure(yield_pump, tv_p, (struct timeval *)tv_p, 32 )==-1)
    {
      //should never happen, but better safe than sorry...
      LogScreen("Panic! Unable to re-initialize yield pump\n");
      RaiseExitRequestTrigger();
    }
  }
  return;
}

#ifdef NON_PREEMPTIVE_OS_PROFILING
int reset_profiling_flag = 1;
void reset_ts_profiling(void) { reset_profiling_flag = 1; }

unsigned long fixup_timeslice( unsigned long tslice, int contest )
{
  if (contest == 0)
    tslice = (tslice + 0x0E) & 0x0FFFFFFF0L;
  else
  {
    unsigned long n, i = 0;
    for (n = (1<<31); n > 0; n>>=1 )
    {
      if ((tslice & n) != 0)
      {
        i = n;
        break;
      }
    }
    if (i < 0x80)
      i = 0x80;
    tslice = i;
  }
  return tslice;
}

unsigned long do_ts_profiling( unsigned long tslice, int contest, int threadnum )
{
  static struct timeval tvstart = {0,0}, tvstop = {0,0};
  static unsigned long tslice_table[2] = {0,0};
  static unsigned long totalslice_table[2] = {0,0};
  static unsigned long tslice_lkg[2] = {0,0};   /* last-known-good */
  static unsigned long fubared_run_count = 0;
  static unsigned long underrun_count = 0, goodrun_count = 0;

  struct timeval tvnow;
  unsigned long ts;
  CliTimer(&tvnow);

  if (reset_profiling_flag)
  {
    tvstop.tv_sec = 0;
    tvstop.tv_usec = 0;
  }
  if (tvstop.tv_sec == 0 && tvstop.tv_usec == 0)
  {
    if (( tslice_table[0] = tslice_lkg[0] ) == 0 )
      tslice_table[0] = INITIAL_TIMESLICE_RC5;
    if (( tslice_table[1] = tslice_lkg[1] ) == 0 )
      tslice_table[1] = INITIAL_TIMESLICE_DES;
  }
  else if (( tvnow.tv_sec > tvstop.tv_sec ) ||
       (( tvnow.tv_sec == tvstop.tv_sec ) &&
        ( tvnow.tv_usec >= tvstop.tv_usec )))
  {
    tvstop.tv_sec = tvnow.tv_sec;  /* the time we really stopped */
    tvstop.tv_usec = tvnow.tv_usec;

    unsigned long hgrain_run_count = 0; /* yield count in (100 * timerunit) */
    unsigned long perc = 0;
    unsigned long usecs = (tvstop.tv_sec - tvstart.tv_sec) * 1000000L;

    if (tvstop.tv_usec < tvstart.tv_usec)
    {
      if (usecs) /* ie >= 1000000L */
      {
        tvstop.tv_usec += 1000000L;
        usecs -= 1000000L;
      }
      else                                /* timer is running backwards */
        tvstop.tv_usec = tvstart.tv_usec; /* let usecs = 0, and let % fail */
    }
    usecs += (tvstop.tv_usec - tvstart.tv_usec);

    if (usecs) /* will also be zero if running backwards */
      perc = (((unsigned long)(TIMER_GRANULARITY))*100) / usecs;
    if (perc)  /* yield count in one hundred timer units */
      hgrain_run_count = (runstatics.yield_run_count * 10000) / perc;
#ifdef DEBUG
printf("%d. oldslice = %lu, y_real = %lu/%lu, y_adj (%lu%%) = %lu/%lu ",
          threadnum, tslice_table[contest], runstatics.yield_run_count, usecs,
          perc, hgrain_run_count, (unsigned long)(TIMER_GRANULARITY * 100) );
fflush(stdout);
#endif

    if (!perc)
    {
      /* nothing - data is unreliable (user pressed ^S or something) */
    }
    else if (hgrain_run_count == 0) /* badly lagging or timer bad */
    {
      if (((fubared_run_count++) & 0xFF ) == 1)
        Log("Running inefficiently. Timer is possibly bad.\n");
      tslice_table[0] = MIN_SANE_TIMESLICE_RC5;
      tslice_table[1] = MIN_SANE_TIMESLICE_DES;
    }
    else if (hgrain_run_count < (MIN_RUNS_PER_TIME_GRAIN * 100))
    {                             /* so decrease timeslice */
      unsigned long under_par =
        ((MIN_RUNS_PER_TIME_GRAIN * 100) - hgrain_run_count) / 100;

      fubared_run_count = 0;
      goodrun_count = 0;

      if (under_par)  /* change is large enough to warrant adjustement */
      {
        underrun_count++;
        if (under_par == MIN_RUNS_PER_TIME_GRAIN)
        {
#ifdef DEBUG
      printf("under_par: divide by 0!\n");
#endif
          under_par--;
        }
        ts = (totalslice_table[0]/runstatics.yield_run_count)/
                                    (MIN_RUNS_PER_TIME_GRAIN-under_par);
        if (tslice_table[0] > ts)
        {
          tslice_table[0] -= ts;
#ifdef DEBUG
printf("-%lu=> ", ts );
#endif
          if (tslice_table[0] < MIN_SANE_TIMESLICE_RC5)
            tslice_table[0] = MIN_SANE_TIMESLICE_RC5;
          else if ((underrun_count < 3) && tslice_lkg[0] &&
                                       (tslice_lkg[0] > tslice_table[0]))
            tslice_table[0] = tslice_lkg[0];
        }
        ts = (totalslice_table[1]/runstatics.yield_run_count)/
                                    (MIN_RUNS_PER_TIME_GRAIN-under_par);
        if (tslice_table[1] > ts)
        {
          tslice_table[1] -= ts;
          if (tslice_table[1] < MIN_SANE_TIMESLICE_DES)
            tslice_table[1] = MIN_SANE_TIMESLICE_DES;
          else if ((underrun_count < 3) && tslice_lkg[1] &&
                                       (tslice_lkg[1] > tslice_table[1]))
            tslice_table[1] = tslice_lkg[1];
        }
      }
    }
    else if (hgrain_run_count > (MAX_RUNS_PER_TIME_GRAIN * 100))
    {                             /* so increase timeslice */
      unsigned long over_par =
        (hgrain_run_count - (MAX_RUNS_PER_TIME_GRAIN * 100)) / 100;

      fubared_run_count = 0;
      underrun_count = 0;
      goodrun_count = 0;

      if (over_par) /* don't do micro adjustments */
      {
        ts = tslice_table[0];
        if (over_par ==  MAX_RUNS_PER_TIME_GRAIN)
        {
#ifdef DEBUG
          printf("over_par: divide by 0!\n");
#endif
          over_par++;
        }
        tslice_table[0] += (totalslice_table[0]/runstatics.yield_run_count)/
                                     (over_par-MAX_RUNS_PER_TIME_GRAIN);
#ifdef DEBUG
printf("+%u=> ", tslice_table[0]-ts );
#endif

        if (tslice_table[0] > MAX_SANE_TIMESLICE_RC5)
          tslice_table[0] = MAX_SANE_TIMESLICE_RC5;
        else if ( tslice_table[0] < tslice_lkg[0])
          tslice_table[0] = tslice_lkg[0];

        tslice_table[1] += (totalslice_table[1]/runstatics.yield_run_count)/
                                     (over_par-MAX_RUNS_PER_TIME_GRAIN);
        if (tslice_table[1] > MAX_SANE_TIMESLICE_DES)
          tslice_table[1] = MAX_SANE_TIMESLICE_DES;
        else if ( tslice_table[1] < tslice_lkg[1])
          tslice_table[1] = tslice_lkg[1];
      }
    }
    else
    {
      fubared_run_count = 0;
      underrun_count = 0;

      ts = (totalslice_table[0]/runstatics.yield_run_count);
      if (ts > tslice_lkg[0])
        {
        tslice_lkg[0] = ts;
        if ( tslice_lkg[0] < MIN_SANE_TIMESLICE_RC5 )
          tslice_lkg[0] =  MIN_SANE_TIMESLICE_RC5;
      }
      tslice_table[0] = tslice_lkg[0] + ((tslice_lkg[0]/10) * goodrun_count);
      ts = (totalslice_table[1]/runstatics.yield_run_count);
      if (ts > tslice_lkg[1])
      {
        tslice_lkg[1] = ts;
        if ( tslice_lkg[1] < MIN_SANE_TIMESLICE_DES )
          tslice_lkg[0] =  MIN_SANE_TIMESLICE_DES;
      }
      tslice_table[1] = tslice_lkg[1] + ((tslice_lkg[1]/10) * goodrun_count);
      goodrun_count++;
#ifdef DEBUG
printf("+-0=> " );
#endif
    }
    tvstop.tv_sec = 0;
    tvstop.tv_usec = 0;
#ifdef DEBUG
printf("%u\n", tslice_table[contest] );
#endif
  }
  if (tvstop.tv_sec == 0 && tvstop.tv_usec == 0)
  {
    totalslice_table[0] = threadnum; /* dummy code to use up the variable */
    totalslice_table[0] = 0;
    totalslice_table[1] = 0;
    tvstop.tv_sec  = tvstart.tv_sec = tvnow.tv_sec;
    tvstop.tv_usec = tvstart.tv_usec = tvnow.tv_usec;

    tvstop.tv_usec += TIMER_GRANULARITY;
    if (tvstop.tv_usec >= 1000000L)
    {
      tvstop.tv_sec += tvstop.tv_usec/1000000L;
      tvstop.tv_usec %= 1000000L;
    }
    runstatics.yield_run_count = 0;
    reset_profiling_flag = 0;
  }

  if (tslice != tslice_table[contest])
  {
    tslice_table[contest] = fixup_timeslice( tslice_table[contest], contest );
    tslice = tslice_table[contest];
  }
  totalslice_table[contest]+=tslice;

  #if (CLIENT_OS == OS_NETWARE)
  yield_pump(NULL);
  #endif

  return tslice;
}
#endif

// ----------------------------------------------------------------------

#if (CLIENT_OS == OS_MACOS)
  OSStatus Go_mt( void * parm )
#else
  void Go_mt( void * parm )
#endif
{
  struct thread_param_block *targ = (thread_param_block *)parm;
  Problem *thisprob = NULL;
  unsigned int threadnum = targ->threadnum;

#if (CLIENT_OS == OS_RISCOS)
/*if (threadnum == 1)
  {
    thisprob = GetProblemPointerFromIndex(threadnum);
    thisprob->Run();
    return;
  } */
#elif (CLIENT_OS == OS_WIN32)
  if (targ->realthread)
  {
    DWORD LAffinity, LProcessAffinity, LSystemAffinity;
    OSVERSIONINFO osver;
    unsigned int numthreads = targ->numthreads;

    osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
    GetVersionEx(&osver);
    if ((VER_PLATFORM_WIN32_NT == osver.dwPlatformId) && (numthreads > 1))
    {
      if (GetProcessAffinityMask(GetCurrentProcess(), &LProcessAffinity, &LSystemAffinity))
      {
        LAffinity = 1L << threadnum;
        if (LProcessAffinity & LAffinity)
          SetThreadAffinityMask(GetCurrentThread(), LAffinity);
      }
    }
  }
#elif (CLIENT_OS == OS_NETWARE)
if (targ->realthread)
  {
    nwCliInitializeThread( threadnum+1 ); //in netware.cpp
  }
#elif (CLIENT_OS == OS_MACOS)
  if (targ->realthread)
  {
    MPEnterCriticalRegion(MP_count_region, kDurationForever);
    MP_active++;
    MPExitCriticalRegion(MP_count_region);
  }
#elif (defined(_POSIX_THREADS_SUPPORTED)) //cputypes.h
  if (targ->realthread)
  {
    sigset_t signals_to_block;
    sigemptyset(&signals_to_block);
    sigaddset(&signals_to_block, SIGINT);
    sigaddset(&signals_to_block, SIGTERM);
    sigaddset(&signals_to_block, SIGKILL);
    sigaddset(&signals_to_block, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &signals_to_block, NULL);
  }
#endif

  if (targ->realthread)
    SetThreadPriority( targ->priority ); /* 0-9 */

  targ->is_suspended = 1;
  targ->do_refresh = 1;

  while (!CheckExitRequestTriggerNoIO())
  {
    int didwork = 0; /* did we work? */
    if (targ->do_refresh)
      thisprob = GetProblemPointerFromIndex(threadnum);
    if (thisprob == NULL || targ->do_suspend || CheckPauseRequestTriggerNoIO())
    {
//printf("run: isnull? %08x, ispausereq? %d, issusp? %d\n", thisprob, CheckPauseRequestTriggerNoIO(), targ->do_suspend);
      if (thisprob == NULL)  // this is a bad condition, and should not happen
        runstatics.refillneeded = 1;// ..., ie more threads than problems
      if (targ->realthread)
      {
        #if (CLIENT_OS == OS_MACOS)
           mp_sleep(1);     // Mac needs special sleep call in MP threads
        #else
           NonPolledSleep(1); // don't race in this loop
        #endif
      }
    }
    else if (!thisprob->IsInitialized())
    {
//printf("run: not initialized\n");
      runstatics.refillneeded = 1;
      if (targ->realthread)
        yield_pump(NULL);
    }
    else
    {
//printf("run: doing run\n");
      static struct 
      {  unsigned int contest; u32 msec, max, min; volatile u32 optimal;
      } dyn_timeslice[CONTEST_COUNT] = {
        {  RC5, 1000, 0x80000000,  0x00100,  0x10000 },
        {  DES, 1000, 0x80000000,  0x00100,  0x10000 },
        {  OGR,    OGR_TIMESLICE_MSEC, OGR_TIMESLICE_MAX, 0x0100,  0x1000 },
        {  CSC, 1000, 0x80000000,  0x00100,  0x10000 }
      };  
      int run; u32 optimal_timeslice = 0; u32 runtime_ms;
      unsigned int contest_i = thisprob->contest;
      u32 last_count = thisprob->core_run_count; 
                  
      #ifdef NON_PREEMPTIVE_OS_PROFILING
      thisprob->tslice = do_ts_profiling(thisprob->tslice,contest_i,threadnum);
      optimal_timeslice = 0;
      #elif (CLIENT_OS == OS_MACOS)
      thisprob->tslice = GetTimesliceToUse(contest_i);
      optimal_timeslice = 0;
      #else
      #if (!defined(DYN_TIMESLICE))
      if (contest_i == OGR)
      #endif
      {
        if (last_count == 0) /* prob hasn't started yet */
          thisprob->tslice = dyn_timeslice[contest_i].optimal;
        optimal_timeslice = thisprob->tslice;
      }
      #endif

      runtime_ms = (thisprob->runtime_sec*1000 + thisprob->runtime_usec/1000);
      targ->is_suspended = 0;
      run = thisprob->Run();
      targ->is_suspended = 1;
      runtime_ms = (thisprob->runtime_sec*1000 + thisprob->runtime_usec/1000) - runtime_ms;

      didwork = (last_count != thisprob->core_run_count);
      if (run != RESULT_WORKING)
      {
        runstatics.refillneeded = 1;
        if (!didwork && targ->realthread)
          yield_pump(NULL);
      }
      
      if (optimal_timeslice != 0) /* we are profiling for preemptive OSs */
      {
        optimal_timeslice = thisprob->tslice; /* get the number done back */
#if defined(DYN_TIMESLICE_SHOWME)
printf("timeslice: %ld  time: %ldms  working? %d\n",(long)optimal_timeslice, (long)runtime_ms, (run==RESULT_WORKING) );
#endif
        if (run == RESULT_WORKING) /* timeslice/time is invalid otherwise */
        {
          if (runtime_ms < (dyn_timeslice[contest_i].msec /* >>1 */))
          {
            optimal_timeslice <<= 1;
            if (optimal_timeslice > dyn_timeslice[contest_i].max)
              optimal_timeslice = dyn_timeslice[contest_i].max;
          }
          else if (runtime_ms > (dyn_timeslice[contest_i].msec /* <<1 */))
          {
            optimal_timeslice -= (optimal_timeslice>>2);
            if (optimal_timeslice == 0)
              optimal_timeslice = dyn_timeslice[contest_i].min;
          }
          thisprob->tslice = optimal_timeslice; /* for the next round */
        }
        else /* ok, we've finished. so save it */
        {  
          u32 opt = dyn_timeslice[contest_i].optimal;
          if (optimal_timeslice > opt)
            dyn_timeslice[contest_i].optimal = optimal_timeslice;
          optimal_timeslice = 0; /* reset for the next prob */
        }
      }

    }
    
    if (!didwork)
    {
      #ifdef NON_PREEMPTIVE_OS_PROFILING
      reset_ts_profiling();
      #endif
      targ->do_refresh = 1;
    }
    if (!targ->realthread)
    {
//printf("run: rereg'd\n");    
      RegPolledProcedure( (void (*)(void *))Go_mt, parm, NULL, 0 );
      runstatics.nonmt_ran = didwork;
      break;
    }
  }

  targ->threadID = 0; //the thread is dead

  #if (CLIENT_OS == OS_BEOS)
  if (targ->realthread)
    exit(0);
  #endif

  #if (CLIENT_OS == OS_MACOS)
  if (targ->realthread)
  {
    ThreadIsDone[threadnum] = 1;
    MPEnterCriticalRegion(MP_count_region, kDurationForever);
    MP_active--;
    MPExitCriticalRegion(MP_count_region);
  }
  return(noErr);
  #endif
}

// -----------------------------------------------------------------------

static int __StopThread( struct thread_param_block *thrparams )
{
  if (thrparams)
  {
    yield_pump(NULL);   //give threads some air
    if (thrparams->threadID) //thread did not exit by itself
    {
      if (thrparams->realthread) //real thread
      {
        #if (CLIENT_OS == OS_OS2)
        DosSetPriority( 2, PRTYC_REGULAR, 0, 0); /* thread to normal prio */
        DosWaitThread( &(thrparams->threadID), DCWW_WAIT);
        #elif (CLIENT_OS == OS_WIN32)
        while (thrparams->threadID) Sleep(100);
        #elif (CLIENT_OS == OS_BEOS)
        static status_t be_exit_value;
        wait_for_thread(thrparams->threadID, &be_exit_value);
        #elif (CLIENT_OS == OS_NETWARE)
        while (thrparams->threadID) delay(100);
        #elif (CLIENT_OS == OS_MACOS)
        #error use while (thrparams->threadID) tick_sleep(60); here
        while (ThreadIsDone[thrparams->threadnum] == 0) tick_sleep(60);
        #elif (defined(_POSIX_THREADS_SUPPORTED)) //cputypes.h
        pthread_join( thrparams->threadID, (void **)NULL);
        #endif
      }
    }
    ClientEventSyncPost( CLIEVENT_CLIENT_THREADSTOPPED, (long)thrparams->threadnum );
    free( thrparams );
  }
  return 0;
}

// -----------------------------------------------------------------------

static struct thread_param_block *__StartThread( unsigned int thread_i,
        unsigned int numthreads, unsigned int priority, int no_realthreads )
{
  int success = 1, use_poll_process = 0;

  struct thread_param_block *thrparams = (struct thread_param_block *)
                         malloc( sizeof(struct thread_param_block) );
  if (thrparams)
  {
    // Start the thread for this cpu
    memset( (void *)(thrparams), 0, sizeof( struct thread_param_block ));
    thrparams->threadID = 0;              /* whatever type */
    thrparams->numthreads = numthreads;   /* unsigned int */
    thrparams->threadnum = thread_i;      /* unsigned int */
    thrparams->realthread = 1;            /* int */
    thrparams->priority = priority;       /* unsigned int */
    thrparams->do_suspend = 0;
#if (CLIENT_OS == OS_RISCOS)
    thrparams->do_suspend = /*thread_i?1:*/0;
#endif
    thrparams->is_suspended = 0;
    thrparams->do_refresh = 1;
    thrparams->thread_data1 = 0;          /* ulong, free for thread use */
    thrparams->thread_data2 = 0;          /* ulong, free for thread use */
    thrparams->next = NULL;

    use_poll_process = 0;

    if ( no_realthreads )
      use_poll_process = 1;
    else
    {
      #if (!defined(CLIENT_SUPPORTS_SMP)) //defined in cputypes.h
        use_poll_process = 1; //no thread support or cores are not thread safe
      #elif (CLIENT_OS == OS_WIN32)
        unsigned int thraddr;
        thrparams->threadID = _beginthread( Go_mt, 8192, (void *)thrparams );
        success = ( (thrparams->threadID) != 0);
      #elif (CLIENT_OS == OS_OS2)
        thrparams->threadID = _beginthread( Go_mt, NULL, 8192, (void *)thrparams );
        success = ( thrparams->threadID != -1);
      #elif (CLIENT_OS == OS_NETWARE)
        if (!nwCliIsSMPAvailable())
          use_poll_process = 1;
        else
          success = ((thrparams->threadID = BeginThread( Go_mt, NULL, 8192,
                                   (void *)thrparams )) != -1);
      #elif (CLIENT_OS == OS_BEOS)
        char thread_name[32];
        long be_priority = thrparams->priority+1;
        // Be OS priority for rc5des should be adjustable from 1 to 10
        // 1 is lowest, 10 is higest for non-realtime and non-system tasks
        sprintf(thread_name, "RC5DES crunch#%d", thread_i + 1);
        thrparams->threadID = spawn_thread((long (*)(void *)) Go_mt,
               thread_name, be_priority, (void *)thrparams );
        if ( ((thrparams->threadID) >= B_NO_ERROR) &&
             (resume_thread(thrparams->threadID) == B_NO_ERROR) )
          success = 1;
      #elif (CLIENT_OS == OS_MACOS)
        OSErr thread_error;
        MPTaskID new_threadid;
        ThreadIsDone[thread_i] = 0;
        thread_error = MPCreateTask(Go_mt, (void *)thrparams, (unsigned long)0, (OpaqueMPQueueID *)kMPNoID,
                (void *)0, (void *)0, (unsigned long)0, &new_threadid);
        if (thread_error != noErr)
          new_threadid = NULL;
        else
          success = 1;

        thrparams->threadID = new_threadid;
        #if defined(MAC_GUI)
        #error please change this to use event posted in the if (success) section below
        #error CalcPercent() and GetKeysdone() may return unexpected results
        if (success)
        {
          Problem *thisprob;
          thisprob = GetProblemPointerFromIndex( thread_i );
          MakeGUIThread(thisprob->contest, thread_i);
          InitializeThreadProgress(thread_i, thisprob->CalcPercent(),
                     thisprob->GetKeysDone());
          UpdateClientInfo(&client_info);
        }
        #endif
      #elif defined(_POSIX_THREADS_SUPPORTED) //defined in cputypes.h
        #if defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
          SetGlobalPriority( thrparams->priority );
          pthread_attr_t thread_sched;
          pthread_attr_init(&thread_sched);
          pthread_attr_setscope(&thread_sched,PTHREAD_SCOPE_SYSTEM);
          pthread_attr_setinheritsched(&thread_sched,PTHREAD_INHERIT_SCHED);
          if (pthread_create( &(thrparams->threadID), &thread_sched,
                (void *(*)(void*)) Go_mt, (void *)thrparams ) == 0)
            success = 1;
          SetGlobalPriority( 9 ); //back to normal
        #else
          if (pthread_create( &(thrparams->threadID), NULL,
             (void *(*)(void*)) Go_mt, (void *)thrparams ) == 0 )
            success = 1;
        #endif
      #else
        use_poll_process = 1;
      #endif
    }

    if (use_poll_process)
    {
      thrparams->realthread = 0;            /* int */
      #if (CLIENT_OS == OS_MACOS)
      thrparams->threadID = (MPTaskID)RegPolledProcedure((void (*)(void *))Go_mt,
                                (void *)thrparams , NULL, 0 );
      #else
      thrparams->threadID = RegPolledProcedure(Go_mt,
                                (void *)thrparams , NULL, 0 );
      #endif
      success = ((int)thrparams->threadID != -1);
    }

    if (success)
    {
      ClientEventSyncPost( CLIEVENT_CLIENT_THREADSTARTED, (long)thread_i );
      #if (CLIENT_OS == OS_MACOS) && defined(MAC_GUI)
      {
        #error please change this to use event posted in the if (success) section above
        #error CalcPercent() and GetKeysdone() may return unexpected results
        Problem *thisprob;
        thisprob = GetProblemPointerFromIndex( 0 );
        MakeGUIThread(thisprob->contest, 0);
        InitializeThreadProgress(0, thisprob->CalcPermille(),
                  thisprob->GetKeysDone());
        UpdateClientInfo(&client_info);
      }
      #endif
      yield_pump(NULL);   //let the thread start
    }
    else
    {
      free( thrparams );
      thrparams = NULL;
    }
  }
  return thrparams;
}


// -----------------------------------------------------------------------

// returns:
//    -2 = exit by error (all contests closed)
//    -1 = exit by error (critical)
//     0 = exit for unknown reason
//     1 = exit by user request
//     2 = exit by exit file check
//     3 = exit by time limit expiration
//     4 = exit by block count expiration
int Client::Run( void )
{
  unsigned int prob_i;
  int force_no_realthreads = 0;
  struct thread_param_block *thread_data_table = NULL;

  int TimeToQuit = 0, exitcode = 0;
  int local_connectoften = 0;
  unsigned int load_problem_count = 0;
  unsigned int getbuff_errs = 0;

  time_t timeNow;
  time_t timeRun=0, timeLast=0, timeNextConnect=0, timeNextCheckpoint = 0;

  time_t last_scheduledupdatetime = 0; /* we reset the next two vars on != */
  //unsigned int flush_scheduled_count = 0; /* used for exponential staging */
  unsigned int flush_scheduled_adj   = 0; /*time adjustment to next schedule*/
  time_t ignore_scheduledupdatetime_until = 0; /* ignore schedupdtime until */

  int checkpointsDisabled = (nodiskbuffers != 0);
  unsigned int checkpointsPercent = 0;
  int isPaused=0, wasPaused=0;

  ClientEventSyncPost( CLIEVENT_CLIENT_RUNSTARTED, 0 );

  // =======================================
  // Notes:
  //
  // Do not break flow with a return()
  // [especially not after problems are loaded]
  //
  // Code order:
  //
  // Initialization: (order is important, 'F' denotes code that can fail)
  // 1.    UndoCheckpoint() (it is not affected by TimeToQuit)
  // 2.    Determine number of problems
  // 3. F  Load (or try to load) that many problems (needs number of problems)
  // 4. F  Initialize polling process (needed by threads)
  // 5. F  Spin up threads
  // 6.    Unload over-loaded problems (problems for which we have no worker)
  // 7.    Initialize percent bar (needs final problem table size)
  //
  // Run...
  //
  // Deinitialization:
  // 8. Shut down threads
  // 9. Deinitialize polling process
  // 10. Unload problems
  // 11. Throw away checkpoints
  // =======================================

  // --------------------------------------
  // Recover blocks from checkpoint files before anything else
  // we always recover irrespective of TimeToQuit
  // --------------------------------------

  if (!checkpointsDisabled) //!nodiskbuffers
  {
    if (CheckpointAction( CHECKPOINT_OPEN, 0 )) //-> !0 if checkpts disabled
    {
      checkpointsDisabled = 1;
    }
  }

  // --------------------------------------
  // BETA check
  // --------------------------------------

  if (!TimeToQuit && checkifbetaexpired()!=0) //prints a message
  {
    TimeToQuit = 1;
    exitcode = -1;
  }

  // --------------------------------------
  // Determine the number of problems to work with. Number is used everywhere.
  // --------------------------------------

  if (!TimeToQuit)
  {
    force_no_realthreads = 0; /* this is a hint. it does not reflect capability */
    unsigned int numcrunchers = (unsigned int)numcpu;

    #if (CLIENT_OS == OS_FREEBSD && CLIENT_OS_MINOR != 4)
    if (numcrunchers > 1)
    {
      LogScreen("FreeBSD threads are not SMP aware (do not automatically\n"
                "migrate to distribute processor load). Please run one\n"
                "client per processor.\n");
      numcrunchers = 1;
    }
    #endif
    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_OS2) || (CLIENT_OS==OS_BEOS)
    if (numcrunchers == 0) // must run with real threads because the
      numcrunchers = 1;    // main thread runs at normal priority
    #endif
    #if (CLIENT_OS == OS_NETWARE)
    if (numcrunchers == 1) // NetWare client prefers non-threading
      numcrunchers = 0;    // if only one thread/processor is to used
    #endif

    #if (CLIENT_OS == OS_MACOS)
    if ((!haveMP) || (!useMP())) numcrunchers = 0;  // no real threads if MP not present or not wanted
    #endif

    if (numcrunchers < 1) /* == 0 = user requested non-mt */
    {
      force_no_realthreads = 1;
      numcrunchers = 1;
    }
    if (numcrunchers > GetNumberOfSupportedProcessors()) //max by cli instance
      numcrunchers = GetNumberOfSupportedProcessors();   //not by platform
    load_problem_count = numcrunchers;
  }

  // -------------------------------------
  // load (or rather, try to load) that many problems
  // -------------------------------------

  if (!TimeToQuit)
  {
    if (load_problem_count > 1)
      Log( "Loading one block per cruncher...\n" );
    load_problem_count = LoadSaveProblems( load_problem_count, 0 );

    if (CheckExitRequestTrigger())
    {
      TimeToQuit = 1;
      exitcode = -2;
    } 
    else if (load_problem_count == 0)
    {
    Log("Unable to load any blocks. Quitting...\n");
    TimeToQuit = 1;
    exitcode = -2;
    }
  }

  // --------------------------------------
  // Initialize the async "process" subsystem
  // --------------------------------------

  if (!TimeToQuit && InitializePolling())
  {
    Log( "Unable to initialize async subsystem.\n");
    TimeToQuit = 1;
    exitcode = -1;
  }

  // -------------------------------
  // create a yield pump for OSs that need one
  // -------------------------------

  #if defined(NON_PREEMPTIVE_OS)
  if (!TimeToQuit)
  {
    static struct timeval tv = {0,500};

    if (RegPolledProcedure(yield_pump, (void *)&tv, (timeval *)&tv, 32 ) == -1)
    {
      Log("Unable to initialize yield pump\n" );
      TimeToQuit = -1;
      exitcode = -1;
    }
  }
  #endif

  // --------------------------------------
  // Spin up the crunchers
  // --------------------------------------

  if (!TimeToQuit)
  {
    struct thread_param_block *thrparamslast = thread_data_table;
    unsigned int planned_problem_count = load_problem_count;
    load_problem_count = 0;

    for ( prob_i = 0; prob_i < planned_problem_count; prob_i++ )
    {
      struct thread_param_block *thrparams =
         __StartThread( prob_i, planned_problem_count,
                        priority, force_no_realthreads );
      if ( thrparams )
      {
        if (!thread_data_table)
          thread_data_table = thrparams;
        else
          thrparamslast->next = thrparams;
        thrparamslast = thrparams;
        load_problem_count++;
      }
      else
      {
        break;
      }
    }

    if (load_problem_count == 0)
    {
      Log("Unable to initialize cruncher(s). Quitting...\n");
      TimeToQuit = 1;
      exitcode = -1;
    }
    else
    {
      char srange[20];
      srange[0] = 0;
      if (load_problem_count > 1 && load_problem_count <= 26 /* a-z */)
      {
        sprintf( srange, " ('a'%s'%c')",
          ((load_problem_count==2)?(" and "):("-")),
          'a'+(load_problem_count-1) );
      }
      Log("%u cruncher%s%s ha%s been started.%c%c%u failed to start)\n",
             load_problem_count,
             ((load_problem_count==1)?(""):("s")), srange,
             ((load_problem_count==1)?("s"):("ve")),
             ((load_problem_count < planned_problem_count)?(' '):('\n')),
             ((load_problem_count < planned_problem_count)?('('):(0)),
             (planned_problem_count - load_problem_count) );
    }

    // resize the problem table if we've loaded too much
    if (load_problem_count < planned_problem_count)
    {
      if (TimeToQuit)
        LoadSaveProblems(planned_problem_count,PROBFILL_UNLOADALL);
      else
        LoadSaveProblems(load_problem_count, PROBFILL_RESIZETABLE);
    }
  }

  //------------------------------------
  // display the percent bar so the user sees some action
  //------------------------------------

  if (!TimeToQuit && !percentprintingoff)
  {
    LogScreenPercent( load_problem_count ); //logstuff.cpp
  }

  //============================= MAIN LOOP =====================
  //now begin looping until we have a reason to quit
  //------------------------------------

  // -- cramer - until we have a better way of telling how many blocks
  //             are loaded and if we can get more, this is gonna be a
  //             a little complicated.  getbuff_errs and nonewblocks
  //             control the exit process.  getbuff_errs indicates the
  //             number of attempts to load new blocks that failed.
  //             nonewblocks indcates that we aren't get anymore blocks.
  //             Together, they can signal when the buffers have been
  //             truely exhausted.  The magic below is there to let
  //             the client finish processing those blocks before exiting.

  // Start of MAIN LOOP
  while (TimeToQuit == 0)
  {
    //------------------------------------
    //sleep, run or pause...
    //------------------------------------

    SetGlobalPriority( priority );
    if (isPaused)
      sleep(3);
    else
    {
      int i = 0;
      while ((i++)<5
            && !runstatics.refillneeded
            && !CheckExitRequestTriggerNoIO()
            && ModeReqIsSet(-1)==0)
        sleep(1);
    }
    SetGlobalPriority( 9 );

    //------------------------------------
    // Fixup timers
    //------------------------------------

    timeNow = CliTimer(NULL)->tv_sec;
    if (timeLast!=0 && ((unsigned long)timeNow) > ((unsigned long)timeLast))
      timeRun += (timeNow - timeLast); //make sure time is monotonic
    timeLast = timeNow;

    //----------------------------------------
    // Check for time limit...
    //----------------------------------------

    if ( !TimeToQuit && (minutes > 0) && (timeRun > (time_t)( minutes*60 )))
    {
      Log( "Shutdown - reached time limit.\n" );
      TimeToQuit = 1;
      exitcode = 3;
    }

    //----------------------------------------
    // Has -runbuffers exhausted all buffers?
    //----------------------------------------

    // cramer magic (voodoo)
    if (!TimeToQuit && nonewblocks > 0 &&
      ((unsigned int)getbuff_errs >= load_problem_count))
    {
      TimeToQuit = 1;
      exitcode = 4;
    }

    //----------------------------------------
    // Check for user break
    //----------------------------------------

    #if defined(BETA)
    if (!TimeToQuit && !CheckExitRequestTrigger() && checkifbetaexpired()!=0)
    {
      TimeToQuit = 1;
      exitcode = -1;
    }
    #endif
    if (!TimeToQuit && CheckExitRequestTrigger())
    {
      Log( "%s...\n",
         (CheckRestartRequestTrigger()?("Restarting"):("Shutting down")) );
      TimeToQuit = 1;
      exitcode = 1;
    }
    if (!TimeToQuit)
    {
      isPaused = CheckPauseRequestTrigger();
      if (isPaused)
      {
        if (!wasPaused)
          LogScreen("Paused...\n");
        wasPaused = 1;
      }
      else if (wasPaused)
      {
        LogScreen("Running again after pause...\n");
        wasPaused = 0;
      }
    }

    //------------------------------------
    //update the status bar, check all problems for change, do reloading etc
    //------------------------------------

    if (!TimeToQuit && !isPaused)
    {
      if (!percentprintingoff)
        LogScreenPercent( load_problem_count ); //logstuff.cpp
      getbuff_errs+=LoadSaveProblems(load_problem_count,PROBFILL_GETBUFFERRS);
      runstatics.refillneeded = 0;

      if (CheckExitRequestTriggerNoIO())
        continue;
    }

    //------------------------------------
    // Check for universally coordinated update
    //------------------------------------

    #define TIME_AFTER_START_TO_UPDATE 10800 // Three hours
    #define UPDATE_INTERVAL 600 // Ten minutes

    if (!TimeToQuit && scheduledupdatetime != 0 && 
      (((unsigned long)timeNow) < ((unsigned long)ignore_scheduledupdatetime_until)) &&
      (((unsigned long)timeNow) >= ((unsigned long)scheduledupdatetime)) &&
      (((unsigned long)timeNow) < (((unsigned long)scheduledupdatetime)+TIME_AFTER_START_TO_UPDATE)) )
    {
      if (last_scheduledupdatetime != ((time_t)scheduledupdatetime))
      {
        last_scheduledupdatetime = (time_t)scheduledupdatetime;
        //flush_scheduled_count = 0;
        flush_scheduled_adj = (rand()%UPDATE_INTERVAL);
        Log("Buffer update scheduled in %u minutes %02u seconds.\n",
             flush_scheduled_adj/60, flush_scheduled_adj%60 );
        flush_scheduled_adj += timeNow - last_scheduledupdatetime;
      }
      if ( (((unsigned long)flush_scheduled_adj) < TIME_AFTER_START_TO_UPDATE) &&
        (((unsigned long)timeNow) >= (unsigned long)(flush_scheduled_adj+last_scheduledupdatetime)) )
      {
        //flush_scheduled_count++; /* for use with exponential staging */
        flush_scheduled_adj += ((UPDATE_INTERVAL>>1)+
                               (rand()%(UPDATE_INTERVAL>>1)));
        
        int desisrunning = 0;
        if (GetBufferCount(DES, 0/*in*/, NULL) != 0) /* do we have DES blocks? */
          desisrunning = 1;
        else
        {
          for (prob_i = 0; prob_i < load_problem_count; prob_i++ )
          {
            Problem *thisprob = GetProblemPointerFromIndex( prob_i );
            if (thisprob == NULL)
              break;
            if (thisprob->IsInitialized() && thisprob->contest == 1)
            {
              desisrunning = 1;
              break;
            }
          }
          if (desisrunning == 0)
          {
            int rc = BufferUpdate( BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH, 0 );
            if (rc > 0 && (rc & BUFFERUPDATE_FETCH)!=0)
              desisrunning = (GetBufferCount( DES, 0/*in*/, NULL) != 0);
          }
        }  
        if (desisrunning)
        {
          ignore_scheduledupdatetime_until = timeNow + TIME_AFTER_START_TO_UPDATE;
          /* if we got DES blocks, start ignoring sched update time */
        }
      }
    } 

    //----------------------------------------
    // If not quitting, then write checkpoints
    //----------------------------------------

    if (!TimeToQuit && !checkpointsDisabled && !CheckPauseRequestTrigger())
    {
      if (timeRun > timeNextCheckpoint)
      {
        unsigned long perc_now = 0;
        unsigned int probs_counted = 0;
        for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
        {
          Problem *thisprob = GetProblemPointerFromIndex(prob_i);
          if ( thisprob )
          {
            perc_now += ((thisprob->CalcPermille() + 5)/10);
            probs_counted++;
          }
        }
        perc_now /= probs_counted;

//LogScreen("ckpoint refresh check. %d%% dif\n", 
//                           abs((int)(checkpointsPercent - ((int)perc_now))));
        if ( ( timeNextCheckpoint == 0 ) || ( CHECKPOINT_FREQ_PERCDIFF < 
          abs((int)(checkpointsPercent - ((unsigned int)perc_now))) ) )
        {
          checkpointsPercent = (unsigned int)perc_now;
          if (CheckpointAction( CHECKPOINT_REFRESH, load_problem_count ))
            checkpointsDisabled = 1;
          timeNextCheckpoint = timeRun + (time_t)(CHECKPOINT_FREQ_SECSDIFF);
//LogScreen("next refresh in %u secs\n", CHECKPOINT_FREQ_SECSDIFF);
        }
      }
    }

    //------------------------------------
    // Lurking
    //------------------------------------

    local_connectoften = (connectoften != 0);
    #if defined(LURK)
    if (dialup.lurkmode)
    {
      connectoften = 0;
      local_connectoften = (!TimeToQuit && 
      !ModeReqIsSet(MODEREQ_FETCH|MODEREQ_FLUSH) &&
      dialup.CheckIfConnectRequested());
    }
    #endif

    //------------------------------------
    //handle 'connectoften' requests
    //------------------------------------

    if (!TimeToQuit && local_connectoften && timeRun > timeNextConnect)
    {
      int doupd = 1;
      if (timeNextConnect != 0)
      {
        int i;
        for (i = 0; i < CONTEST_COUNT; i++ )
        {
          unsigned cont_i = (unsigned int)loadorder_map[i];
          if (cont_i < CONTEST_COUNT) /* not disabled */
          {
            if (GetBufferCount( cont_i, 1, NULL ) > 0) 
              break;  /* at least one out-buffer is not empty */
            if (GetBufferCount( cont_i, 0, NULL ) >= 
               ((long)(inthreshold[cont_i]))) /*at least one in-buffer is full*/
            { 
              doupd = 0;
              break;
            }
          }
        }
      }
      if ( doupd )
      {
        ModeReqSet(MODEREQ_FETCH|MODEREQ_FLUSH|MODEREQ_FQUIET);
      }
      timeNextConnect = timeRun + 60;
    }

    //----------------------------------------
    // If not quitting, then handle mode requests
    //----------------------------------------

    if (!TimeToQuit && ModeReqIsSet(-1))
    {
      //For interactive benchmarks, assume that we have "normal priority"
      //at this point and threads are running at lower priority. If that is
      //not the case, then benchmarks are going to return wrong results.
      //The messy way around that is to suspend the threads.
      ModeReqRun(this);
    }
  }  // End of MAIN LOOP

  //======================END OF MAIN LOOP =====================

  RaiseExitRequestTrigger(); // will make other threads exit

  // ----------------
  // Shutting down: shut down threads
  // ----------------

  if (thread_data_table)  //we have threads running
  {
    LogScreen("Waiting for threads to end...\n");
    while (thread_data_table)
    {
      struct thread_param_block *thrdatap = thread_data_table;
      thread_data_table = thrdatap->next;
      __StopThread( thrdatap );
    }
  }

  // ----------------
  // Close the async "process" handler
  // ----------------

  DeinitializePolling();

  // ----------------
  // Shutting down: save prob buffers, flush if nodiskbuffers, kill checkpts
  // ----------------

   LoadSaveProblems(load_problem_count, PROBFILL_UNLOADALL );
   CheckpointAction( CHECKPOINT_CLOSE, 0 ); /* also done by LoadSaveProb */

  ClientEventSyncPost( CLIEVENT_CLIENT_RUNFINISHED, 0 );

  #if (CLIENT_OS == OS_VMS)
    nice(0);
  #endif

  return exitcode;
}

// ---------------------------------------------------------------------------

