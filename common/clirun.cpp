// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: clirun.cpp,v $
// Revision 1.21  1998/10/31 22:36:11  silby
// Hack to get freebsd-mt to build.  It seems to work fine, but should *really* be looked over by a pthreads familiar person.
//
// Revision 1.20  1998/10/30 12:00:20  cyp
// Fixed a missing do_suspend=0 initialization.
//
// Revision 1.19  1998/10/27 22:22:27  remi
// Added a missing '\'.
//
// Revision 1.18  1998/10/27 19:37:12  cyp
// Synchronized again.
//
// Revision 1.17  1998/10/27 00:43:04  remi
// OLD_NICENCESS -> OLDNICENESS
//
// Revision 1.16  1998/10/27 00:39:24  remi
// Added a few #ifdef to allow compilation of a glibc client without threads.
// #ifdef'ed the old niceness/priority code - removed ::SetContestDoneState() 
// (already in buffupd.cpp)
//
// Revision 1.15  1998/10/25 11:26:35  silby
// Added call to yield_pump in go_mt for win32 so that new CLI is responsive.
//
// Revision 1.143 1998/10/21 12:50:04  cyp
// Promoted u8 contestids in Fetch/Flush/Update to unsigned ints.
//
// Revision 1.142 1998/10/20 17:26:43  remi
// Added 3 missing #ifdef(MULTITHREAD) to allow compilation of a non-mt client
// on glibc-based systems.
//
// Revision 1.141 1998/10/18 21:51:26  dbaker
// added yield for freebsd
//
// Revision 1.14  1998/10/24 15:24:37  sampo
// Added MacOS yielding code
//
// Revision 1.13  1998/10/11 05:26:47  cyp
// Fixes for new-and-improved win32/win16 "console" message pumping.
//
// Revision 1.12  1998/10/11 00:37:50  cyp
// Removed call to SelectCore() [now done from main()] and
// added support for ModeReq
//
// Revision 1.11  1998/10/07 08:10:48  chrisb
// Fixed parameter cast error in call to RegPolledProcedure(yield_pump,...)
// in Client::Run()
//
// Revision 1.10  1998/10/06 21:10:38  cyp
// Removed call to LogSetTimeStampMode(). Function is obsolete.
//
// Revision 1.9  1998/10/05 02:12:12  cyp
// Removed explicit time stamping ([%s],Time()). This is now done automatically
// by Log...(). Added LogSetTimeStampingMode(1); at the top of Client::Run().
//
// Revision 1.8  1998/10/03 03:44:10  cyp
// Removed pthread yield section completely. Changed win16 SurrenderCPU()
// function to a straight win api Yield() call. Fixed a re-RegPollProc()
// that specified a different delay from the initial reg...(). Wrapped
// long comments (yet once more).
//
// Revision 1.7  1998/10/02 17:06:17  chrisb
// removed a #define DEBUG I left in by mistake
//
// Revision 1.6  1998/10/02 16:59:01  chrisb
// lots of fiddling in a vain attempt to get the NON_PREEMPTIVE_OS_PROFILING 
// to be a bit sane under RISC OS
//
// Revision 1.5  1998/09/29 23:36:26  silby
// Commented out call to pthreads_yield since it's not supported in all 
// POSIX implementations (apparently.)
//
// Revision 1.4  1998/09/29 10:13:16  chrisb
// Removed Remi's (CLIENT_OS == OS_NETWARE) stuff around yield_pump. Fixed a 
// comparison bug in the checkpoint retrieval stuff (Client::Run). 
// Miscellaneous RISC OS wibblings.
//
// Revision 1.3  1998/09/28 22:19:17  remi
// Cleared 2 warnings, and noticed that yield_pump() seems to be Netware-only.
// BTW, I've not found pthread_yield() in libpthread 0.7 & glibc2 (RH 5.1 
// Sparc and Debian 2.0 x86), nor in libpthread 0.6 & libc5.
//
// Revision 1.2  1998/09/28 13:29:28  cyp
// Removed checkifbetaexpired() declaration conflict.
//
// Revision 1.1  1998/09/28 03:39:58  cyp
// Spun off from client.cpp
//
//
//
#if (!defined(lint) && defined(__showids__))
const char *clirun_cpp(void) {
return "@(#)$Id: clirun.cpp,v 1.21 1998/10/31 22:36:11 silby Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "version.h"   // CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "problem.h"   // Problem class
#include "network.h"  
#include "mail.h"
#include "scram.h"
#include "convdes.h"  // convert_key_from_des_to_inc 
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()
#include "sleepdef.h" //sleep(), usleep()
#include "setprio.h"  //SetThreadPriority(), SetGlobalPriority()
#include "threadcd.h"
#include "buffwork.h"
#include "clirate.h"
#include "clitime.h"   //CliTimer(), Time()/(CliGetTimeString(NULL,1))
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clisrate.h"
#include "clicdata.h"
#include "pathwork.h"
#include "cpucheck.h"  //GetTimesliceBaseline(), GetNumberOfSupportedProcessors()
#include "probman.h"   //GetProblemPointerFromIndex()
#include "probfill.h"  //LoadSaveProblems(), RandomWork(), FILEENTRY_xxx macros
#include "modereq.h"   //ModeReq[Set|IsSet|Run]()

// --------------------------------------------------------------------------

static int IsFilenameValid( const char *filename )
{ return ( filename && *filename != 0 && strcmp( filename, "none" ) != 0 ); }

static int DoesFileExist( const char *filename )
{
  if ( !IsFilenameValid( filename ) )
    return 0;
  return ( access( GetFullPathForFilename( filename ), 0 ) == 0 );
}

#if defined(BETA)
static int checkifbetaexpired(void)
{
#if defined(BETA_EXPIRATION_TIME) && (BETA_EXPIRATION_TIME != 0)

  timeval currenttime;
  timeval expirationtime;

  expirationtime.tv_sec = BETA_EXPIRATION_TIME;
  expirationtime.tv_usec= 0;

  CliTimer(&currenttime);
  if (currenttime.tv_sec > expirationtime.tv_sec ||
      currenttime.tv_sec < (BETA_EXPIRATION_TIME - 1814400))
    {
    Log("This beta release expired on %s. Please\n"
        "download a newer beta, or run a standard-release client.\n", 
        CliGetTimeString(&expirationtime,1) );
    return 1;
    }
#endif
  return 0;
}
#endif

// ----------------------------------------------------------------------

struct thread_param_block
{
  #if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) 
    unsigned long threadID;
  #elif (CLIENT_OS == OS_NETWARE)
    int threadID;
  #elif (CLIENT_OS == OS_BEOS)
    thread_id threadID;
  #elif (defined(_POSIX_THREADS) || defined(_PTHREAD_H) || \
    defined(_POSIX_THREAD_PRIORITY_SCHEDULING)) && defined(MULTITHREAD)
    pthread_t threadID;
  #else
    int threadID;
  #endif
  unsigned int threadnum;
  unsigned int numthreads;
  int realthread;
  s32 timeslice;
  unsigned int priority;
  int do_suspend;
  int do_refresh;
  int is_suspended;
  unsigned long thread_data1;
  unsigned long thread_data2;
  struct thread_param_block *next;
};

// ----------------------------------------------------------------------

#if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_MACOS) || \
    (CLIENT_OS == OS_RISCOS) || (CLIENT_OS == OS_WIN32S) || \
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
#elif (CLIENT_OS == OS_DOS)
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
  #define MIN_RUNS_PER_TIME_GRAIN     2
  #define MAX_RUNS_PER_TIME_GRAIN     5
  #define INITIAL_TIMESLICE_RC5    512
  #define INITIAL_TIMESLICE_DES    512
  #define MIN_SANE_TIMESLICE_RC5    256
  #define MIN_SANE_TIMESLICE_DES    256
  #define MAX_SANE_TIMESLICE_RC5   1048576
  #define MAX_SANE_TIMESLICE_DES   1048576
//  #error "Please check timer granularity and timeslice constants"
//  #undef NON_PREEMPTIVE_OS_PROFILING  //or undef to do your own profiling
#elif (CLIENT_OS == OS_MACOS)
  #define TIMER_GRANULARITY       125000
  #define MIN_RUNS_PER_TIME_GRAIN     12
  #define MAX_RUNS_PER_TIME_GRAIN     50
  #define INITIAL_TIMESLICE_RC5      128
  #define INITIAL_TIMESLICE_DES      256
  #define MIN_SANE_TIMESLICE_RC5     128
  #define MIN_SANE_TIMESLICE_DES     256
  #define MAX_SANE_TIMESLICE_RC5   16384
  #define MAX_SANE_TIMESLICE_DES   16384
  //#error "Please check timer granularity and timeslice constants"
  #undef NON_PREEMPTIVE_OS_PROFILING  //or undef to do your own profiling
#else
  #error "Unknown OS. Please check timer granularity and timeslice constants"
  #undef NON_PREEMPTIVE_OS_PROFILING  //or undef to do your own profiling
#endif  

#endif /* CLIENT_OS == netware, macos, riscos, win16, win32s */

// ----------------------------------------------------------------------

static struct
{
  int nonmt_ran;
  unsigned long yield_run_count;
} runcounters = {0,0};  

static void yield_pump( void *tv_p )
{
  #if (CLIENT_OS == OS_MACOS)
    EventRecord event;
  #endif
  static int pumps_without_run = 0;
  runcounters.yield_run_count++;

  #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    thr_yield();
  #elif (CLIENT_OS == OS_FREEBSD)
    sched_yield();
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
    sched_yield();
  #elif (CLIENT_OS == OS_MACOS)
    WaitNextEvent( everyEvent, &event, 0, nil );
  #else
    #error where is your yield function?
    NonPolledUSleep( 0 ); /* yield */
  #endif

  // used in conjunction with go_nonmt
  if (tv_p)  
    {
    if (runcounters.nonmt_ran)
      pumps_without_run = 0;
    #ifdef NON_PREEMPTIVE_OS_PROFILING
    else if ((++pumps_without_run) > 5)
      {
      pumps_without_run = 0;
      LogScreen("Yielding too fast. Doubled pump interval.\n"); 
      struct timeval *tv = (struct timeval *)tv_p;
      tv->tv_usec<<=1; tv->tv_sec<<=1;
      if (tv->tv_usec>1000000)
        { tv->tv_sec+=tv->tv_usec/1000000; tv->tv_usec%=1000000; }
      }
    #endif
    if (RegPolledProcedure(yield_pump, tv_p, (struct timeval *)tv_p, 32 )==-1)
      {         //should never happen, but better safe than sorry...
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
      hgrain_run_count = (runcounters.yield_run_count * 10000) / perc; 
#ifdef DEBUG
printf("%d. oldslice = %lu, y_real = %lu/%lu, y_adj (%lu%%) = %lu/%lu ",
          threadnum, tslice_table[contest], runcounters.yield_run_count, usecs,
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
        ts = (totalslice_table[0]/runcounters.yield_run_count)/
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
        ts = (totalslice_table[1]/runcounters.yield_run_count)/
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
        tslice_table[0] += (totalslice_table[0]/runcounters.yield_run_count)/
                                     (over_par-MAX_RUNS_PER_TIME_GRAIN);
#ifdef DEBUG
printf("+%u=> ", tslice_table[0]-ts );
#endif

        if (tslice_table[0] > MAX_SANE_TIMESLICE_RC5)
          tslice_table[0] = MAX_SANE_TIMESLICE_RC5;
        else if ( tslice_table[0] < tslice_lkg[0])
          tslice_table[0] = tslice_lkg[0];

        tslice_table[1] += (totalslice_table[1]/runcounters.yield_run_count)/
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

      ts = (totalslice_table[0]/runcounters.yield_run_count);
      if (ts > tslice_lkg[0])
        {
        tslice_lkg[0] = ts;
        if ( tslice_lkg[0] < MIN_SANE_TIMESLICE_RC5 )
          tslice_lkg[0] =  MIN_SANE_TIMESLICE_RC5;
        }
      tslice_table[0] = tslice_lkg[0] + ((tslice_lkg[0]/10) * goodrun_count);
      ts = (totalslice_table[1]/runcounters.yield_run_count);
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
    totalslice_table[1]=  0;
    tvstop.tv_sec  = tvstart.tv_sec = tvnow.tv_sec;
    tvstop.tv_usec = tvstart.tv_usec = tvnow.tv_usec;
         
    tvstop.tv_usec += TIMER_GRANULARITY;
    if (tvstop.tv_usec > 1000000L)
      {
      tvstop.tv_sec += tvstop.tv_usec/1000000L;
      tvstop.tv_usec %= 1000000L;
      }
    runcounters.yield_run_count = 0; 
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

static void Go_nonmt( void * parm )
{
  unsigned int threadnum, prob_i;
  u32 run;
  runcounters.nonmt_ran = 1;

  if (!CheckExitRequestTriggerNoIO())
    {
    if (CheckPauseRequestTriggerNoIO()) 
      {
      #ifdef NON_PREEMPTIVE_OS_PROFILING
      reset_ts_profiling();
      #endif
      }
    else
      {
      struct thread_param_block *targ = (thread_param_block *)parm;
      threadnum = targ->threadnum;
      prob_i = (unsigned int)((targ->thread_data1) & 1);
      run = 1; /* assume not initialized, ie switch to other problem */

      Problem *thisprob = GetProblemPointerFromIndex((threadnum<<1)+prob_i);

      if (thisprob && thisprob->IsInitialized())
        {
        if ((targ->thread_data1 & 2)==0) /* haven't adjusted timeslice yet */
          {
          targ->thread_data1 |= 2;
          //adjust timeslice here
          }

        #ifdef NON_PREEMPTIVE_OS_PROFILING
        thisprob->tslice = do_ts_profiling( thisprob->tslice, 
                           thisprob->contest, threadnum );
        #endif

        run = thisprob->Run( threadnum );
        }
      if (run != 0) /* flip the order */
        {
        targ->thread_data1 &= ~1L; /* assume only one problem loaded */
        if (targ->numthreads >= 2 && !prob_i) 
          targ->thread_data1 |= 1L;
        targ->thread_data1 &= ~2L; /* timeslice needs to be reset */
        #ifdef NON_PREEMPTIVE_OS_PROFILING
        reset_ts_profiling();
        #endif
        }
      }
    RegPolledProcedure( Go_nonmt, parm, NULL, 0 );
    }
  return;
}

// ----------------------------------------------------------------------


#if defined(MULTITHREAD)
void Go_mt( void * parm )
{
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
  struct thread_param_block FAR *targ = (thread_param_block FAR *)parm;
  #else
  struct thread_param_block *targ = (thread_param_block *)parm;
  #endif

  Problem *thrprob[2];
  unsigned int threadnum = targ->threadnum;
  u32 run;

#if (CLIENT_OS == OS_WIN32)
  {
  DWORD LAffinity, LProcessAffinity, LSystemAffinity;
  OSVERSIONINFO osver;
  s32 numthreads = targ->numthreads;

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
  {
  nwCliInitializeThread( threadnum+1 ); //in netware.cpp
  }
#elif (CLIENT_OS == OS_OS2)
#elif (CLIENT_OS == OS_BEOS)
#else
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

  SetThreadPriority( targ->priority ); /* 0-9 */
  targ->is_suspended = 1;
  targ->do_refresh = 1;

  while (!CheckExitRequestTriggerNoIO())
    {
    for (s32 probnum = 0; probnum < 2 ; probnum++ )
      {
      run = 0;
      while (!CheckExitRequestTriggerNoIO() && (run == 0))
        {
        if (CheckPauseRequestTriggerNoIO() || targ->do_suspend) 
          {
          run = 0;
          yield_pump(NULL); // don't race in this loop
          }
        else if (targ->do_refresh)
          {
          run = 0;
          thrprob[0] = GetProblemPointerFromIndex((threadnum<<1)+0);
          thrprob[1] = GetProblemPointerFromIndex((threadnum<<1)+1);
          targ->do_refresh = 0;
          }
        else if (thrprob[probnum])
          {
          targ->is_suspended = 0;
          // This will return without doing anything if uninitialized...
          #if (CLIENT_OS == OS_NETWARE)
          (thrprob[probnum])->tslice = (GetTimesliceBaseline() >> 1);
          run = (thrprob[probnum])->Run( threadnum ); 
          yield_pump(NULL);
          #else
          run = (thrprob[probnum])->Run( threadnum );
          //#if (CLIENT_OS==OS_WIN32) // change to accomodate new CLI
          //  yield_pump(NULL);
          //  #endif
          #endif
          targ->is_suspended = 1;
          } 
        else
          run = 1;
        }
      }
    yield_pump(NULL);
    }
  
  //the thread is dead
  targ->threadID = 0;
  
  SetThreadPriority( 9 ); /* allow it to exit faster (specially for OS2) */

  #if (CLIENT_OS == OS_BEOS)
  exit(0);
  #endif
}
#endif

// -----------------------------------------------------------------------

static int __StopThread( struct thread_param_block *thrparams )
{
  if (thrparams)
    {
    if (thrparams->threadID) //thread did not exit by itself
      {
      if (thrparams->realthread) //real thread
        {
        yield_pump(NULL);   //give threads some air
        
        #if (CLIENT_OS == OS_OS2)
        DosWaitThread( &(thrparams->threadID), DCWW_WAIT);
        #elif (CLIENT_OS == OS_WIN32)
        WaitForSingleObject((HANDLE)thrparams->threadID, INFINITE);
        CloseHandle((HANDLE)thrparams->threadID);
        #elif (CLIENT_OS == OS_BEOS)
        static status_t be_exit_value;
        wait_for_thread(thrparams->threadID, &be_exit_value);
        #elif (CLIENT_OS == OS_NETWARE)
        nwCliWaitForThreadExit( thrparams->threadID ); //in netware.cpp
        #elif (defined(_POSIX_THREAD_PRIORITY_SCHEDULING) || \
              defined(_POSIX_THREADS) || defined(_PTHREAD_H)) && \
        defined(MULTITHREAD)
        pthread_join( thrparams->threadID, (void **)NULL);
        #endif
        }
      }
    free( thrparams );
    }
  return 0;
}  

// -----------------------------------------------------------------------

static struct thread_param_block *__StartThread( unsigned int thread_i,
                    s32 numthreads, s32 timeslice, unsigned int priority )
{
  int success = 0, use_poll_process = 0;
  
  struct thread_param_block *thrparams = (struct thread_param_block *)
                         malloc( sizeof(struct thread_param_block) );
  if (thrparams)
    {
    // Start the thread for this cpu
    memset( (void *)(thrparams), 0, sizeof( struct thread_param_block ));
    thrparams->threadID = (THREADID)(0);  /* whatever type */
    thrparams->numthreads = numthreads;   /* s32 */
    thrparams->threadnum = thread_i;      /* unsigned int */
    thrparams->realthread = 1;            /* int */
    thrparams->timeslice = timeslice;     /* s32 */
    thrparams->priority = priority;       /* unsigned int */
    thrparams->do_suspend = thrparams->is_suspended = 0;
    thrparams->do_refresh = 1;            
    thrparams->thread_data1 = 0;          /* ulong, free for thread use */
    thrparams->thread_data2 = 0;          /* ulong, free for thread use */
    thrparams->next = NULL;
  
    #if ((CLIENT_CPU != CPU_X86) && (CLIENT_CPU != CPU_88K) && \
       (CLIENT_CPU != CPU_SPARC) && (CLIENT_CPU != CPU_POWERPC))
       //core routines are not thread safe
       #ifdef MULTITHREAD
       #undef MULTITHREAD
       #endif
    #endif

    if (numthreads == 0) /* polled process */
      use_poll_process = 1;
    else
      {
      #if (CLIENT_OS == OS_WIN32) && defined(MULTITHREAD)
        unsigned int thraddr;
        thrparams->threadID = _beginthread( Go_mt, 8192, (void *)thrparams );
        success = ( (thrparams->threadID) != 0);
      #elif (CLIENT_OS == OS_OS2) && defined(MULTITHREAD)
        thrparams->threadID = _beginthread( Go_mt, NULL, 8192, (void *)thrparams );
        success = ( thrparams->threadID != -1);
      #elif (CLIENT_OS == OS_NETWARE) && defined(MULTITHREAD)
        if (!nwCliIsSMPAvailable())
          use_poll_process = 1;
        else 
          success = ((thrparams->threadID = BeginThread( Go_mt, NULL, 8192, 
                                 (void *)thrparams )) != -1);
      #elif (CLIENT_OS == OS_BEOS) && defined(MULTITHREAD)
        char thread_name[32];
        long be_priority;
    
        #error "please check be_prio (priority is now 0-9 [9 is highest/normal])"
        be_priority = ((10*(B_LOW_PRIORITY + B_NORMAL_PRIORITY + 1))/
                                     (9-(thrparams->priority)))/10;
        #ifdef OLDNICENESS
        switch(niceness)
          {
          case 0: be_priority = B_LOW_PRIORITY; break;
          case 1: be_priority = (B_LOW_PRIORITY + B_NORMAL_PRIORITY) / 2; break;
          case 2: be_priority = B_NORMAL_PRIORITY; break;
          default: be_priority = B_LOW_PRIORITY; break;
          }
        #endif
        sprintf(thread_name, "RC5DES crunch#%d", thread_i + 1);
        thrparams->threadID = spawn_thread((long (*)(void *)) Go_mt, 
               thread_name, be_priority, (void *)thrparams );
        if ( ((thrparams->threadID) >= B_NO_ERROR) &&
             (resume_thread(thrparams->threadID) == B_NO_ERROR) )
          success = 1;
      #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING) && defined(MULTITHREAD)
        SetGlobalPriority( thrparams->priority );  
        pthread_attr_t thread_sched;
        pthread_attr_init(&thread_sched);
        pthread_attr_setscope(&thread_sched,PTHREAD_SCOPE_SYSTEM);
        pthread_attr_setinheritsched(&thread_sched,PTHREAD_INHERIT_SCHED);
        if (pthread_create( &(thrparams->threadID), &thread_sched, 
              (void *(*)(void*)) Go_mt, (void *)thrparams ) == 0)
          success = 1;
        SetGlobalPriority( 9 ); //back to normal
      #elif (defined(_POSIX_THREADS) || defined(_PTHREAD_H)) && defined(MULTITHREAD) && (CLIENT_OS != OS_FREEBSD)
        if (pthread_create( &(thrparams->threadID), NULL, 
           (void *(*)(void*)) Go_mt, (void *)thrparams[thread_i] ) == 0 )
          success = 1;
      #else
        use_poll_process = 1;
      #endif
      //everything from this point on shouldn't need MULTITHREAD so ...
      #undef MULTITHREAD 
      }

    if (use_poll_process)
      { 
      thrparams->realthread = 0;            /* int */
      if (timeslice > (1<<12)) 
        thrparams->timeslice = (1<<12);
      thrparams->threadID = RegPolledProcedure(Go_nonmt, 
                              (void *)thrparams , NULL, 0 );
      success = (thrparams[thread_i].threadID != (THREADID)(-1));
      }
        
    if (success)
      {
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
  FileEntry fileentry;
  unsigned int thread_i, cont_i, prob_i;

  Problem *mainprob = NULL; //used in single threaded mode

  struct thread_param_block *thread_data_table = NULL;

  #ifdef OLDNICENESS //fake priority if 'niceness' is used intead of 'priority'
  unsigned int priority = ((niceness==2)?(9):((niceness==1)?(4):(0)));
  #endif

  int TimeToQuit = 0, exitcode = 0, running_threaded = 0;
  unsigned int load_problem_count = 0, planned_problem_count = 0;
  unsigned int getbuff_errs = 0;
  
  time_t timeNow;
  time_t timeRun=0, timeLast=0, timeNextCheckpoint=0, timeNextConnect=0;

  // =======================================
  // Notes:
  //
  // Do not break flow with a return() 
  // [especially not after problems are loaded]
  //
  // Code order:
  //
  // Initialization: (order is important, 'F' symbolizes code that can fail)
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
  // --------------------------------------

  if (UndoCheckpoint()) // we always recover irrespective of TimeToQuit
    {                   // returns !0 if we have a break request
    TimeToQuit = 1;
    exitcode = -1;
    }

  // --------------------------------------
  // BETA check
  // --------------------------------------

  #if defined(BETA)
  if (!TimeToQuit && checkifbetaexpired()!=0) //prints a message
    {
    TimeToQuit = 1;
    exitcode = -1;
    }
  #endif

  // --------------------------------------
  // Determine the number of problems to work with. Number is used everywhere.
  // --------------------------------------

  if (!TimeToQuit)
    {
    if (numcputemp == 0) //user requests non-mt
      {
      load_problem_count = numcputemp = 1;
      #if (CLIENT_OS==OS_WIN32)   // win32 client _has_ to run multithreaded 
         load_problem_count = 2;  // since the main thread _has_ to run at 
      #endif                      // normal priority to keep the window responsive
      }
    #if (CLIENT_OS == OS_NETWARE)
    else if (numcputemp == 1) // NetWare client prefers non-threading  
      load_problem_count = 1; // if only one thread/processor is to used
    #endif
    else
      {
      if (((unsigned int)(numcputemp)) > 
              GetNumberOfSupportedProcessors()) //max by client instance
        numcputemp = (s32)GetNumberOfSupportedProcessors();   //not by platform
      load_problem_count = 2*((unsigned int)(numcputemp));
      }
    }

  // -------------------------------------
  // load (or rather, try to load) that many problems
  // -------------------------------------

  if (!TimeToQuit)
    {
    planned_problem_count = load_problem_count;
    if ( load_problem_count > 1 )
      Log( "Loading two blocks per thread...\n" );
    load_problem_count = LoadSaveProblems( load_problem_count, 0 );
    numcputemp = (load_problem_count + 1) >> 1;
    if (load_problem_count == 0)
      {
      TimeToQuit = 1;
      exitcode = -2;
      }
    }

  // --------------------------------------
  // The contestdone state may have changed, so check it 
  // --------------------------------------

  if (!TimeToQuit && contestdone[0] && contestdone[1])
    {
    Log( "Both contests are marked as closed. This may mean that\n"
         "the contests are over. Check at http://www.distributed.net/\n" );
    TimeToQuit = 1;
    exitcode = -2;
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

  // --------------------------------------
  // Spin up the threads
  // --------------------------------------


  if (!TimeToQuit && load_problem_count > 1 )
    {
    struct thread_param_block *thrparamslast = thread_data_table;
    for ( thread_i = 0; thread_i < (load_problem_count>>1); thread_i++ )
      {
      struct thread_param_block *thrparams = 
                __StartThread( thread_i, numcputemp, timeslice, priority );
      
      if ( thrparams )
        {
        if (!thread_data_table)
          thread_data_table = thrparams;
        else
          thrparamslast->next = thrparams;
        thrparamslast = thrparams;
        }
      else
        {
        Log("Could not start child thread '%c'.\n", thread_i+'A');

        if ( thread_i == 0 ) //was it the first thread that failed?
          load_problem_count = 1; //then switch to non-threaded mode
        else
          load_problem_count = thread_i << 1;
        break;
        }
      }
    if (load_problem_count == 1)
      {
      Log("Switching to single-threaded mode.\n" );
      numcputemp = 1;
      }
    else
      {
      running_threaded = 1;

      numcputemp = load_problem_count>>1;
      if (load_problem_count == 2)
        Log("1 Child thread has been started.\n");
      else if (load_problem_count > 2)
        Log("%d Child threads ('a'%s'%c') have been started.\n",
         load_problem_count>>1, 
            ((load_problem_count>4)?("-"):(" and ")),
            'a'+((load_problem_count>>1)-1));
      }
    } //if ( load_problem_count > 1 )

  
  // -------------------------------
  // resize the problem table if we loaded too much
  // -------------------------------

  if (planned_problem_count != load_problem_count)
    {
    prob_i = load_problem_count;
    if (load_problem_count == 1)
      prob_i++;
    
    for ( ; prob_i < planned_problem_count; prob_i++ )
      {
      Problem *thisprob = GetProblemPointerFromIndex( prob_i );
      if (thisprob && thisprob->IsInitialized())
        {
        cont_i = (unsigned int)thisprob->RetrieveState( 
                                         (ContestWork *) &fileentry, 1 );
        fileentry.contest = (u8)(cont_i);
        fileentry.op      = htonl( OP_DATA );
        fileentry.cpu     = FILEENTRY_CPU;
        fileentry.os      = FILEENTRY_OS;
        fileentry.buildhi = FILEENTRY_BUILDHI; 
        fileentry.buildlo = FILEENTRY_BUILDLO;
        fileentry.checksum =
             htonl( Checksum( (u32 *) &fileentry, (sizeof(FileEntry)/4)-2));
        Scramble( ntohl( fileentry.scramble ),
              (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
        // put it back...
        InternalPutBuffer( in_buffer_file[cont_i], &fileentry );
        }
      }
    }

  // -------------------------------
  // create a yield pump for OSs that need one 
  // -------------------------------
  
  #if defined(NON_PREEMPTIVE_OS) || (CLIENT_OS == OS_WIN32)
  if (!TimeToQuit)
    {
    static struct timeval tv = {0,500};
    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_MACOS)
      tv.tv_usec = 1000;
    #endif
   
    if (RegPolledProcedure(yield_pump, (void *)&tv, (timeval *)&tv, 32 ) == -1)
      {
      Log("Unable to initialize yield pump\n" );
      TimeToQuit = -1; 
      exitcode = -1;
      }
    #if 0
    else
      LogScreen("Yield pump has started... \n" );
    #endif
    }
  #endif

  // -------------------------------
  // create a problem runner for non-preemptive OSs that are not threaded
  // -------------------------------

  #ifdef NON_PREEMPTIVE_OS
  if (!TimeToQuit && !running_threaded /* load_problem_count == 1 */)
    {
    struct thread_param_block *thrparams = __StartThread( 
                  0 /*thread_i*/, 0 /*numthreads*/, timeslice, priority );
    if (thrparams)
      {
      #if 0
      LogScreen("Crunch handler has started...\n" );
      #endif
      thread_data_table = thrparams;
      running_threaded = 1;
      }
    else
      {
      Log("Unable to initialize crunch handler\n" );
      TimeToQuit = -1; 
      exitcode = -1;
      } 
    }
  #endif

  //------------------------------------
  // display the percent bar so the user sees some action
  //------------------------------------

  if (!TimeToQuit && !percentprintingoff)
    LogScreenPercent( load_problem_count ); //logstuff.cpp

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
    if (running_threaded)
      {
      // prevent the main thread from racing & bogging everything down.
      sleep(3);
      }
    else if (CheckPauseRequestTrigger())
      {
      sleep(1);
      }
    else //only one problem and we are not paused
      {
      if (!mainprob)
        mainprob = GetProblemPointerFromIndex(0);
      if (mainprob)
        {
        //Actually run a problem
        mainprob->Run( 0 ); //threadnum
          
        #if (defined(NON_PREEMPTIVE_OS) || (CLIENT_OS == OS_WIN32))
          yield_pump(NULL);
        #endif
        }
      }
    SetGlobalPriority( 9 );

    //------------------------------------
    // Fixup timers
    //------------------------------------

    timeNow = CliTimer(NULL)->tv_sec;
    if (timeLast!=0 && (timeNow < (timeLast+(600))) && (timeNow > timeLast))
      timeRun += timeLast - timeNow; //make sure time is monotonic
    timeLast = timeNow;

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
      Log( "Received %s request...\n",
           (CheckRestartRequestTrigger()?("restart"):("shutdown")) );
      TimeToQuit = 1;
      exitcode = 1;
      }

    //------------------------------------
    //update the status bar, check all problems for change, do reloading etc
    //------------------------------------

    if (!TimeToQuit && !CheckPauseRequestTrigger())
      {
      if (!percentprintingoff)
        LogScreenPercent( load_problem_count ); //logstuff.cpp
      getbuff_errs += LoadSaveProblems(load_problem_count, PROBFILL_GETBUFFERRS);
      }

    //------------------------------------
    // Lurking
    //------------------------------------

    #if defined(LURK)
    if (!TimeToQuit && !ModeReqIsSet(MODEREQ_FETCH|MODEREQ_FLUSH) && 
        dialup.lurkmode && dialup.CheckIfConnectRequested())
      {
      ModeReqSet(MODEREQ_FETCH|MODEREQ_FLUSH);
      }
    #endif

    //------------------------------------
    //handle 'connectoften' requests
    //------------------------------------

    if (!TimeToQuit && connectoften && timeRun > timeNextConnect)
      {
      timeNextConnect = timeRun + 60;
      ModeReqSet(MODEREQ_FETCH|MODEREQ_FLUSH);
      }

    //----------------------------------------
    // Check for time limit...
    //----------------------------------------

    if ( !TimeToQuit && (minutes > 0) && (timeRun > (time_t)( minutes*60 )))
      {
      Log( "Shutdown - %u.%02u hours expired\n", minutes/60, (minutes%60) );
      TimeToQuit = 1;
      exitcode = 3;
      }

    //----------------------------------------
    // Check for 32 consecutive solutions
    //----------------------------------------

    for (int tmpc = 0; tmpc < 2; tmpc++)
      {
      const char *contname = CliGetContestNameFromID( tmpc ); //clicdata.cpp
      if ((consecutivesolutions[tmpc] >= 32) && !contestdone[tmpc])
        {
        contestdone[tmpc] = 1;
        WriteContestandPrefixConfig( );
        if (!TimeToQuit)
          {
          Log( "Too many consecutive %s solutions detected.\n"  
          "Either the contest is over, or this client is pointed at a test port.\n"
          "Marking contest as closed. Further %s blocks will not be processed.\n", 
          contname, contname );
          }
        }
      }
    if (!TimeToQuit && contestdone[0] && contestdone[1])
      {
      TimeToQuit = 1;
      Log( "Both RC5 and DES are marked as closed.\n");
      exitcode = -2;
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
    // Reached the -b limit?
    //----------------------------------------

    // Done enough blocks?
    if (!TimeToQuit && blockcount > 0 && 
         ( totalBlocksDone[0]+totalBlocksDone[1] >= (u32) blockcount ) )
      {
      Log( "Shutdown - %d blocks completed\n", 
                           (u32)totalBlocksDone[0]+totalBlocksDone[1] );
      TimeToQuit = 1;
      exitcode = 4;
      }

    //----------------------------------------
    // If not quitting, then write checkpoints
    //----------------------------------------

    if (!TimeToQuit && !nodiskbuffers && (timeRun > timeNextCheckpoint))
      {
      timeNextCheckpoint = timeRun + (time_t)(checkpoint_min*60);
      if (!CheckPauseRequestTrigger())
        {
        //Checkpoints may be slightly late (a few seconds). However,
        //this eliminates checkpoint catchup due to pausefiles/clock
        //changes/other nasty things that change the clock
        DoCheckpoint(load_problem_count);
        }
      } 
      
    //----------------------------------------
    // If not quitting, then handle mode requests
    //----------------------------------------
    
    if (!TimeToQuit && ModeReqIsSet(-1))
      {
      //Assume that we have "normal priority" at this point and 
      //threads are running at lower priority. If this is not the case, 
      //then benchmarks are going to return wrong results. The messy
      //way around this is to suspend the threads.
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
  // Shutting down: save problem buffers
  // ----------------

   LoadSaveProblems(load_problem_count, PROBFILL_UNLOADALL );

   // ----------------
   // Shutting down: discard checkpoint files, do a net flush if nodiskbuffers
   // ----------------

   for (cont_i = 0; cont_i < 2; cont_i++ )
     {
     if ( DoesFileExist( checkpoint_file[cont_i] ) )
       EraseCheckpointFile( checkpoint_file[cont_i] );
     if (nodiskbuffers)    // we had better flush everything.
       ForceFlush((u8)cont_i);
     }
   if (randomchanged)  
     WriteContestandPrefixConfig();

  
  #if (CLIENT_OS == OS_VMS)
    nice(0);
  #endif

  return exitcode;
}

// ---------------------------------------------------------------------------

int Client::UndoCheckpoint( void )
{
  FileEntry fileentry;
  unsigned int outcont_i, cont_i, recovered;
  int remaining, lastremaining;
  int breakreq = 0;
  u32 optype;

  if (!nodiskbuffers)
    {
    for ( cont_i = 0; cont_i < 2; cont_i++ )
      {
      if ( IsFilenameValid( checkpoint_file[cont_i] ) &&
           IsFilenameValid( in_buffer_file[cont_i] ) && 
           DoesFileExist( checkpoint_file[cont_i] ) )
        {
        recovered = 0;
        lastremaining = -1;
        while ((remaining = (int)InternalGetBuffer( 
          checkpoint_file[cont_i], &fileentry, &optype, cont_i )) != -1)
          {
          Descramble( ntohl( fileentry.scramble ),
                    (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
          outcont_i = (unsigned int)fileentry.contest;
          Scramble( ntohl( fileentry.scramble ),
                    (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
          
          if (((lastremaining!=-1) && (lastremaining!=(remaining + 1))) ||
            ( InternalPutBuffer( in_buffer_file[outcont_i], &fileentry )==-1)
            || ((breakreq = ( CheckExitRequestTrigger() != 0 ))!=0) )
            {
            recovered = 0;
            break;
            }
          recovered++;
          lastremaining = remaining;
          }
        if (recovered)  
          {
          LogScreen("Recovered %u block%s from %s\n", recovered, 
            ((recovered == 1)?(""):("s")), checkpoint_file[cont_i] );
          }
        }
      }
    }
  return ((breakreq)?(-1):(0));
}  

// ---------------------------------------------------------------------------

int Client::DoCheckpoint( unsigned int load_problem_count )
{
  FileEntry fileentry;
  unsigned int cont_i, prob_i;
  int have_valid_filenames = 0;
  
  for (cont_i = 0; cont_i < 2; cont_i++)
    {
    // Remove prior checkpoint information (if any).
    if ( IsFilenameValid( checkpoint_file[cont_i] ) )
      {
      EraseCheckpointFile( checkpoint_file[cont_i] ); 
      have_valid_filenames = 1;
      }
    }

  if ( !nodiskbuffers && have_valid_filenames )
    {
    for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
      {
      Problem *thisprob = GetProblemPointerFromIndex(prob_i);
      if ( thisprob )
        {
        cont_i = (unsigned int)thisprob->RetrieveState(
                                           (ContestWork *) &fileentry, 0);
        if (cont_i == 0 || cont_i == 1)
          {
          if ( IsFilenameValid( checkpoint_file[cont_i] ) )
            {
            fileentry.contest = (u8)cont_i;
            fileentry.op      = htonl( OP_DATA );
            fileentry.cpu     = FILEENTRY_CPU;
            fileentry.os      = FILEENTRY_OS;
            fileentry.buildhi = FILEENTRY_BUILDHI; 
            fileentry.buildlo = FILEENTRY_BUILDLO;
            fileentry.checksum=
               htonl( Checksum( (u32 *) &fileentry, (sizeof(FileEntry)/4)-2));
            Scramble( ntohl( fileentry.scramble ),
                         (u32 *) &fileentry, ( sizeof(FileEntry) / 4 ) - 1 );
            if (InternalPutBuffer( checkpoint_file[cont_i], &fileentry )== -1)
              {
              //Log( "Checkpoint %d, Buffer Error \"%s\"\n", 
              //                     cont_i, checkpoint_file[cont_i] );
              }
            }
          } 
        } 
      }  // for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
    } // if ( !nodiskbuffers )

  return 0;
}

// ---------------------------------------------------------------------------

