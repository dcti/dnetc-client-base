// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: clirun.cpp,v $
// Revision 1.45  1998/12/01 07:11:50  foxyloxy
// Fixed IRIX MT finally! Woohoo! (Embarrasingly trivial and stupid fix...
// doh)
//
// Revision 1.44  1998/12/01 02:18:27  cyp
// win32 change: __StopThread() boosts a cruncher's priority to its own
// priority level when waiting for the thread to die. This is primarily
// for win9x that appears to allow idle threads to starve.
//
// Revision 1.43  1998/12/01 00:14:47  cyp
// Cleared an unused variable warning.
//
// Revision 1.42  1998/12/01 23:25:31  cyp
// Fixed count bug when one or more threads failed to start, but more than
// one succeeded. blockcount limit is now checked by probfill (which bumps
// the limit if one or more threads is still crunching).
//
// Revision 1.41  1998/11/26 22:17:04  dicamillo
// Restore BeOS priority ranging from 1 to 10.
//
// Revision 1.40  1998/11/26 07:31:03  cyp
// Updated to reflect changed checkpoint and buffwork methods. 
//
// Revision 1.39  1998/11/25 09:23:30  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.38  1998/11/25 06:01:26  dicamillo
// Update for BeOS priorities and NonPolledUSleep for BeOS.
//
// Revision 1.37  1998/11/19 20:39:20  cyp
// Fixed "not doing checkpoints" problem. There is only one checkpoint file
// now.
//
// Revision 1.36  1998/11/12 03:13:10  silby
// Changed freebsd message.
//
// Revision 1.35  1998/11/12 03:08:38  silby
// oops, forgot a );
//
// Revision 1.34  1998/11/12 03:06:52  silby
// Added an int cast that was bothering freebsd, and added a message about freebsd's junky posix implementation.
//
// Revision 1.33  1998/11/10 14:04:08  chrisb
// changed a < to <= so reports ('a'-'z') when there are  26 threads
//
// Revision 1.32  1998/11/10 10:44:06  silby
// Excess yield_pump removed for win32 now that priorities are correct.
//
// Revision 1.31  1998/11/09 20:05:09  cyp
// Did away with client.cktime altogether. Time-to-Checkpoint is calculated
// dynamically based on problem completion state and is now the greater of 1
// minute and time_to_complete_1_percent (an average change of 1% that is).
//
// Revision 1.30  1998/11/09 18:01:25  cyp
// Fixed timeRun adjustment. (checkpoints were always being updated every 3 secs)
//
// Revision 1.29  1998/11/09 01:15:54  remi
// Linux/aout doesn't have sched_yield(), replaced by NonPolledUSleep( 0 );
//
// Revision 1.28  1998/11/06 04:23:16  cyp
// Fixed incorrect thread data * index in pthread startup code. This may
// be the cause for the FreeBSD crashes.
//
// Revision 1.27  1998/11/03 04:23:54  cyp
// Added missing #if ... defined(MULTITHREAD) to def out pthread_sigmask
//
// Revision 1.26  1998/11/03 01:46:51  cyp
// Commit to overwrite corrupted clirun in the tree.
//
// Revision 1.25  1998/11/03 00:42:36  cyp
// Client now runs only one problem per thread. Merged go_nonmt into go_mt.
//
// Revision 1.24  1998/11/02 04:40:18  cyp
// Removed redundant ::numcputemp. ::numcpu does it all.
//
// Revision 1.23  1998/10/31 22:55:10  silby
// freebsd non-mt changes completed, working perfectly.
//
// Revision 1.22  1998/10/31 22:49:40  silby
// Change for freebsd_nonmt.
//
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
// pthread implementations.
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
#if (!defined(lint) && defined(__showids__))
const char *clirun_cpp(void) {
return "@(#)$Id: clirun.cpp,v 1.45 1998/12/01 07:11:50 foxyloxy Exp $"; }
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
#include "checkpt.h"
#include "cpucheck.h"  //GetTimesliceBaseline(), GetNumberOfSupportedProcessors()
#include "probman.h"   //GetProblemPointerFromIndex()
#include "probfill.h"  //LoadSaveProblems(), FILEENTRY_xxx macros
#include "modereq.h"   //ModeReq[Set|IsSet|Run]()
// --------------------------------------------------------------------------

static struct
{
  int nonmt_ran;
  unsigned long yield_run_count;
  volatile int refillneeded;
} runstatics = {0,0,0};  

// --------------------------------------------------------------------------

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

static void yield_pump( void *tv_p )
{
  #if (CLIENT_OS == OS_MACOS)
    EventRecord event;
  #endif
  static int pumps_without_run = 0;
  runstatics.yield_run_count++;

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
    #if defined(__ELF__)
    sched_yield();
    #else // a.out libc4
    NonPolledUSleep( 0 ); /* yield */
    #endif
  #elif (CLIENT_OS == OS_MACOS)
    WaitNextEvent( everyEvent, &event, 0, nil );
  #elif (CLIENT_OS == OS_BEOS)
    NonPolledUSleep( 0 ); /* yield */
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
    totalslice_table[1]=  0;
    tvstop.tv_sec  = tvstart.tv_sec = tvnow.tv_sec;
    tvstop.tv_usec = tvstart.tv_usec = tvnow.tv_usec;
         
    tvstop.tv_usec += TIMER_GRANULARITY;
    if (tvstop.tv_usec > 1000000L)
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

void Go_mt( void * parm )
{
  struct thread_param_block *targ = (thread_param_block *)parm;
  Problem *thisprob = NULL;
  unsigned int threadnum = targ->threadnum;
  u32 run;

#if (CLIENT_OS == OS_RISCOS)
/*if (threadnum == 1)
    {
    thisprob = GetProblemPointerFromIndex(threadnum);
    thisprob->Run( threadnum ); 
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
#elif (CLIENT_OS == OS_OS2)
#elif (CLIENT_OS == OS_BEOS)
#elif ((defined(_POSIX_THREADS) || defined(_PTHREAD_H)) && defined(MULTITHREAD))
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
    if (targ->do_refresh)
      thisprob = GetProblemPointerFromIndex(threadnum);
    run = 1; //assume didn't run
    if (thisprob && thisprob->IsInitialized() && !thisprob->finished && 
       !CheckPauseRequestTriggerNoIO() && !targ->do_suspend)
      {
      #ifdef NON_PREEMPTIVE_OS_PROFILING
      thisprob->tslice = do_ts_profiling( thisprob->tslice, 
                          thisprob->contest, threadnum );
      #endif

      targ->is_suspended = 0;
      // This will return without doing anything if uninitialized...
      run = thisprob->Run( threadnum ); 
      targ->is_suspended = 1;
      
      #if (CLIENT_OS == OS_NETWARE)
      yield_pump(NULL);
      #endif
      }
    if (run != 0)
      {
      if (thisprob == NULL || !thisprob->IsInitialized() || thisprob->finished)
        {
        runstatics.refillneeded = 1;
        yield_pump(NULL);
        }
      else //(CheckPauseRequestTriggerNoIO() || targ->do_suspend)
        {
        //if (targ->realthread)
          NonPolledSleep(1); // don't race in this loop
        }
      #ifdef NON_PREEMPTIVE_OS_PROFILING
      reset_ts_profiling();
      #endif
      targ->do_refresh = 1;
      }
    if (!targ->realthread)
      {
      RegPolledProcedure( Go_mt, parm, NULL, 0 );
      runstatics.nonmt_ran = 1;
      break;
      }
    }
  
  targ->threadID = 0; //the thread is dead
  
  #if (CLIENT_OS == OS_BEOS)
  if (targ->realthread)
    exit(0);
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
        DosWaitThread( &(thrparams->threadID), DCWW_WAIT);
        #elif (CLIENT_OS == OS_WIN32)
        SetThreadPriority( (HANDLE)thrparams->threadID, 
           GetThreadPriority(GetCurrentThread()) );
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
         unsigned int numthreads, s32 timeslice, unsigned int priority, 
         int no_realthreads )
{
  int success = 0, use_poll_process = 0;

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
    thrparams->timeslice = timeslice;     /* s32 */
    thrparams->priority = priority;       /* unsigned int */
#if (CLIENT_OS == OS_RISCOS)
    thrparams->do_suspend = /*thread_i?1:*/0;
#else
    thrparams->do_suspend = 0;
#endif
    thrparams->is_suspended = 0;
    thrparams->do_refresh = 1;            
    thrparams->thread_data1 = 0;          /* ulong, free for thread use */
    thrparams->thread_data2 = 0;          /* ulong, free for thread use */
    thrparams->next = NULL;
  
    use_poll_process = 0;
	no_realthreads = 0;
    if ( no_realthreads )
	use_poll_process = 1;
    else
      {
      #if ((CLIENT_CPU != CPU_X86) && (CLIENT_CPU != CPU_88K) && \
         (CLIENT_CPU != CPU_SPARC) && (CLIENT_CPU != CPU_POWERPC) \
	&& (CLIENT_CPU != CPU_MIPS))
         use_poll_process = 1; //core routines are not thread safe
      #elif (CLIENT_OS == OS_WIN32) 
        unsigned int thraddr;
        thrparams->threadID = _beginthread( Go_mt, 8192, (void *)thrparams );
        success = ( (thrparams->threadID) != 0);
      #elif (CLIENT_OS == OS_OS2)
        thrparams->threadID = _beginthread( Go_mt, NULL, 8192, (void *)thrparams );
        success = ( thrparams->threadID != -1);
      #elif (CLIENT_OS == OS_NETWARE) && defined(MULTITHREAD)
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
      #elif (defined(_POSIX_THREADS) || defined(_PTHREAD_H)) && defined(MULTITHREAD)
	if (pthread_create( &(thrparams->threadID), NULL, 
           (void *(*)(void*)) Go_mt, (void *)thrparams ) == 0 )
          success = 1;
      #else
#error usepoolprocess	
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
      thrparams->threadID = RegPolledProcedure(Go_mt, 
                                (void *)thrparams , NULL, 0 );
      success = ((int)thrparams->threadID != -1);
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
  unsigned int cont_i, prob_i;
  int force_no_realthreads = 0;
  struct thread_param_block *thread_data_table = NULL;

  #ifdef OLDNICENESS //fake priority if 'niceness' is used intead of 'priority'
  unsigned int priority = ((niceness==2)?(9):((niceness==1)?(4):(0)));
  #endif

  int TimeToQuit = 0, exitcode = 0;
  unsigned int load_problem_count = 0;
  unsigned int getbuff_errs = 0;
  
  time_t timeNow;
  time_t timeRun=0, timeLast=0, timeNextConnect=0, timeNextCheckpoint = 0;
  int checkpointsDisabled = (nodiskbuffers != 0);
  unsigned int checkpointsPercent = 0;
  int isPaused=0, wasPaused=0;

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
    force_no_realthreads = 0; /* this is a hint. it does not reflect capability */
    unsigned int numcrunchers = (unsigned int)numcpu;

    #if (CLIENT_OS == OS_FREEBSD)
    if (numcrunchers > 1)
      {
      LogScreen("FreeBSD threads are not SMP aware (do not automatically\n"
                "migrate to distribute processor load). Please run one\n"
                "client per processor.\n");
      numcrunchers = 1;
      }
    #endif
    #if (CLIENT_OS == OS_WIN32)
    if (numcrunchers == 0) // win32 must run with real threads because the
      numcrunchers = 1;    // main thread must run at normal priority
    #endif
    #if (CLIENT_OS == OS_NETWARE)
    if (numcrunchers == 1) // NetWare client prefers non-threading  
      numcrunchers = 0;    // if only one thread/processor is to used
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

    if (load_problem_count == 0)
      {
      Log("Unable to load any blocks. Quitting...\n");
      TimeToQuit = 1;
      exitcode = -2;
      }
    }

  // --------------------------------------
  // The contestdone state may have changed, so check it 
  // --------------------------------------

  if (!TimeToQuit && contestdone[0] && contestdone[1])
    {
    Log( "Both contests are marked as closed. This may mean that the\n"
         "contests are over. Check at http://www.distributed.net/\n" );
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

  // -------------------------------
  // create a yield pump for OSs that need one 
  // -------------------------------

  #if defined(NON_PREEMPTIVE_OS)
  if (!TimeToQuit)
    {
    static struct timeval tv = {0,500};
    #if (CLIENT_OS == OS_MACOS)
      tv.tv_usec = 10000;
    #endif
   
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
    char srange[20];
    unsigned int planned_problem_count = load_problem_count;
    load_problem_count = 0;

    for ( prob_i = 0; prob_i < planned_problem_count; prob_i++ )
      {
      struct thread_param_block *thrparams = 
         __StartThread( prob_i, planned_problem_count, 
                        timeslice, priority, force_no_realthreads );
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
      srange[0]=0;
      if (load_problem_count > 1 && load_problem_count <= 26 /* a-z */)
        sprintf( srange, " ('a'%s'%c')", 
          ((load_problem_count==2)?(" and "):("-")), 
          'a'+(load_problem_count-1) );
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
      int i=0;
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
    if (timeLast!=0 && timeNow > timeLast)
      timeRun += (timeNow - timeLast); //make sure time is monotonic
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
      Log( "Shutdown - reached time limit.\n" );
      TimeToQuit = 1;
      exitcode = 3;
      }

    //----------------------------------------
    // Check for 32 consecutive solutions
    //----------------------------------------

    unsigned int closed_count=0;
    for (cont_i=0; cont_i < CONTEST_COUNT; cont_i++)
      {
      const char *contname = CliGetContestNameFromID( cont_i ); //clicdata.cpp
      if ((consecutivesolutions[cont_i] >= 32) && contestdone[cont_i]==0)
        {
        contestdone[cont_i] = 1;
        if (keyport != 3064)
          randomchanged = 1;
        if (!TimeToQuit)
          {
          Log( "Too many consecutive %s solutions detected.\n"  
          "Either the contest is over, or this client is pointed at a test port.\n"
          "Marking contest as closed. Further %s blocks will not be processed.\n", 
            contname, contname );
          }
        }
      if (contestdone[cont_i])
        closed_count++;
      }
    if (!TimeToQuit && closed_count>=CONTEST_COUNT)
      {
      TimeToQuit = 1;
      Log( "All contests are marked as closed. Quitting...\n");
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
    // If not quitting, then write checkpoints
    //----------------------------------------

    if (!TimeToQuit && !checkpointsDisabled && !CheckPauseRequestTrigger())
      {
      if (timeRun > timeNextCheckpoint)
        {
        unsigned long total_percent_now = 0;
        for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
          {
          Problem *thisprob = GetProblemPointerFromIndex(prob_i);
          if ( thisprob )
            total_percent_now += thisprob->CalcPercent();
          }
        prob_i = checkpointsPercent;
        checkpointsPercent = (total_percent_now/load_problem_count);

        if (checkpointsPercent != prob_i)
          {
          if (CheckpointAction( CHECKPOINT_REFRESH, load_problem_count ))
            checkpointsDisabled = 1;
          timeNextCheckpoint = timeRun + (time_t)(60);
          }
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
  // Shutting down: save prob buffers, flush if nodiskbuffers, kill checkpts
  // ----------------

   LoadSaveProblems(load_problem_count, PROBFILL_UNLOADALL );
   CheckpointAction( CHECKPOINT_CLOSE, 0 ); /* also done by LoadSaveProb */

  
  #if (CLIENT_OS == OS_VMS)
    nice(0);
  #endif

  return exitcode;
}

// ---------------------------------------------------------------------------

