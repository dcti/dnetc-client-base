// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: clirun.cpp,v $
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
return "@(#)$Id: clirun.cpp,v 1.8 1998/10/03 03:44:10 cyp Exp $"; }
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
#define Time() (CliGetTimeString(NULL,1))
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clisrate.h"
#include "clicdata.h"
#include "pathwork.h"
#include "cpucheck.h"  //GetTimesliceBaseline(), GetNumberOfSupportedProcessors()
#include "probman.h"   //GetProblemPointerFromIndex()
#include "probfill.h"  //LoadSaveProblems(), RandomWork(), FILEENTRY_xxx macros

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
    Log("[%s] This beta release expired on %s. Please\n"
        " %s  download a newer beta, or run a standard-release client.\n", 
        CliGetTimeString(&currenttime,1), CliGetTimeString(&expirationtime,1), 
        CliGetTimeString(NULL,0) );
    return 1;
    }
#endif
  return 0;
}
#endif

// ----------------------------------------------------------------------

struct thread_param_block
{
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) 
    unsigned long threadID;
  #elif (CLIENT_OS == OS_NETWARE)
    int threadID;
  #elif (CLIENT_OS == OS_BEOS)
    thread_id threadID;
  #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING) 
    pthread_t threadID;
  #elif defined(_POSIX_THREADS) || defined(_PTHREAD_H)
    pthread_t threadID;
  #else
    int threadID;
  #endif
  unsigned int threadnum;
  unsigned int numthreads;
  int realthread;
  s32 timeslice;
  unsigned int priority;
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
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_DOS)
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
  #define MAX_RUNS_PER_TIME_GRAIN     100
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
  #error "Please check timer granularity and timeslice constants"
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
  static int pumps_without_run = 0;
  runcounters.yield_run_count++;

  #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    thr_yield();
  #elif (CLIENT_OS == OS_OS2)
    DosSleep(0);
  #elif (CLIENT_OS == OS_IRIX)
    sginap(0);
  #elif (CLIENT_OS == OS_WIN32)
    Sleep(0);
  #elif (CLIENT_OS == OS_DOS)
    dosCliYield(); //dpmi yield
  #elif (CLIENT_OS == OS_NETWARE)
    nwCliThreadSwitchLowPriority();
  #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    Yield();
  #elif (CLIENT_OS == OS_RISCOS)
    if (riscos_in_taskwindow)
    { riscos_upcall_6(); }
  #else
    NonPolledUSleep( 0 ); /* yield */
  #endif

  // used in conjunction with go_nonmt
  if (tv_p)  
    {
    if (runcounters.nonmt_ran)
      pumps_without_run = 0;
    else if ((++pumps_without_run) > 5)
      {
      pumps_without_run = 0;
      LogScreen("[%s] Yielding too fast. Doubled pump interval.\n", Time()); 
      struct timeval *tv = (struct timeval *)tv_p;
      tv->tv_usec<<=1; tv->tv_sec<<=1;
      if (tv->tv_usec>1000000)
        { tv->tv_sec=tv->tv_usec/1000000; tv->tv_usec%=1000000; }
      }
    if (RegPolledProcedure(yield_pump, tv_p, NULL, 32 ) == -1)
      {         //should never happen, but better safe than sorry...
      LogScreen("[%s] Panic! Unable to re-initialize yield pump\n", Time()); 
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
        Log("[%s] Running inefficiently. Timer is possibly bad.\n", Time() );
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

  thrprob[0] = GetProblemPointerFromIndex((threadnum<<1)+0);
  thrprob[1] = GetProblemPointerFromIndex((threadnum<<1)+1);

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

  while (!CheckExitRequestTriggerNoIO())
    {
    for (s32 probnum = 0; probnum < 2 ; probnum++ )
      {
      run = 0;
      while (!CheckExitRequestTriggerNoIO() && (run == 0))
        {
        if (CheckPauseRequestTriggerNoIO()) 
          {
          run = 0;
          NonPolledSleep( 1 ); // don't race in this loop
          }
        else if (thrprob[probnum])
          {
          // This will return without doing anything if uninitialized...
          #if (CLIENT_OS == OS_NETWARE)
          (thrprob[probnum])->tslice = (GetTimesliceBaseline() >> 1);
          run = (thrprob[probnum])->Run( threadnum ); 
          yield_pump(NULL);
          #else
          run = (thrprob[probnum])->Run( threadnum ); 
          #endif
          } 
        else
          run = (u32)(-1);
        }
      }
    NonPolledSleep( 1 ); 
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
        #elif (CLIENT_OS == OS_BEOS)
        static status_t be_exit_value;
        wait_for_thread(thrparams->threadID, &be_exit_value);
        #elif (CLIENT_OS == OS_NETWARE)
        nwCliWaitForThreadExit( thrparams->threadID ); //in netware.cpp
        #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING) || \
              defined(_POSIX_THREADS) || defined(_PTHREAD_H)
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
    thrparams->next = NULL;
    thrparams->threadID = (THREADID)(0);  /* whatever type */
    thrparams->threadnum = thread_i;      /* unsigned int */
    thrparams->numthreads = numthreads;   /* s32 */
    thrparams->timeslice = timeslice;     /* s32 */
    thrparams->priority = priority;       /* unsigned int */
    thrparams->realthread = 1;            /* int */
    thrparams->thread_data1 = 0; /* ulong, free for thread use */
    thrparams->thread_data2 = 0; /* ulong, free for thread use */
  
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
        thrparams->threadID = _beginthread( Go_mt, 8192, (void *)thrparams );
        success = ( (thrparams->threadID) != 0);
      #elif (CLIENT_OS == OS_OS2) && defined(MULTITHREAD)
        thrparams->threadID = _beginthread( Go_mt, NULL, 8192, (void *)thrparams );
        success = ( thrparams->threadID != -1);
      #elif (CLIENT_OS == OS_NETWARE) && defined(MULTITHREAD)
        if (!nwCliIsSMPAvailable())
          use_poll_process = 1;
        else
          {
          thrparams->threadID = BeginThread( Go_mt, NULL, 8192, 
                                 (void *)thrparams );
          success = ( thrparams->threadID != -1 );
          }
      #elif (CLIENT_OS == OS_BEOS) && defined(MULTITHREAD)
        char thread_name[32];
        long be_priority;
    
        #error "please check be_prio (priority is now 0-9 [9 is highest/normal])"
        be_priority = ((10*(B_LOW_PRIORITY + B_NORMAL_PRIORITY + 1))/
                                     (9-(thrparams->priority)))/10;
        #if 0
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
      #elif (defined(_POSIX_THREADS) || defined(_PTHREAD_H)) && defined(MULTITHREAD)
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
      yield_pump(NULL);   //let the thread start
    else
      {
      delete thrparams;
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

  //priority is a temporary hack
  unsigned int priority = ((niceness==2)?(9):((niceness==1)?(4):(0)));
  int running_threaded = 0;
  unsigned int load_problem_count = 0;
  unsigned int planned_problem_count = 0;
  s32 nextcheckpointtime = 0;
  s32 TimeToQuit = 0, getbuff_errs = 0;
  s32 exitchecktime, exitcode = 0;
  u32 connectloops = 0;

  #if (CLIENT_OS == OS_WIN32) && defined(NEEDVIRTUALMETHODS)
    connectrequested = 0;         // uses public class member
  #else
    u32 connectrequested = 0;
  #endif

  // =======================================
  // Notes:
  //
  // Do not break flow with a return() 
  // [especially not after problems are loaded]
  //
  // Code order:
  //
  // Initialization: (order is important, 'F' symbolizes code that can fail)
  // 1. F  SelectCore() (because those messages are not timestamped)
  // 2.    CkpointToBufferInput() (it is not affected by TimeToQuit)
  // 3.    Determine number of problems (needs select core)
  // 4. F  Load (or try to load) that many problems (needs number of problems)
  // 5. F  Initialize polling process (needed by threads)
  // 6. F  Spin up threads
  // 7.    Unload over-loaded problems (problems for which we have no worker)
  // 9.    Initialize percent bar (needs final problem table size)
  // 8.    Initialize timers
  //
  // Run... 
  //
  // Deinitialization:
  // 10. Shut down threads
  // 11. Deinitialize polling process
  // 12. Unload problems
  // 13. Throw away checkpoints
  // =======================================

  // --------------------------------------
  // Select an appropriate core and process priority
  // should come before anything else because messages are not timestamped
  // --------------------------------------
  
  if (!TimeToQuit)        
    {
    if (SelectCore())
      {
      TimeToQuit = 1;
      exitcode = -1;
      }
    }

  // --------------------------------------
  // Recover blocks from checkpoint files before anything else
  // --------------------------------------

  // we always recover, irrespective of the TimeToQuit flag
  for ( cont_i = 0; cont_i < 2; cont_i++ )
    {
    // Recover checkpoint info in case we had previously quit abnormally.
    if ( DoesFileExist( checkpoint_file[cont_i] ) )
      {
      s32 recovered = CkpointToBufferInput( (u8)cont_i ); 
      if (recovered != 0) 
        Log( "[%s] Recovered %d block%s from %s checkpoint file\n", Time(),
             recovered, ((recovered==1)?(""):("s")), 
             ((cont_i==0)?("RC5"):("DES")) );
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
    if (numcputemp == 0) //user requests non-mt
      load_problem_count = numcputemp = 1;
    #if (CLIENT_OS == OS_NETWARE)
    else if (numcputemp == 1) //NetWare client prefers non-threading if only one 
      load_problem_count = 1; //thread/processor is to used
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
      Log( "[%s] Loading two blocks per thread...\n", Time() );
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
    Log( "[%s] Both contests are marked as closed. This may mean that\n"
         " %s  the contests are over. Check at http://www.distributed.net/\n",
         Time(), CliGetTimeString(NULL,0) );
    TimeToQuit = 1;
    exitcode = -2;
    }

  // --------------------------------------
  // Initialize the async "process" subsystem
  // --------------------------------------

  if (!TimeToQuit && InitializePolling())
    {
    Log( "[%s] Unable to initialize async subsystem.\n", Time() );
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
        Log("[%s] Could not start child thread '%c'.\n",Time(),thread_i+'A');

        if ( thread_i == 0 ) //was it the first thread that failed?
          load_problem_count = 1; //then switch to non-threaded mode
        else
          load_problem_count = thread_i << 1;
        break;
        }
      }
    if (load_problem_count == 1)
      {
      Log("[%s] Switching to single-threaded mode.\n", Time());
      numcputemp = 1;
      }
    else
      {
      running_threaded = 1;

      numcputemp = load_problem_count>>1;
      if (load_problem_count == 2)
        Log("[%s] 1 Child thread has been started.\n", Time());
      else if (load_problem_count > 2)
        Log("[%s] %d Child threads ('a'%s'%c') have been started.\n",
         Time(), load_problem_count>>1, 
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
  // special for non-preemptive OSs
  // -------------------------------

  #ifdef NON_PREEMPTIVE_OS
    {
    //------------------------------------
    //run inside the polling loop
    //------------------------------------
  
    if (!TimeToQuit)
      {
      // add a yield pump to the polling loop
  
      static struct timeval tv = {0,10000}; /* 100/s */
   
      #if (CLIENT_OS == OS_NETWARE)
      tv.tv_usec = 500;
      #endif
    
      if (RegPolledProcedure(yield_pump, (void *)&tv, NULL, 32 ) == -1)
        {
        Log("[%s] Unable to initialize yield pump\n", Time() );
        TimeToQuit = -1; 
        exitcode = -1;
        }
      else
        {
        LogScreen("[%s] Yield pump has started... \n", Time() );
        }
      }
    
    if (!TimeToQuit && !running_threaded /* load_problem_count == 1 */)
      {
      struct thread_param_block *thrparams = __StartThread( 
                    0 /*thread_i*/, 0 /*numthreads*/, timeslice, priority );
      if (thrparams)
        {
        LogScreen("[%s] Crunch handler has started...\n", Time() );
        thread_data_table = thrparams;
        running_threaded = 1;
        }
      else
        {
        Log("[%s] Unable to initialize crunch handler\n", Time() );
        TimeToQuit = -1; 
        exitcode = -1;
        } 
      }
    }
  #endif

  //------------------------------------
  // display the percent bar so the user sees some action
  //------------------------------------

  if (!TimeToQuit && !percentprintingoff)
    LogScreenPercent( load_problem_count ); //logstuff.cpp

  // --------------------------------------
  // Initialize the timers
  // --------------------------------------

  timeStarted = CliTimer( NULL )->tv_usec;
  exitchecktime = timeStarted + 5;

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
    //Do keyboard stuff for clients that allow user interaction during the run
    //------------------------------------

    #if ((CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)) && !defined(NEEDVIRTUALMETHODS)
      { 
      while ( kbhit() )
        {
        int hitchar = getch();
        if (hitchar == 0) //extended keystroke
          getch();
        else
          {
          if (hitchar == 3 || hitchar == 'X' || hitchar == 'x' || hitchar == '!')
            {
            // exit after current blocks
            if (blockcount > 0)
              blockcount = min(blockcount, (s32) (totalBlocksDone[0] + totalBlocksDone[1] + numcputemp));
            else 
              blockcount = (s32) (totalBlocksDone[0] + totalBlocksDone[1] + numcputemp);
            Log("Exiting after current block\n");
              exitcode = 1;
            }
          if ((load_problem_count > 1) && (hitchar == 'u' || hitchar == 'U'))
            {
            Log("Keyblock Update forced\n");
            connectrequested = 1;
            }
          }
        }
      }
    #endif

    //------------------------------------
    // Lurking
    //------------------------------------

    #if defined(LURK)
    if (!connectrequested && dialup.lurkmode)
      connectrequested=dialup.CheckIfConnectRequested();
    #endif

    //------------------------------------
    //special update request (by keyboard or by lurking) handling
    //------------------------------------

    if (load_problem_count > 1)
      {
      if ((connectoften && ((connectloops++)==19)) || (connectrequested > 0) )
        {
        // Connect every 20*3=60 seconds
        // Non-MT 60 + (time for a client.run())
        connectloops=0;
        if (connectrequested == 1) // forced update by a user
          {
          Update(0 ,1,1,1);  // RC5 We care about the errors, force update.
          Update(1 ,1,1,1);  // DES We care about the errors, force update.
          LogScreen("Keyblock Update completed.\n");
          connectrequested=0;
          }
        else if (connectrequested == 2) // automatic update
          {
          Update(0 ,0,0);  // RC5 We don't care about any of the errors.
          Update(1 ,0,0);  // DES 
          connectrequested=0;
          }
        else if (connectrequested == 3) // forced flush
          {
          Flush(0,NULL,1,1); // Show errors, force flush
          Flush(1,NULL,1,1);
          LogScreen("Flush request completed.\n");
          connectrequested=0;
          }
        else if (connectrequested == 4) // forced fetch
          {
          Fetch(0,NULL,1,1); // Show errors, force fetch
          Fetch(1,NULL,1,1);
          LogScreen("Fetch request completed.\n");
          connectrequested=0;
          };
        }
      }

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
          
        #ifdef NON_PREEMPTIVE_OS
          yield_pump(NULL);
        #endif
        }
      }
    SetGlobalPriority( 9 );

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
      Log( "\n[%s] Received %s request...\n", Time(),
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

    //----------------------------------------
    // Check for time limit...
    //----------------------------------------

    if ( !TimeToQuit && (minutes > 0) && ( ( CliTimer( NULL )->tv_usec ) > 
                            ( time_t )( timeStarted + ( 60 * minutes ) ) ) )
      {
      Log( "\n[%s] Shutdown - %u.%02u hours expired\n", Time(), 
                                                minutes/60, (minutes%60) );
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
          Log( "\n[%s] Too many consecutive %s solutions detected.\n", Time(), contname );
          Log( "[%s] Either the contest is over, or this client is pointed at a test port.\n", Time() );
          Log( "[%s] Marking %s contest as closed.\n", Time(), contname );
          Log( "[%s] Further %s blocks will not be processed.\n", Time(), contname );
          }
        }
      }
    if (!TimeToQuit && contestdone[0] && contestdone[1])
      {
      TimeToQuit = 1;
      Log( "\n[%s] Both RC5 and DES are marked as closed.\n", Time() );
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
      Log( "[%s] Shutdown - %d blocks completed\n", Time(), 
                           (u32)totalBlocksDone[0]+totalBlocksDone[1] );
      TimeToQuit = 1;
      exitcode = 4;
      }

    //----------------------------------------
    // If not quitting, then write checkpoints
    //----------------------------------------

    if (!TimeToQuit && !nodiskbuffers && ((CliTimer(NULL)->tv_sec) > (time_t)nextcheckpointtime ))
      {
      nextcheckpointtime = (CliTimer(NULL)->tv_sec) + (checkpoint_min * 60);
      if (!CheckPauseRequestTrigger())
        {
        //Checkpoints may be slightly late (a few seconds). However,
        //this eliminates checkpoint catchup due to pausefiles/clock
        //changes/other nasty things that change the clock
        DoCheckpoint(load_problem_count);
        }
      } 
    }  // End of MAIN LOOP

  //======================END OF MAIN LOOP =====================

  RaiseExitRequestTrigger(); // will make other threads exit

  // ----------------
  // Shutting down: shut down threads
  // ----------------

  if (thread_data_table)  //we have threads running
    {
    LogScreen("[%s] Waiting for threads to end...\n", Time());
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
   // Shutting down: delete checkpoint files, do a net flush if nodiskbuffers
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

void Client::DoCheckpoint( unsigned int load_problem_count )
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
              //Log( "[%s] Checkpoint %d, Buffer Error \"%s\"\n", Time(), 
              //                          cont_i, checkpoint_file[cont_i] );
              }
            }
          } 
        } 
      }  // for ( prob_i = 0 ; prob_i < load_problem_count ; prob_i++)
    } // if ( !nodiskbuffers )

  return;
}

// ---------------------------------------------------------------------------

s32 Client::SetContestDoneState( Packet * packet)
{
  u32 detect;

  // Set the contestdone state, if possible...
  // Move contestdone[] from 0->1, or 1->0.
  detect = 0;
  if (packet->descontestdone == ntohl(0xBEEFF00DL)) {
    if (contestdone[1]==0) {detect = 2; contestdone[1] = 1;}
  } else {
    if (contestdone[1]==1) {detect = 2; contestdone[1] = 0;}
  }
  if (detect == 2) {
    Log( "Received notification: %s contest %s.\n",
         (detect == 2 ? "DES" : "RC5"),
         (contestdone[(int)detect-1]?"is not currently active":"has started") );
  }

  if (packet->rc564contestdone == ntohl(0xBEEFF00DL)) {
    if (contestdone[0] == 0) {detect = 1; contestdone[0] = 1;}
  } else {
    if (contestdone[0] == 1) {detect = 1; contestdone[0] = 0;}
  }
  if (detect == 1) {
    Log( "Received notification: %s CONTEST %s\n",
        (detect == 2 ? "DES" : "RC5"),
        (contestdone[(int)detect-1]?"IS OVER":"HAS STARTED") );
  }

  if (detect != 0) {
    WriteContestandPrefixConfig();
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------

