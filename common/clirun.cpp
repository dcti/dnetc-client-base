/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
const char *clirun_cpp(void) {
return "@(#)$Id: clirun.cpp,v 1.101 1999/10/11 17:06:23 cyp Exp $"; }

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
#include "cliident.h"  // CliIsDevelVersion()
#include "clitime.h"   // CliTimer(), Time()/(CliGetTimeString(NULL,1))
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "clicdata.h"  // CliGetContestNameFromID()
#include "checkpt.h"   // CHECKPOINT_[OPEN|CLOSE|REFRESH|_FREQ_[SECS|PERC]DIFF]
#include "cpucheck.h"  // GetTimesliceBaseline(), GetNumberOfSupportedProcessors()
#include "probman.h"   // GetProblemPointerFromIndex()
#include "probfill.h"  // LoadSaveProblems(), FILEENTRY_xxx macros
#include "modereq.h"   // ModeReq[Set|IsSet|Run]()
#include "clievent.h"  // ClientEventSyncPost() and constants

#if (CLIENT_OS == OS_FREEBSD)
#include <sys/mman.h>     /* minherit() */
#include <sys/wait.h>     /* wait() */
#include <sys/resource.h> /* WIF*() macros */
#include <sys/sysctl.h>   /* sysctl()/sysctlbyname() */
//#define USE_THREADCODE_ONLY_WHEN_SMP_KERNEL_FOUND /* otherwise its for >=3.0 */
#define FIRST_THREAD_UNDER_MAIN_CONTROL /* otherwise main is separate */
#endif

// --------------------------------------------------------------------------

static struct
{
  volatile int refillneeded;
} runstatics = {0};

//#define DYN_TIMESLICE_SHOWME 1

struct __dyn_timeslice_struct
{
  unsigned int contest; 
  u32 usec;              /* time */
  u32 max, min, optimal; /* ... timeslice/nodes */
};

static struct __dyn_timeslice_struct 
  default_dyn_timeslice_table[CONTEST_COUNT] = 
{
  {  RC5, 1000000, 0x80000000,  0x00100,  0x10000 },
  {  DES, 1000000, 0x80000000,  0x00100,  0x10000 },
  {  OGR,  200000,   0x100000,  0x00100,  0x10000 },
  {  CSC, 1000000, 0x80000000,  0x00100,  0x10000 }
}; 

// =====================================================================

static int checkifbetaexpired(void)
{
  if (CliIsDevelVersion()) /* cliident.cpp */
  {
    timeval expirationtime;

    #ifndef BETA_PERIOD
    #define BETA_PERIOD (7L*24L*60L*60L) /* one week from build date */
    #endif    /* where "build date" is time of newest module in ./common/ */
    expirationtime.tv_sec = CliGetNewestModuleTime() + (time_t)BETA_PERIOD;
    expirationtime.tv_usec= 0;

    if ((CliTimer(NULL)->tv_sec) > expirationtime.tv_sec)
    {
      Log("This beta release expired on %s. Please\n"
          "download a newer beta, or run a standard-release client.\n",
          CliGetTimeString(&expirationtime,1) );
      return 1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------

struct thread_param_block
{
  #if (defined(_POSIX_THREADS_SUPPORTED)) //cputypes.h
    pthread_t threadID;
  #elif (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32)
    unsigned long threadID;
  #elif (CLIENT_OS == OS_NETWARE)
    int threadID;
  #elif (CLIENT_OS == OS_BEOS)
    thread_id threadID;
  #elif (CLIENT_OS == OS_MACOS)
    MPTaskID threadID;
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
  int is_non_preemptive_os;
  unsigned long thread_data1;
  unsigned long thread_data2;
  struct __dyn_timeslice_struct *dyn_timeslice_table;  
  struct __dyn_timeslice_struct rt_dyn_timeslice_table[CONTEST_COUNT];
  struct thread_param_block *next;
  #if (CLIENT_OS == OS_NETWARE)
  unsigned long thread_restart_time;
  #endif
};

// ----------------------------------------------------------------------

static void __thread_sleep__(int secs)
{
  #if (CLIENT_OS == OS_MACOS)
    mp_sleep(secs);     // Mac needs special sleep call in MP threads
  #else
    NonPolledSleep(secs);
  #endif
}

static void __thread_yield__(void)
{
  #if (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS)
    thr_yield();
  #elif (CLIENT_OS == OS_BSDI)
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
      riscos_upcall_6();
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
  #elif (CLIENT_OS == OS_FREEBSD)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_QNX)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_AIX)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_ULTRIX)
    NonPolledUSleep( 0 ); /* yield */
  #elif (CLIENT_OS == OS_HPUX)
    sched_yield();
  #elif (CLIENT_OS == OS_DEC_UNIX)
   #error if you have "MULTITHREAD" defined something is broken. grep for multithread.
   #if defined(MULTITHREAD)
     sched_yield();
   #else
     NonPolledUSleep(0);
   #endif
  #else
    #error where is your yield function?
    NonPolledUSleep( 0 ); /* yield */
  #endif
}

// ----------------------------------------------------------------------

#if (CLIENT_OS == OS_MACOS)
  OSStatus Go_mt( void * parm )
#else
  void Go_mt( void * parm )
#endif
{
  struct thread_param_block *thrparams = (thread_param_block *)parm;
  Problem *thisprob = NULL;
  unsigned int threadnum = thrparams->threadnum;

#if (CLIENT_OS == OS_RISCOS)
/*if (threadnum == 1)
  {
    thisprob = GetProblemPointerFromIndex(threadnum);
    thisprob->Run();
    return;
  } */
#elif (CLIENT_OS == OS_WIN32)
  if (thrparams->realthread)
  {
    DWORD LAffinity, LProcessAffinity, LSystemAffinity;
    OSVERSIONINFO osver;
    unsigned int numthreads = thrparams->numthreads;

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
  int usepollprocess = 0;
  if (thrparams->realthread)
  {
    thrparams->threadID = GetThreadID(); /* in case we got here first */
    nwCliInitializeThread( threadnum+1 );
    /* rename thread, migrate to MP, bind to cpu... */
    if (nwCliGetPollingAllowedFlag() && threadnum == 0)
    {
      thrparams->thread_restart_time = 0;  
      usepollprocess = 1;
    }
  }
#elif (CLIENT_OS == OS_MACOS)
  if (thrparams->realthread)
  {
    MPEnterCriticalRegion(MP_count_region, kDurationForever);
    MP_active++;
    MPExitCriticalRegion(MP_count_region);
  }
#elif (defined(_POSIX_THREADS_SUPPORTED)) //cputypes.h
  if (thrparams->realthread)
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

  if (thrparams->realthread)
    SetThreadPriority( thrparams->priority ); /* 0-9 */

  thrparams->is_suspended = 1;
  thrparams->do_refresh = 1;

  while (!CheckExitRequestTriggerNoIO())
  {
    int didwork = 0; /* did we work? */
    if (thrparams->do_refresh)
      thisprob = GetProblemPointerFromIndex(threadnum);
    if (thisprob == NULL || thrparams->do_suspend || CheckPauseRequestTriggerNoIO())
    {
//printf("run: isnull? %08x, ispausereq? %d, issusp? %d\n", thisprob, CheckPauseRequestTriggerNoIO(), thrparams->do_suspend);
      if (thisprob == NULL)  // this is a bad condition, and should not happen
        runstatics.refillneeded = 1;// ..., ie more threads than problems
      if (thrparams->realthread)
        __thread_sleep__(1); // don't race in this loop
    }
    else if (!thisprob->IsInitialized())
    {
      runstatics.refillneeded = 1;
      if (thrparams->realthread)
        __thread_yield__();
    }
    else
    {
      int run; u32 optimal_timeslice = 0;
      u32 elapsed_sec, elapsed_usec, runtime_usec;
      unsigned int contest_i = thisprob->contest;
      u32 last_count = thisprob->core_run_count; 
                  
      #if (CLIENT_OS == OS_MACOS)
      {
        thisprob->tslice = GetTimesliceToUse(contest_i);
        optimal_timeslice = 0;
      }
      #else
      {
        #if (!defined(DYN_TIMESLICE)) /* compile time override */
        if (thrparams->is_non_preemptive_os || contest_i == OGR)
        #endif
        {
          if (last_count == 0) /* prob hasn't started yet */
            thisprob->tslice = thrparams->dyn_timeslice_table[contest_i].optimal;
          optimal_timeslice = thisprob->tslice;
        }
      }
      #endif

      elapsed_sec = thisprob->runtime_sec;
      elapsed_usec = thisprob->runtime_usec;

      thrparams->is_suspended = 0;
      #if (CLIENT_OS == OS_NETWARE)
      if (usepollprocess)
        run = nwCliRunProblemAsCallback( thisprob, threadnum+1, thrparams->priority);
      else
      #endif
      run = thisprob->Run();
      thrparams->is_suspended = 1;

      if (thisprob->runtime_usec < elapsed_usec)
      {
        if (thisprob->runtime_sec <= elapsed_sec) /* clock is bad */
          elapsed_sec = (0xfffffffful / 1000000ul) + 1; /* overflow it */
        else
        {
          elapsed_sec = (thisprob->runtime_sec-1) - elapsed_sec;
          elapsed_usec = (thisprob->runtime_usec+1000000ul) - elapsed_usec;
        }
      }
      else
      {
        elapsed_sec = thisprob->runtime_sec - elapsed_sec;
        elapsed_usec = thisprob->runtime_usec - elapsed_usec;
      }
      runtime_usec = 0xfffffffful;
      if (elapsed_sec <= (0xfffffffful / 1000000ul))
        runtime_usec = (elapsed_sec * 1000000ul) + elapsed_usec;

      didwork = (last_count != thisprob->core_run_count);
      if (run != RESULT_WORKING)
      {
        runstatics.refillneeded = 1;
        if (!didwork && thrparams->realthread)
          __thread_yield__();
      }
      
      if (optimal_timeslice != 0)
      {
        if (thrparams->is_non_preemptive_os) /* non-preemptive environment */
        {
          __thread_yield__();
        }
        optimal_timeslice = thisprob->tslice; /* get the number done back */
        #if defined(DYN_TIMESLICE_SHOWME)
        {
          static unsigned int ctr = UINT_MAX;
          static unsigned long totaltime = 0, totalts = 0;
          if (ctr == UINT_MAX)
          {
            totaltime = totalts = 0;
            ctr = 0;
          }
          totaltime += runtime_usec;
          totalts += optimal_timeslice;
          ctr++;
          if (ctr >= 1000 || totaltime > 100000000ul)
          {
            if (ctr)
              LogScreen("ctr: %u avg timeslice: %lu  avg time: %luus\n", ctr, totalts/ctr, totaltime/ctr );
            totaltime = totalts = 0;
            ctr = 0;
          }
        }
        #endif
        if (run == RESULT_WORKING) /* timeslice/time is invalid otherwise */
        {
          unsigned int usec5perc = (thrparams->dyn_timeslice_table[contest_i].usec / 20);
          if (runtime_usec < (thrparams->dyn_timeslice_table[contest_i].usec - usec5perc))
          {
            optimal_timeslice <<= 1;
            if (optimal_timeslice > thrparams->dyn_timeslice_table[contest_i].max)
              optimal_timeslice = thrparams->dyn_timeslice_table[contest_i].max;
          }
          else if (runtime_usec > (thrparams->dyn_timeslice_table[contest_i].usec + usec5perc))
          {
            optimal_timeslice -= (optimal_timeslice>>2);
            //optimal_timeslice >>= 1;
            if (optimal_timeslice < thrparams->dyn_timeslice_table[contest_i].min)
              optimal_timeslice = thrparams->dyn_timeslice_table[contest_i].min;
          }
          thisprob->tslice = optimal_timeslice; /* for the next round */
        }
        else /* ok, we've finished. so save it */
        {  
          u32 opt = thrparams->dyn_timeslice_table[contest_i].optimal;
          if (optimal_timeslice > opt)
            thrparams->dyn_timeslice_table[contest_i].optimal = optimal_timeslice;
          optimal_timeslice = 0; /* reset for the next prob */
        }
      }

      #if (CLIENT_OS == OS_NETWARE)
      /* Try and circumvent NetWare's scheduling optimization for our
         threads. (NetWare sees us doing lots of work, and ends up
         rescheduling us quickly because it thinks we need it). By 
         re-chaining to a child, we not only zero our cruncher's stats,
         we also give MP a chance to better shuffle things about.
      */
      if (!usepollprocess && thrparams->thread_restart_time && 
         thrparams->realthread && didwork && !CheckExitRequestTriggerNoIO())
      {
        unsigned long tnow = GetCurrentTime();
        if (tnow > thrparams->thread_restart_time)
        {
          int thrid;
          thrparams->thread_restart_time = tnow + (10*60*18); /* 10min */
          if ((thrid = nwCliRebootThread(threadnum+1, Go_mt, parm )) != -1)
          {
            thrparams->threadID = thrid;
            return; /* poof! child is doing our work now */
          }
        }
      }  
      #endif
    }
    
    if (!didwork)
    {
      thrparams->do_refresh = 1; /* we need to reload the problem */
    }
    if (!thrparams->realthread)
    {
      RegPolledProcedure( (void (*)(void *))Go_mt, parm, NULL, 0 );
      break;
    }
  }

  thrparams->threadID = 0; //the thread is dead

  #if (CLIENT_OS == OS_BEOS)
  if (thrparams->realthread)
    exit(0);
  #endif
  #if (CLIENT_OS == OS_MACOS)
  if (thrparams->realthread)
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
    __thread_yield__();   //give threads some air
    if (thrparams->threadID) //thread did not exit by itself
    {
      if (thrparams->realthread) //real thread
      {
        #if (defined(_POSIX_THREADS_SUPPORTED)) //cputypes.h
        pthread_join( thrparams->threadID, (void **)NULL);
        #elif (CLIENT_OS == OS_OS2)
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
        while (thrparams->threadID) tick_sleep(60);
        #elif (CLIENT_OS == OS_FREEBSD)
        while (thrparams->threadID) NonPolledUSleep(1000);
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
        unsigned int numthreads, unsigned int priority, int no_realthreads,
        int is_non_preemptive_os )
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
    thrparams->is_non_preemptive_os = is_non_preemptive_os; /* int */
    thrparams->do_suspend = 0;
#if (CLIENT_OS == OS_RISCOS)
    thrparams->do_suspend = /*thread_i?1:*/0;
#endif
    thrparams->is_suspended = 0;
    thrparams->do_refresh = 1;
    thrparams->thread_data1 = 0;          /* ulong, free for thread use */
    thrparams->thread_data2 = 0;          /* ulong, free for thread use */
    thrparams->dyn_timeslice_table = &(thrparams->rt_dyn_timeslice_table[0]);
    memcpy( (void *)(thrparams->dyn_timeslice_table), 
            default_dyn_timeslice_table, sizeof(default_dyn_timeslice_table));
    thrparams->next = NULL;

    use_poll_process = 0;

    if ( no_realthreads )
      use_poll_process = 1;
    else
    {
      #if (!defined(CLIENT_SUPPORTS_SMP)) //defined in cputypes.h
        use_poll_process = 1; //no thread support or cores are not thread safe
      #elif (CLIENT_OS == OS_FREEBSD)
      static int ok2thread = 0; /* <0 = no, >0 = yes, 0 = unknown */
      #ifdef USE_THREADCODE_ONLY_WHEN_SMP_KERNEL_FOUND
      if (ok2thread == 0)
      {
        if (GetNumberOfDetectedProcessors()<2)
          ok2thread = -1;
        else
        { 
          int issmp = 0; size_t len = sizeof(issmp);
          if (sysctlbyname("machdep.smp_active", &issmp, &len, NULL, 0)!=0)
          {
            issmp = 0; len = sizeof(issmp);
            if (sysctlbyname("smp.smp_active", &issmp, &len, NULL, 0)!=0)
              issmp = 0;
          }
          if (issmp)
            issmp = (len == sizeof(issmp));
          if (!issmp)
            ok2thread = -1;
        }
      }
      #endif
      if (ok2thread == 0)
      {
        char buffer[64];size_t len=sizeof(buffer),len2=sizeof(buffer);
        if (sysctlbyname("kern.ostype",buffer,&len,NULL,0)!=0) 
          ok2thread = -2;
        else if (len<7 || memcmp(buffer,"FreeBSD",7)!=0)      
          ok2thread = -3;
        else if (sysctlbyname("kern.osrelease",buffer,&len2,NULL,0)!=0)
          ok2thread = -4;
        else if (len<3 || !isdigit(buffer[0]) || buffer[1]!='.' || !isdigit(buffer[2]))
          ok2thread = -5;
        else
        {
          ok2thread = ((buffer[0]-'0')*100)+((buffer[2]-'0')*10);
          //printf("found fbsd ver %d\n", ok2thread );
          if (ok2thread < 300) /* FreeBSD < 3.0 */
            ok2thread = -6;
        }
      }
      if (ok2thread<1) /* not FreeBSD or not >=3.0 (or non-SMP kernel) */
      { 
        use_poll_process = 1; /* can't use this stuff */
      }
      else 
      { 
        static int assertedvms = 0;
        if (thrparams->threadnum == 0) /* the first to spin up */
        { 
          #define ONLY_NEEDED_VMPAGES_INHERITABLE         //children faster
          //#define ALL_DATA_VMPAGES_INHERITABLE          //parent faster
          //#define ALL_TEXT_AND_DATA_VMPAGES_INHERITABLE //like linux threads

          assertedvms = -1; //assume failed
          #if defined(ONLY_NEEDED_VMPAGES_INHERITABLE)
          {
            //children are faster, main is slower
            extern int TBF_MakeTriggersVMInheritable(void); /* probman.cpp */
            extern int TBF_MakeProblemsVMInheritable(void); /* triggers.cpp */
            if (TBF_MakeTriggersVMInheritable()!=0 ||
                TBF_MakeProblemsVMInheritable()!=0 ||
                minherit((void *)&runstatics, sizeof(runstatics), 0)==0)
              assertedvms = +1; //success
          }
          #else
          {
            extern _start; //iffy since crt0 is at the top only by default
            extern etext;
            extern edata;
            extern end;
            //printf(".text==(start=0x%p - etext=0x%p) .rodata+.data==(etext - edata=0x%p) "
            //  ".bss==(edata - end=0x%p) heap==(end - sbrk(0)=0x%p)\n", 
            //  &_start, &etext,&edata,&end,sbrk(0));
            #if defined(ALL_TEXT_AND_DATA_VMPAGES_INHERITABLE)
            //.text+.rodata+.data+.bss+heap (so far)
            if (minherit((void *)&_start,(sbrk(0)-((char *)&_start)),0)==0)
              assertedvms = +1; //success
            #else
            //main is faster, children are slower
            //.rodata+.data+.bss+heap (so far)
            if (minherit((void *)&etext,(sbrk(0)-((char *)&etext)),0)==0)
              assertedvms = +1; //success
            #endif
          }
          #endif

          #ifdef FIRST_THREAD_UNDER_MAIN_CONTROL
          use_poll_process = 1; /* the first thread is always non-real */ 
          #endif
        }
        if (assertedvms != +1)
          use_poll_process = 1; /* can't use this stuff */ 
      }
      if (use_poll_process == 0)
      {
        thrparams->threadID = 0;
        success = 0;
        if (minherit((void *)thrparams, sizeof(struct thread_param_block), 
                                  0 /* undocumented VM_INHERIT_SHARE*/)==0)
        {
          int rforkflags=RFPROC|RFTHREAD|RFSIGSHARE|RFNOWAIT/*|RFMEM*/;
          pid_t new_threadID = rfork(rforkflags);
          if (new_threadID == -1) /* failed */
            success = 0;
          else if (new_threadID == 0) /* child */
          { 
            int newprio=((22*(9-thrparams->priority))+5)/10;/*scale 0-9 to 20-0*/
            thrparams->threadID = getpid(); /* may have gotten here first */
            setpriority(PRIO_PROCESS,thrparams->threadID,newprio);
            //nice(newprio); 
            if ((rforkflags & RFNOWAIT) != 0) /* running under init (pid 1) */
             { seteuid(65534); setegid(65534); } /* so become nobody */
            Go_mt( (void *)thrparams );
            _exit(0);
          }  
          else if ((rforkflags & RFNOWAIT) != 0) 
          {    /* thread is detached (a child of init), so we can't wait() */
            int count = 0;
            while (count<100 && thrparams->threadID==0) /* set by child */
              NonPolledUSleep(1000);
            if (thrparams->threadID) /* child started ok */
              success = 1;
            else
              kill(new_threadID, SIGKILL); /* its hung. kill it */
          }
          else /* "normal" parent, so we can wait() for spinup */
          {  
            int status, res;
            NonPolledUSleep(3000); /* wait for fork1() */
            res = waitpid(new_threadID,&status,WNOHANG|WUNTRACED);
            success = 0;
            if (res == 0) 
            {
              //printf("waitpid() returns 0\n");          
              thrparams->threadID = new_threadID; /* may have gotten here first */
              success = 1;
            }
            #if 0 /* debug stuff */
            else if (res == -1)
              printf("waitpid() returns -1, err=%s\n", strerror(errno));
            else if (res != new_threadID)
              printf("strange. waitpid() returns something wierd\n");
            else if (WIFEXITED(status))
              printf("child %d called exit(%d) or _exit(%d)\n",new_threadID,
                WEXITSTATUS(status), WEXITSTATUS(status));
            else if (WIFSIGNALED(status))
            {
              int sig = WTERMSIG(status);
              printf("child %d caught %s signal %d\n",new_threadID,
                     (sig < NSIG ? sys_signame[sig] : "unknown"), sig );
            }
            else if (WIFSTOPPED(status))
              printf("child %d stopped on %d signal\n",new_threadID,WSTOPSIG(status));
            else
              printf("unkown cause for stop\n");
            #endif
            if (!success)
              kill(new_threadID, SIGKILL); /* its hung. kill it */
          } /* if parent or child */
        } /* thrparam inheritance ok */
      } /* FreeBSD >= 3.0 (+ SMP kernel optional) + global inheritance ok */
      #elif (CLIENT_OS == OS_WIN32)
        if (winGetVersion() < 400) /* win32s */
          use_poll_process = 1;
        else
        {
          thrparams->threadID = _beginthread( Go_mt, 8192, (void *)thrparams );
          success = ( (thrparams->threadID) != 0);
        }
      #elif (CLIENT_OS == OS_OS2)
        thrparams->threadID = _beginthread( Go_mt, NULL, 8192, (void *)thrparams );
        success = ( thrparams->threadID != -1);
      #elif (CLIENT_OS == OS_NETWARE)
      thrparams->thread_restart_time = 0;
      if (nwCliAreCrunchersRestartable())
        thrparams->thread_restart_time = GetCurrentTime()+(10*60*18);
      success = ((thrparams->threadID = nwCliCreateThread( 
                                        Go_mt, (void *)thrparams )) != -1);
      #elif (CLIENT_OS == OS_BEOS)
        char thread_name[128];
        long be_priority = thrparams->priority+1;
        // Be OS priority should be adjustable from 1 to 10
        // 1 is lowest, 10 is higest for non-realtime and non-system tasks
        sprintf(thread_name, "%s crunch#%d", utilGetAppName(), thread_i + 1);
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
      thrparams->dyn_timeslice_table = &(default_dyn_timeslice_table[0]);
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
      __thread_yield__();   //let the thread start
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
  int is_non_preemptive_os = 0;
  struct thread_param_block *thread_data_table = NULL;

  int TimeToQuit = 0, exitcode = 0;
  int probmanIsInit = 0;
  int local_connectoften = 0;
  unsigned int load_problem_count = 0;
  unsigned int getbuff_errs = 0;

  time_t timeNow,timeRun=0,timeLast=0; /* Last is also our "firstloop" flag */
  time_t timeNextConnect=0, timeNextCheckpoint = 0;

  time_t last_scheduledupdatetime = 0; /* we reset the next two vars on != */
  //unsigned int flush_scheduled_count = 0; /* used for exponential staging */
  unsigned int flush_scheduled_adj   = 0; /*time adjustment to next schedule*/
  time_t ignore_scheduledupdatetime_until = 0; /* ignore schedupdtime until */

  int checkpointsDisabled = (nodiskbuffers != 0);
  unsigned int checkpointsPercent = 0;
  int dontSleep=0, isPaused=0, wasPaused=0;
  
  numcpu = ValidateProcessorCount( numcpu, 0 /* notquietly */ ); //cpucheck.cpp
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
  // 3. F  Create problem table (InitializeProblemManager())
  // 4. F  Load (or try to load) that many problems (needs number of problems)
  // 5. F  Initialize polling process (needed by threads)
  // 6. F  Spin up threads
  // 7.    Unload over-loaded problems (problems for which we have no worker)
  //
  // Run...
  //
  // Deinitialization:
  // 8.    Shut down threads
  // 9.    Deinitialize polling process
  // 10.   Unload problems
  // 11.   Deinitialize problem table
  // 12.   Throw away checkpoints
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

  if (!TimeToQuit && checkifbetaexpired()) //prints a message
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

    #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_OS2) || (CLIENT_OS==OS_BEOS)
    if (numcrunchers == 0) // must run with real threads because the
      numcrunchers = 1;    // main thread runs at normal priority
    #endif
    #if (CLIENT_OS == OS_NETWARE)
    //if (numcrunchers == 1) // NetWare client prefers non-threading
    //  numcrunchers = 0;    // if only one thread/processor is to used
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
  // build a problem table
  // -------------------------------------

  probmanIsInit = InitializeProblemManager(load_problem_count);
  if (probmanIsInit > 0)
    load_problem_count = (unsigned int)probmanIsInit;
  else
  {
    Log( "Unable to initialize problem manager. Quitting...\n" );
    probmanIsInit = 0;
    load_problem_count = 0;
    TimeToQuit = 1;
  }

  // -------------------------------------
  // load (or rather, try to load) that many problems
  // -------------------------------------

  if (!TimeToQuit)
  {
    if (load_problem_count > 1)
      Log( "Loading crunchers with work...\n" );
    load_problem_count = LoadSaveProblems( this, load_problem_count, 0 );

    if (CheckExitRequestTrigger())
    {
      TimeToQuit = 1;
      exitcode = -2;
    } 
    else if (load_problem_count == 0)
    {
      Log("Unable to load any work. Quitting...\n");
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

  // --------------------------------------
  // fixup the dyn_timeslice_table if running a non-preemptive OS 
  // --------------------------------------

  is_non_preemptive_os = 0;  /* assume this until we know better */
  #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S) || \
      (CLIENT_OS == OS_RISCOS) || (CLIENT_OS == OS_NETWARE) || \
      (CLIENT_OS == OS_WIN32) /* need to check for win32s */
  is_non_preemptive_os = 1; /* assume this until we know better */
  #if (CLIENT_OS == OS_WIN32)                /* only if win32s */
  if (winGetVersion()>=400)
    is_non_preemptive_os = 0;
  #elif (CLIENT_OS == OS_NETWARE)            
  if (nwCliIsPreemptiveEnv())            /* settable on NetWare 5 */
    is_non_preemptive_os = 0;
  #endif
  if (!TimeToQuit && is_non_preemptive_os)
  {      
    int tsinitd;
    for (tsinitd=0;tsinitd<CONTEST_COUNT;tsinitd++)
    {
      #if (CLIENT_OS == OS_RISCOS)
      if (riscos_in_taskwindow)
      {
        default_dyn_timeslice_table[tsinitd].usec = 30000;
        default_dyn_timeslice_table[tsinitd].optimal = 32768;
      }
      else
      {
        default_dyn_timeslice_table[tsinitd].usec = 1000000;
        default_dyn_timeslice_table[tsinitd].optimal = 131072;
      }
      #elif (CLIENT_OS == OS_NETWARE) 
      /* The switchcount<->runtime ratio is inversely proportionate. 
         By definition, 1000ms == 1.0 switchcounts/sec. In real life it
         looks something like this:  (note the middle magic of "55".
         55ms == one timer tick)
         msecs:   880  440  220  110  55  27.5   14  6.8  3.4   1.7  
         count: ~2.75 ~5.5  ~11  ~22 ~55  ~110 ~220 ~440 ~880 ~1760
         For simplicity, we use 30ms (~half-a-tick) as max for prio 9
         and 3ms as min (about the finest monotonic res we can squeeze 
         from the timer on a 386)
      */
      default_dyn_timeslice_table[tsinitd].optimal = GetTimesliceBaseline();
      default_dyn_timeslice_table[tsinitd].usec = 500 * (priority+1);
      #else /* x86 */
      default_dyn_timeslice_table[tsinitd].usec = 27500; /* 55/2 ms */
      default_dyn_timeslice_table[tsinitd].optimal = GetTimesliceBaseline();
      #endif
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
                        priority, force_no_realthreads,
                        is_non_preemptive_os );
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
        LoadSaveProblems(this, planned_problem_count,PROBFILL_UNLOADALL);
      else
        LoadSaveProblems(this, load_problem_count, PROBFILL_RESIZETABLE);
    }
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

  dontSleep = 1; // don't sleep in the first loop 
                 // (do percbar, connectoften, checkpt first)

  // Start of MAIN LOOP
  while (TimeToQuit == 0)
  {
    //------------------------------------
    //sleep, run or pause...
    //------------------------------------

    if (dontSleep)
      dontSleep = 0; //for the next round
    else
    {             
      SetGlobalPriority( priority );
      if (isPaused)
        NonPolledSleep(3); //sleep(3);
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
    }

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

    if ( !TimeToQuit && (minutes > 0) && (timeRun >= (time_t)( minutes*60 )))
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
    
    #if (CLIENT_OS == OS_DOS)   
    { // DOS4G signal handling is broken: ^C doesn't get through if the keyb 
      // buffer isn't empty. But we can't do this in triggers.cpp because
      // we may need normal keyb handling for config.
      if (kbhit())
      {
        int c = getch();
        if (c == 0)
          c |= getch() << 8;
        if (c == 0x03)
          RaiseExitRequestTrigger();
      }
    }
    #endif
    if (!TimeToQuit && CheckExitRequestTrigger())
    {
      Log( "%s...\n",
         (CheckRestartRequestTrigger()?("Restarting"):("Shutting down")) );
      TimeToQuit = 1;
      exitcode = 1;
    }
    if (!TimeToQuit && checkifbetaexpired()) /* prints a message */
    {
      TimeToQuit = 1;
      exitcode = -1;
    }
    if (!TimeToQuit)
    {
      isPaused = CheckPauseRequestTrigger();
      if (isPaused)
      {
        if (!wasPaused)
          Log("Paused...\n");
        wasPaused = 1;
      }
      else if (wasPaused)
      {
        Log("Running again after pause...\n");
        wasPaused = 0;
      }
    }

    //------------------------------------
    //update the status bar, check all problems for change, do reloading etc
    //------------------------------------

    if (!TimeToQuit && !isPaused)
    {
      int anychanged;
      if (!percentprintingoff)
        LogScreenPercent( load_problem_count ); //logstuff.cpp
      anychanged = LoadSaveProblems(this,load_problem_count,PROBFILL_ANYCHANGED);
      runstatics.refillneeded = 0;

      if (CheckExitRequestTriggerNoIO())
        continue;
      else if (anychanged)      /* load/save action occurred */
        timeNextCheckpoint = 0; /* re-checkpoint right away */
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
            int rc = BufferUpdate( this, BUFFERUPDATE_FETCH|BUFFERUPDATE_FLUSH, 0 );
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
      if (timeRun >= timeNextCheckpoint)
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

        if ( ( timeNextCheckpoint == 0 ) || ( CHECKPOINT_FREQ_PERCDIFF < 
          abs((int)(checkpointsPercent - ((unsigned int)perc_now))) ) )
        {
          checkpointsPercent = (unsigned int)perc_now;
          if (CheckpointAction( CHECKPOINT_REFRESH, load_problem_count ))
            checkpointsDisabled = 1;
          timeNextCheckpoint = timeRun + (time_t)(CHECKPOINT_FREQ_SECSDIFF);
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

    if (!TimeToQuit && local_connectoften && timeRun >= timeNextConnect)
    {
      int doupd = 1;
      if (timeNextConnect != 0)
      {
        int i, have_non_empty = 0, have_one_full = 0;
        for (i = 0; i < CONTEST_COUNT; i++ )
        {
          unsigned cont_i = (unsigned int)loadorder_map[i];
          if (cont_i < CONTEST_COUNT) /* not disabled */
          {
            if (GetBufferCount( cont_i, 1, NULL ) > 0) 
            {
              have_non_empty = 1; /* at least one out-buffer is not empty */
              break;
            }
            if (GetBufferCount( cont_i, 0, NULL ) >= 
               ((long)(inthreshold[cont_i]))) 
            {         
              have_one_full = 1; /* at least one in-buffer is full */
            }
          }
        }
        doupd = (have_non_empty || !have_one_full);
      }
      if ( doupd )
      {
        ModeReqSet(MODEREQ_FETCH|MODEREQ_FLUSH|MODEREQ_FQUIET);
      }
      timeNextConnect = timeRun + 30;
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
      dontSleep = 1; //go quickly through the loop
    }
  }  // End of MAIN LOOP

  //======================END OF MAIN LOOP =====================

  RaiseExitRequestTrigger(); // will make other threads exit

  // ----------------
  // Shutting down: shut down threads
  // ----------------

  if (thread_data_table)  //we have threads running
  {
    LogScreen("Waiting for crunchers to stop...\n");
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

  if (probmanIsInit)
  {
    LoadSaveProblems( this,load_problem_count, PROBFILL_UNLOADALL );
    CheckpointAction( CHECKPOINT_CLOSE, 0 ); /* also done by LoadSaveProb */
    DeinitializeProblemManager();
  }

  ClientEventSyncPost( CLIEVENT_CLIENT_RUNFINISHED, 0 );

  #if (CLIENT_OS == OS_VMS)
    nice(0);
  #endif

  return exitcode;
}

// ---------------------------------------------------------------------------


