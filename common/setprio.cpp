/* Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ------------------------------------------------------------------
 *  'prio' is a value on the scale of 0 to 9, where 0 is the lowest
 *  priority and 9 is the highest priority [9 is what the priority would 
 *  be if priority were not set, ie is 'normal' priority. Unices don't
 *  re-nice() if the prio is 9 (assume external control)] 
 * ------------------------------------------------------------------
*/
const char *setprio_cpp(void) {
return "@(#)$Id: setprio.cpp,v 1.50.2.15 2000/07/13 20:55:10 cyp Exp $"; }

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "sleepdef.h"  // sleep(), usleep()

/* -------------------------------------------------------------------- */

//See public functions at end for explanation
static int __SetPriority( unsigned int prio, int set_for_thread )
{
  if (((int)prio) < 0 || prio > 9) 
    prio = 0;
 
  #if (CLIENT_OS == OS_MACH)
  {
    if ( set_for_thread )
    {
      cthread_t thrid = cthread_self();
      int newprio = __mach_get_max_thread_priority( thrid, NULL );
      if (prio < 9) 
        newprio = (newprio * 9)/10;
      if (newprio == 0)
        newprio++;
      if (cthread_priority( thrid, newprio, 0 ) != KERN_SUCCESS)
        return -1;
    }
  }
  #elif (CLIENT_OS == OS_OS2)
  {
    if ( set_for_thread )
      DosSetPriority( PRTYS_THREAD, PRTYC_IDLETIME, ((32 * prio)/10), 0);
    //main thread always runs at normal priority
  }
  #elif (CLIENT_OS == OS_WIN32)
  {
    static int useidleclass = -1;           // track detection state.
    int threadprio = 0, classprio = 0;
    HANDLE our_thrid = GetCurrentThread();  // Win32 pseudo-handle constant.

    if (set_for_thread && !win32ConIsLiteUI()) // full-GUI crunchers always
      prio = 0;                                // run at idle prio
  
    /* ************************** Article ID: Q106253 *******************
                              process priority class
    THREAD_PRIORITY          Normal, in      Normal, in
                      Idle   Background      Foreground    High    Realtime
    _TIME_CRITICAL     15        15              15         15        31
    _HIGHEST            6         9              11         15        26
    _ABOVE_NORMAL       5         8              10         14        25
    _NORMAL             4         7               9         13        24
    _BELOW_NORMAL       3         6               8         12        23
    _LOWEST             2         5               7         11        22
    _IDLE               1         1               1          1        16
    ******************************************************************* */
  
    /* If we want to run with an idle priority *class*, we need to be able
       to set a *thread* priority of _TIME_CRITICAL for the main and 
       window-handler threads, otherwise I/O will be laggy beyond belief.

       If we cannot set _TIME_CRITICAL, then we have no choice but to use a 
       NORMAL_PRIO_CLASS. Such is win32 scheduling; stupid, stupid, stupid.
    */
    if (useidleclass == -1) /* haven't selected yet */
    {         
      SetPriorityClass( GetCurrentProcess(), IDLE_PRIORITY_CLASS );
      Sleep(1);
      SetThreadPriority( our_thrid, THREAD_PRIORITY_TIME_CRITICAL);    
      if (GetThreadPriority( our_thrid ) == THREAD_PRIORITY_TIME_CRITICAL)
        useidleclass = 1;
      else
      {
        useidleclass = 0;
        SetPriorityClass( GetCurrentProcess(), NORMAL_PRIORITY_CLASS );
        Sleep(1);
      }
      SetThreadPriority( our_thrid, THREAD_PRIORITY_NORMAL );    
    }

    if (useidleclass == 1)
    {
      classprio = IDLE_PRIORITY_CLASS;
      if (!set_for_thread) threadprio = THREAD_PRIORITY_TIME_CRITICAL;/* 15 */
      else if (prio >= 7)  threadprio = THREAD_PRIORITY_HIGHEST;      /*  6 */
      else if (prio >= 5)  threadprio = THREAD_PRIORITY_ABOVE_NORMAL; /*  5 */
      else if (prio >= 4)  threadprio = THREAD_PRIORITY_NORMAL;       /*  4 */
      else if (prio >= 3)  threadprio = THREAD_PRIORITY_BELOW_NORMAL; /*  3 */
      else if (prio >= 2)  threadprio = THREAD_PRIORITY_LOWEST;       /*  2 */
      else /* prio < 2 */  threadprio = THREAD_PRIORITY_IDLE;         /*  1 */
    }
    else /* if (useidleclass == 0) */
    {
      classprio = NORMAL_PRIORITY_CLASS;
      if (!set_for_thread) threadprio = THREAD_PRIORITY_NORMAL;       /*  8 */
      else if (prio >= 7)  threadprio = THREAD_PRIORITY_BELOW_NORMAL; /*  6 */
      else if (prio >= 5)  threadprio = THREAD_PRIORITY_LOWEST;       /*  5 */
      else                 threadprio = THREAD_PRIORITY_IDLE;         /*  1 */
    }
    //SetPriorityClass( GetCurrentProcess(), classprio );
    //Sleep(1);

    SetThreadPriority( our_thrid, threadprio );
  }
  #elif (CLIENT_OS == OS_MACOS)
  {
    if ( set_for_thread )
    {
      // nothing
    }
    else
    {
      // nothing
    }
  }
  #elif (CLIENT_OS == OS_WIN16)
  {
    if ( set_for_thread )
    {
      // nothing
    }
    else
    {
      // nothing
    }
  }
  #elif (CLIENT_OS == OS_NETWARE)
  {
    if ( set_for_thread )
    {
      MPKSetThreadPriority( MPKCurrentThread(), 1 );
    }
    else
    {
      // nothing
    }
  }
  #elif (CLIENT_OS == OS_DOS)
  {
    if ( set_for_thread )
    {
      // nothing
    }
    else
    {
      // nothing
    }
  }
  #elif (CLIENT_OS == OS_BEOS)
  {
    if ( set_for_thread )
    {
      // priority of crunching threads is set when they are created.
    }
    else
    {
      // Main thread runs at normal priority, since it does very little;
    }
  }
  #elif (CLIENT_OS == OS_RISCOS)
  {
    if ( set_for_thread )
    {
      // nothing - non threaded
    }
    else
    {
      // nothing
    }
  }
  #elif (CLIENT_OS == OS_VMS)
  {
    if ( set_for_thread )
    {
      // nothing - non threaded
    }
    else
    {
      nice( (((9-prio) * 10) >> 1)/10 ); /* map from 0-9 to 4-0 */
      // assumes base priority is the default 4. 0 is highest priority.
      // GO-VMS.COM can also be used
    }
  }
  #elif (CLIENT_OS == OS_AMIGAOS)
  {
    #ifdef __PPC__
      if ( set_for_thread )
      {
        SetTaskPri(FindTask(NULL),3);
      }
      #ifdef __POWERUP__
      int pri = -(((133*(9-prio))+5)/10); /* scale from 0-9 to -120 to zero */
      PPCSetTaskAttrsTags(PPCFindTask(NULL),PPCTASKINFOTAG_PRIORITY,pri,TAG_END);
      #else
      struct TaskPPC *task = FindTaskPPC(NULL);
      int newnice = ((22*(9-prio))+5)/10;  /* scale from 0-9 to 20-0 */
      SetNiceValue(task, newnice );
      #endif
    #else
    int pri = -(((133*(9-prio))+5)/10); /* scale from 0-9 to -120 to zero */
    SetTaskPri(FindTask(NULL), pri );
    #endif
  }
  #elif (CLIENT_OS == OS_QNX)
  {
    if ( set_for_thread )
    {
      // nothing - non threaded
    }
    else
    {
      setprio( 0, prio-3 );
    }
  }
  #elif (CLIENT_OS == OS_NTO2)
  {
    if (set_for_thread)
      setprio(0,prio+1);
  }
  #else // all other UNIX-like environments
  {
    if ( set_for_thread )
    {
      #if (CLIENT_OS == OS_FREEBSD) && defined(RTP_PRIO_IDLE)
        /*
        On 14 Nov 99 at 2:06, Remi Guyomarch wrote:
        > options         "P1003_1B"
        > options         "_KPOSIX_PRIORITY_SCHEDULING"
        > options         "_KPOSIX_VERSION=199309L"
        > 
        > When this code is compiled in the kernel, we can use rtprio() 
        > to set the client really idle, unlike the Linux one ;)
        >
        > !@%#!"~@
        > Grr, only root can use rtprio() with rtp.type = RTP_PRIO_IDLE
        */
        /* 
          <quote> (/usr/src/sys/kern/kern_resource.c)
          ... for idle priority, there is a potential for system deadlock 
          if an idleprio process gains a lock on a resource that other 
          processes need (and the idleprio process can't run due to a 
          CPU-bound normal process). Fix me! XXX. 
          </quote>
          Doesn't apply to the client I think.
          I also don't see a dependancy on those options (in 3.4 anyway)
                                                                   -cyp
        */
        /* ah, well. try it anyway. niceness is inherited if it fails. */
        struct rtprio rtp;
        rtp.type = RTP_PRIO_IDLE;
        rtp.prio = RTP_PRIO_MAX - ((RTP_PRIO_MAX * prio) / 9); //0 is highest
        if ( rtprio( RTP_SET, 0, &rtp ) != 0 )
          return -1;
      #elif (!defined(_POSIX_THREADS_SUPPORTED)) //defined in cputypes.h
        /* nothing - native threads, inherit */
      #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
        /* nothing - priority is set when created */
      #else //SCHED_OTHER policy
        #if (CLIENT_OS == OS_FREEBSD)
          #ifndef PRI_OTHER_MAX
            #define PRI_OTHER_MAX 10
          #endif
          #ifndef PRI_OTHER_MIN
            #define PRI_OTHER_MIN 20
          #endif
        #endif
        #if defined(PRI_OTHER_MIN) && defined(PRI_OTHER_MAX)
        if (prio < 9)
        {
          int newprio = ((((PRI_OTHER_MAX - PRI_OTHER_MIN) * prio) / 10) + 
                                                             PRIO_OTHER_MIN);
          if (pthread_setprio(pthread_self(), newprio) < 0)
            return -1;
        }
        #endif    
      #endif
    }
    else if (prio < 9) /* prio=9 means "don't change it" (external control) */
    {
      static int oldnice = 0; /* nothing to do if newnice is also zero */
      int newnice = ((22*(9-prio))+5)/10;  /* scale from 0-9 to 20-0 */
      /* [assumes nice() handles the (-20 to 20) range and not 0-40] */
      if (newnice != oldnice) 
      {             
        if (newnice < oldnice) /* all men are created equal */
          return -1;
        errno = 0;
        nice( newnice-oldnice );
        if ( errno )
          return -1;
        oldnice = newnice;
      }
    }
  }
  #endif

  return 0;
}

/* --------------------------------------------------------------------- */

// called on crunch controller startup. not called by worker threads.
int SetGlobalPriority(unsigned int prio) /* prio level (0-9 inclusive) */
{
  return __SetPriority( prio, 0 ); /* => 0 on success, <0 on error */
}

/* --------------------------------------------------------------------- */

// called by each worker threads at startup, since different scaled 
// priorities might be used for them (ie, they don't inherit from global, 
// or need to override what they inherited)
int SetThreadPriority(unsigned int prio) /* prio level (0-9 inclusive) */
{
  return __SetPriority( prio, 1 ); /* => 0 on success, <0 on error */
}


