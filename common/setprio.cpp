// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/*
   'prio' is a value on the scale of 0 to 9, where 0 is the lowest
   priority and 9 is the highest priority [9 is what the priority would 
   be if priority were not set, ie is 'normal' priority.] 
*/
//
// $Log: setprio.cpp,v $
// Revision 1.49  1999/02/14 05:13:40  cyp
// default prio to lowest on range error.
//
// Revision 1.48  1999/02/13 00:11:21  silby
// Win32 cruncher always runs at lowest prio
//
// Revision 1.47  1999/01/29 18:47:04  jlawson
// fixed formatting.
//
// Revision 1.46  1999/01/17 13:27:42  cyp
// SetPriority() does its own range validation.
//
// Revision 1.45  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.44  1998/12/22 15:58:24  jcmichot
// Fixed QNX prio.
//
// Revision 1.43  1998/12/09 08:33:20  silby
// Freebsd fixes
//
// Revision 1.42  1998/12/04 17:15:51  cyp
// OS/2 change: priority is only set for crunchers. Main thread always runs
// at normal priority.
//
// Revision 1.41  1998/12/04 17:09:30  silby
// Fixed my last change
//
// Revision 1.40  1998/12/04 16:48:11  silby
// Diversifying freebsd prio setting.
//
// Revision 1.39  1998/12/04 11:04:12  cyp
// erp. Fixed a #if defined(_POSIX_THREADS_SUPPORTED) I misplaced.
//
// Revision 1.38  1998/12/01 19:49:14  cyp
// Cleaned up MULT1THREAD #define. See cputypes.h for more info.
//
// Revision 1.37  1998/12/01 15:06:31  cyp
// Changed sucky win32 priorities again. This time with davehart's guidance,
// so hopefully it can stay this way.
//
// Revision 1.36  1998/11/16 20:21:41  foxyloxy
// Irix twiddling. No luck, but I thought I might as well sync up.
//
// Revision 1.35  1998/11/13 15:24:51  silby
// win32 is back to 8,1
//
// Revision 1.34  1998/11/02 04:32:07  cyp
// win32 main-thread priority is adjusted downwards too if running non-threaded.
//
// Revision 1.32  1998/10/29 04:13:19  foxyloxy
//
// Initial IRIX support of new priority handling. Not debugged yet,
// but it won't lock up your system.
//
// Revision 1.31  1998/10/26 03:13:24  cyp
// Changed win32 priority setting so that the main thread always runs at
// normal priority (but in the idle class). Crunch threads are locked at idle.
//
// Revision 1.3  1998/10/20 17:20:17  remi
// Added two missing #ifdef(MULT1THREAD) in __SetPriority()
//
// Revision 1.2  1998/10/11 08:20:34  silby
// win32 is now locked at max idle priority for cracking threads.
//
// Revision 1.1  1998/09/28 01:31:40  cyp
// Created. Note: priority is now on a scale of 0-9 (9 being "normal").
//

#if (!defined(lint) && defined(__showids__))
const char *setprio_cpp(void) {
return "@(#)$Id: setprio.cpp,v 1.49 1999/02/14 05:13:40 cyp Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes

// -----------------------------------------------------------------------

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
    static int useidleclass = -1;
    int threadprio = 0, classprio = 0;
    HANDLE our_thrid = GetCurrentThread();

    #ifdef WIN32GUI
    if (set_for_thread == 1) 
      prio=0;
    #endif
  
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
  
    if (useidleclass == -1) /* not yet detected */
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
  #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
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
      // nothing
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
      nice( (10-(prio+1)) >> 1 ); /* map from 0-9 to 4-0 */
      // assumes base priority is the default 4. 0 is highest priority.
      // GO-VMS.COM can also be used
      }
    }
  #elif (CLIENT_OS == OS_AMIGAOS)
    {
    if ( set_for_thread )
      {
      //nothing - non threaded
      }
    else
      {
      int pri = -(((9-prio) * 10)/5); /* scale from 0-9 to -20 to zero */
      SetTaskPri(FindTask(NULL), pri ); 
      }
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
  #elif (CLIENT_OS == OS_IRIX)
    {
    if ( set_for_thread )
      {
      //nothing - priority is set when created.
      }
    else if (prio == 0)
      {
      schedctl( NDPRI, 0, NDPLOMIN );
      schedctl( RENICE, 0, 39);
      } 
    else if (prio < 9)
      schedctl( NDPRI, 0, (NDPLOMIN - NDPNORMMIN)/prio);
    }
  #else
    {
    #if (CLIENT_OS == OS_FREEBSD)
    #ifndef PRI_OTHER_MAX
    #define PRI_OTHER_MAX 10
    #endif
    #ifndef PRI_OTHER_MIN
    #define PRI_OTHER_MIN 20
    #endif
    #endif
    if ( set_for_thread )
      {
      #if defined(_POSIX_THREADS_SUPPORTED) //defined in cputypes.h
        #if defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
          //nothing - priority is set when created
        #else
          //SCHED_OTHER policy
          int newprio;
          if ( prio == 9 )
            newprio = PRI_OTHER_MAX;
          else
            newprio = (PRI_OTHER_MIN + PRI_OTHER_MAX + 1) / 10;
          if (pthread_setprio(pthread_self(), newprio ) < 0)
            return -1;
        #endif
      #endif
      }
    else 
      {
      static int oldnice = -1;
      int newnice = ((22*(9-prio))+5)/10;  /* scale from 0-9 to 20-0 */
      if (oldnice != -1)
        {
        errno = 0;
        nice( -oldnice );   // note: assumes nice() handles the 
        if ( errno )        // (-20 to 20) range and not 0-40 
          return -1;
        }
      if ( newnice != 0 )
        {
        errno = 0;
        nice( newnice );
        if ( errno )
          return -1;
        }
      oldnice = newnice;
      }
    }
  #endif

  return 0;
}

// -----------------------------------------------------------------------

int SetGlobalPriority(unsigned int prio) 
{
  return __SetPriority( prio, 0 );
}

// -----------------------------------------------------------------------

int SetThreadPriority(unsigned int prio)
{
  return __SetPriority( prio, 1 );
}

// -----------------------------------------------------------------------

