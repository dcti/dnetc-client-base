// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/*
   'prio' is a value on the scale of 0 to 9, where 0 is the lowest
   priority and 9 is the highest priority [9 is what the priority would 
   be if priority were not set, ie is 'normal' priority.] 
  
   ************** priority *can* increase (eg when restarting) ***********
*/
//
// $Log: setprio.cpp,v $
// Revision 1.1  1998/09/28 01:31:40  cyp
// Created. Note: priority is now on a scale of 0-9 (9 being "normal").
//
//
//

#if (!defined(lint) && defined(__showids__))
const char *setprio_cpp(void) {
return "@(#)$Id: setprio.cpp,v 1.1 1998/09/28 01:31:40 cyp Exp $"; }
#endif

#include "cputypes.h"  // CLIENT_OS, CLIENT_CPU
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes

// -----------------------------------------------------------------------

static int __SetPriority( unsigned int prio, int set_for_thread )
{
  #if (CLIENT_OS == OS_MACH)
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
  #elif (CLIENT_OS == OS_OS2)
    int prio_scope, prio_level;
    if ( set_for_thread )
      prio_scope = PRTYS_THREAD;
    else 
      prio_scope = 2;

    if (prio == 9)
      DosSetPriority( prio_scope, PRTYC_REGULAR, 0, 0);
    else
      {
      prio_level = (32 * prio)/10;
      DosSetPriority( prio_scope, PRTYC_IDLETIME, prio_level, 0);
      }
  #elif (CLIENT_OS == OS_WIN32)
    static HANDLE main_thrid = 0;
    if (!set_for_thread && !main_thrid)
      main_thrid = GetCurrentThread();
    if (prio >=9)
      {
      SetPriorityClass( GetCurrentProcess(), NORMAL_PRIORITY_CLASS );
      SetThreadPriority( GetCurrentThread(), THREAD_PRIORITY_NORMAL );
      }
    else //between THREAD_PRIORITY_NORMAL (+0) and THREAD_PRIORITY_IDLE (-15)
      { 
      SetPriorityClass( GetCurrentProcess(), IDLE_PRIORITY_CLASS );
      SetThreadPriority( GetCurrentThread(), 
          ((((THREAD_PRIORITY_IDLE + THREAD_PRIORITY_NORMAL)+1)
                                                           *(9-prio))/10) );
      }
    if (set_for_thread && main_thrid) //if we have threads,...
      SetThreadPriority( main_thrid, THREAD_PRIORITY_NORMAL );
  #elif (CLIENT_OS == OS_MACOS)
    if ( set_for_thread )
      {
      // nothing
      }
    else
      {
      // nothing
      }
  #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
    if ( set_for_thread )
      {
      // nothing
      }
    else
      {
      // nothing
      }
  #elif (CLIENT_OS == OS_NETWARE)
    if ( set_for_thread )
      {
      // nothing
      }
    else
      {
      // nothing
      }
  #elif (CLIENT_OS == OS_DOS)
    if ( set_for_thread )
      {
      // nothing
      }
    else
      {
      // nothing
      }
  #elif (CLIENT_OS == OS_BEOS)
    if ( set_for_thread )
      {
      // priority of crunching threads is set when they are created.
      }
    else
      {
      // Main thread runs at normal priority, since it does very little;
      }
  #elif (CLIENT_OS == OS_RISCOS)
    if ( set_for_thread )
      {
      // nothing - non threaded
      }
    else
      {
      // nothing
      }
  #elif (CLIENT_OS == OS_VMS)
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
  #elif (CLIENT_OS == OS_AMIGAOS)
    if ( set_for_thread )
      {
      //nothing - non threaded
      }
    else
      {
      int pri = -(((9-prio) * 10)/5); /* scale from 0-9 to -20 to zero */
      SetTaskPri(FindTask(NULL), pri ); 
      }
  #elif (CLIENT_OS == OS_QNX)
    if ( niceness == 0 )      setprio( 0, getprio(0)-1 );
    else if ( niceness == 1 ) setprio( 0, getprio(0)+1 );
    // else                  /* nothing */;
    #error FIXME: SetPriority needs to be scaled from 0 (lowest/idle prio) to 9 (normal)
  #elif (CLIENT_OS == OS_IRIX)
    if ( set_for_thread )
      {
      //not threaded?
      }
    else
      {
      if (niceness == 0)
        schedctl( NDPRI, 0, 200 );
      //else //nothing
      }
    #error FIXME: SetPriority needs to be scaled from 0 (lowest/idle prio) to 9 (normal)
  #else
    if ( set_for_thread )
      {
      #if defined(_POSIX_THREAD_PRIORITY_SCHEDULING) 
        //nothing - priority is set when created
      #elif (defined(_POSIX_THREADS) || defined(_PTHREAD_H))
        //SCHED_OTHER policy
        int newprio;
        if ( prio == 9 )
          newprio = PRI_OTHER_MAX;
        else
          newprio = (PRI_OTHER_MIN + PRI_OTHER_MAX + 1) / 10;
        if (pthread_setprio(pthread_self(), newprio ) < 0)
          return -1;
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
  #endif

  if (!prio)          //dummy code to suppress
    return (int)prio; //unused variable warnings 
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

