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
// Added two missing #ifdef(MULTITHREAD) in __SetPriority()
//
// Revision 1.2  1998/10/11 08:20:34  silby
// win32 is now locked at max idle priority for cracking threads.
//
// Revision 1.1  1998/09/28 01:31:40  cyp
// Created. Note: priority is now on a scale of 0-9 (9 being "normal").
//

#if (!defined(lint) && defined(__showids__))
const char *setprio_cpp(void) {
return "@(#)$Id: setprio.cpp,v 1.35 1998/11/13 15:24:51 silby Exp $"; }
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
    /*
       win32's implementation of thread priorities is totally screwy. It
       is technically easier to implement different priorities for different 
       threads based off of normal or high priority than to do so based off 
       of idle priority. M$ is apparently aware of the problem: (a) Win-CE
       does away with process priorities altogether (b) The screen saver
       runs with a class priority of idle but a thread priority of normal
       giving an effective priority of low priority, but by no means idle.
       To implement a class priority of idle, a crunch thread priority of 
       idle and a main thread priority of normal (to keep it responsive) 
       would require setting the thread priority of the main thread to 
       real-time priority. yuk! (This is also not an option on NT without
       admin rights).
       However, it is possible to set a class priority of normal, a crunch 
       priority of idle and a main thread priority of idle.
     definitions: 
       - The "base_priority", of a thread is computed off of the priority 
       level of the process (ie the priority class) which owns it. The 
       *priority class* can only be one of 0x20(normal),0x40(idle),0x80(high),
       or 0x100(real-time). The resulting base_priority will then be either 4 
       (idle), 8 (normal), 12 (high) or 14 (real-time). 
       Changing the priority class causes the kernel to walk the thread
       info structures of all threads belonging to that process and
       change the base_priority field individually for that thread.
       - "thread_priority" is a delta which is added to the "base_priority"
       and can be either +15 or in the range +2 to -15.
       -  The "effective_priority" of a thread, ie the priority a thread has
       when it is scheduled to run is computed from the sum of its 
       "base_priority" and its "thread_priority", and truncated such that it
       lies in the range 1 to 31.
     */
    static HANDLE main_thrid = 0;
    HANDLE our_thrid = GetCurrentThread();
    if (!set_for_thread && !main_thrid)
      main_thrid = our_thrid;

    int newprio=THREAD_PRIORITY_LOWEST;
//    if (prio >= 9)      newprio = THREAD_PRIORITY_NORMAL; /* +0 */
    if (prio >= 5) newprio = THREAD_PRIORITY_BELOW_NORMAL; /* -1 */
    else if (prio >= 1) newprio = THREAD_PRIORITY_LOWEST; /* -2 */
    else if (prio == 0) newprio = THREAD_PRIORITY_IDLE;  /* -15 */
    /* At thread_prio_idle, the crunch stops when a screen saver is active.
       There is no priority level between -15 and -2! */
    SetPriorityClass(GetCurrentProcess(),NORMAL_PRIORITY_CLASS);
    //setting priority class has no effect here since it is changed below
    SetThreadPriority( our_thrid, newprio );
    #if 0 /* old locked method */
    SetThreadPriority( our_thrid, ((!set_for_thread || prio >=9)?
      (THREAD_PRIORITY_NORMAL /* +0 */):( THREAD_PRIORITY_IDLE /* -15 */))); 
    #endif

    if (set_for_thread && main_thrid)
      {
      SetPriorityClass( GetCurrentProcess(), NORMAL_PRIORITY_CLASS );
      SetThreadPriority( main_thrid, THREAD_PRIORITY_NORMAL );
      }
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
      if (prio == 0){
        schedctl( NDPRI, 0, NDPLOMIN );
        schedctl( RENICE, 0, 39);
        } 
      else{
        if (prio < 9)
          schedctl( NDPRI, 0, (NDPLOMIN - NDPNORMMIN)/prio);
        }
      }
  #else
    if ( set_for_thread )
      {
      #if defined(_POSIX_THREAD_PRIORITY_SCHEDULING) && defined(MULTITHREAD)
        //nothing - priority is set when created
      #elif (defined(_POSIX_THREADS) || defined(_PTHREAD_H)) && defined(MULTITHREAD)
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

