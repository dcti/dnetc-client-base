/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ------------------------------------------------------------------
 * This include file ensures that sleep() and usleep() are valid.
 * They MUST actually block/yield for approx. the duration requested.
 * "approx" does not mean "a factor of". :)
 *
 * Porter notes: Check network.cpp and network.h to remove duplicate
 * or conflicting defines or code in the platform specific sections there.
 *
 * 1. if your platform does not support frac second sleep, try using
 *    one of the following substitutes:
 *    a) #define usleep(x) poll(NULL, 0, (x)/1000);
 *    b) #define usleep(x) { struct timespec interval, remainder; \
 *                           interval.tv_sec = 0; interval.tv_nsec = (x)*100;\
 *                           nanosleep(&_interval, &_remainder); }
 *    c) #define usleep(x) { struct timeval tv__ = {0,(x)}; \
 *                           select(0,NULL,NULL,NULL,&tv__); }
 *       (Not all implementations support (c): some don't sleep at all, 
 *       while others sleep forever)
 *    d) roll a usleep() using your sched_yield()[or whatever] and 
 *       gettimeofday() 
 * 2. if usleep(x) or sleep(x) are macros, make sure that 'x' is
 *    enclosed in parens. ie #define sleep(x) myDelay((x)/1000)
 *    otherwise expect freak outs with sleep(sleepstep+10) and the like.
 * ------------------------------------------------------------------
*/ 
#ifndef __SLEEPDEF_H__
#define __SLEEPDEF_H__ "@(#)$Id: sleepdef.h,v 1.31 2000/01/04 12:49:48 cyp Exp $"

#include "cputypes.h"

#if (CLIENT_OS == OS_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #ifdef sleep
    #undef sleep
  #endif
  #define sleep(x) Sleep(1000*(x))
  #define usleep(x) Sleep((x)/1000)
#elif (CLIENT_OS == OS_WIN16)
  // Win16 has Yield(), but we have a gui, so pump messages instead
  #include "w32cons.h"
  #define sleep(x)  w32Sleep((x)*1000)
  #define usleep(x) w32Sleep((x)/1000)
#elif (CLIENT_OS == OS_DOS)
  //usleep and sleep are wrapper around dpmi yield
  //emulated in platforms/dos/cdosyield.cpp
  #include <unistd.h>
  #undef sleep
  #undef usleep
  extern "C" void sleep(unsigned int);
  extern "C" void usleep(unsigned int);
#elif (CLIENT_OS == OS_OS2)
  #ifdef sleep
  #undef sleep    // gets rid of warning messages
  #endif
  #define sleep(x) DosSleep(1000*(x))
  #define usleep(x) DosSleep((x)/1000)
#elif (CLIENT_OS == OS_NETWARE)
  //emulated in platforms/netware/...
  extern "C" void usleep(unsigned long);
  extern "C" unsigned int sleep(unsigned int);
#elif (CLIENT_OS == OS_BEOS)
  #include <unistd.h>
  #define usleep(x) snooze((x))
#elif (CLIENT_OS == OS_DEC_UNIX)
  #include <unistd.h>
  #include <sys/types.h>
  // found in <unistd.h>, but requires _XOPEN_SOURCE_EXTENDED,
  // which causes more trouble...
  extern "C" int usleep(useconds_t);
#elif (CLIENT_OS == OS_IRIX)
  #include <unistd.h>
#ifdef _irix5_
  #undef usleep
  #define usleep(x) poll(NULL, 0, (x)/1000);
  //#define usleep(x) sginap((((x)*CLK_TCK)+500000)/1000000L)
  //CLK_TCK is defined as sysconf(_SC_CLK_TCK) in limits.h and 
  //is 100 (10ms) for non-realtime processes, machine dependant otherwise
#endif /* _irix5_ */
#elif (CLIENT_OS == OS_AMIGAOS)
  extern "C" {
  #ifdef sleep
  #undef sleep
  #endif
  #ifdef sleep
  #undef usleep
  #endif
  #define sleep(n) Delay(n*TICKS_PER_SECOND);
  #define usleep(n) Delay(n*TICKS_PER_SECOND/1000000);
  #error Intentionally left unfixed. (hint: use parenthesis)
  }
#elif (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_SOLARIS)
  //Jul '99: It appears Sol/Sparc and/or Sol/ultra have a
  //problem with usleep() (sudden crash with bad lib link)
  //whether x86/68k also have this bug is unknown. But to be
  //safe, we use poll() too since that is a direct syscall.
  #include <unistd.h>
  #undef usleep
  #define usleep(x) poll(NULL, 0, (x)/1000);
#elif (CLIENT_OS == OS_HPUX)
  #include <unistd.h>
  #include <sys/time.h>
  //HP-UX 10.x has nanosleep() but no usleep(), 9.x has neither, so ...
  //we do whats good for all HP-UX's (according to select(2) manpage)
  #undef usleep
  #define usleep(x) {struct timeval tv__={0,(x)};select(0,NULL,NULL,NULL,&tv__);}
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <unistd.h>
  }
#elif (CLIENT_OS == OS_DYNIX)
  // DYNIX doesn't have nanosleep() or usleep(), but has poll()
  #undef usleep
  #define usleep(x) poll(NULL, 0, (x)/1000);
#elif (CLIENT_OS == OS_ULTRIX)
  #include <sys/time.h>
  #define usleep(x) {struct timeval tv__={0,(x)};select(0,NULL,NULL,NULL,&tv__);}
#else
  #include <unistd.h> //has both sleep() and usleep()
#endif

#ifndef __SLEEP_FOR_POLLING__
#include "pollsys.h"
#undef  sleep
#define sleep(x) PolledSleep(x)
#undef  usleep
#define usleep(x) PolledUSleep(x)
#endif /* __SLEEP_FOR_POLLING__ */

#endif /* __SLEEPDEF_H__ */

