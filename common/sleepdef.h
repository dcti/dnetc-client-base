/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ------------------------------------------------------------------
 * This include file ensures that sleep() and usleep() are valid.
 * They MUST actually block/yield for approx. the duration requested.
 * "approx" does not mean a factor of. :)
 *
 * Porter notes: Check network.cpp and network.h to remove duplicate
 * or conflicting defines or code in the platform specific sections there.
 *
 * 1. if your platform does not support frac second sleep, try using
 *    select() as a substitute: For example:
 *    #define usleep(x) { struct timeval tv = {0,(x)}; \
 *                         select(0,NULL,NULL,NULL,&tv); }
 *    (Not all implementations support this: some don't sleep at all, 
 *    while others sleep forever)
 * 2. If you can't use select() for usleep()ing, you can still roll a 
 *    usleep() using your sched_yield()[or whatever] and gettimeofday() 
 * 3. if usleep(x) or sleep(x) are macros, make sure that 'x' is
 *    enclosed in parens. ie #define sleep(x) myDelay((x)/1000)
 *    otherwise expect freak outs with sleep(sleepstep+10) and the like.
 * ------------------------------------------------------------------
*/ 
#ifndef __SLEEPDEF_H__
#define __SLEEPDEF_H__ "@(#)$Id: sleepdef.h,v 1.19.2.1 1999/04/13 19:45:31 jlawson Exp $"

#include "cputypes.h"

#if (CLIENT_OS == OS_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #ifdef sleep
    #undef sleep
  #endif
  #define sleep(x) Sleep(1000*(x))
  #define usleep(x) Sleep((x)/1000)
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  // Win16 has Yield(), but we have a gui, so pump messages instead
  #include "w32cons.h"
  #define sleep(x)  w32Sleep((x)*1000)
  #define usleep(x) w32Sleep((x)/1000)
#elif (CLIENT_OS == OS_DOS)
  //usleep() and sleep() are wrappers around dpmi yield
  #include "platforms/dos/clidos.h"
#elif (CLIENT_OS == OS_OS2)
  #ifdef sleep
  #undef sleep    // gets rid of warning messages
  #endif
  #define sleep(x) DosSleep(1000*(x))
  #define usleep(x) DosSleep((x)/1000)
#elif (CLIENT_OS == OS_NETWARE)
  extern "C" void delay(unsigned int msecs);
  #define sleep(x)  delay((x)*1000)
  #define usleep(x) delay((x)/1000)
#elif (CLIENT_OS == OS_BEOS)
  #include <unistd.h>
  #define usleep(x) snooze((x))
#elif (CLIENT_OS == OS_MACOS)
  #include <unistd.h>
  void usleep(unsigned int usecs);
  #define sleep(x) my_sleep(x)
  unsigned int my_sleep(unsigned int seconds);
#elif (CLIENT_OS == OS_DEC_UNIX)
  #include <unistd.h>
  #include <sys/types.h>
  // found in <unistd.h>, but requires _XOPEN_SOURCE_EXTENDED,
  // which causes more trouble...
  extern "C" int usleep(useconds_t);
#elif ((CLIENT_OS == OS_SUNOS) && (CLIENT_CPU == CPU_68K))
  #include <unistd.h>
  extern "C" void usleep(unsigned int);
#elif (CLIENT_OS == OS_HPUX)
  #include <unistd.h>
  #include <sys/time.h>
  #ifdef _STRUCT_TIMESPEC
     // HP-UX 10.x has nanosleep() rather than usleep()
     #define usleep(x) { \
       struct timespec interval, remainder; \
       interval.tv_sec = 0; interval.tv_nsec = (x)*100; \
       nanosleep(&interval, &remainder); }
  #else // HP-UX 9.x doesn't have nanosleep() or usleep()
    //#define usleep(x) sleep(1)
    #error FIXME - is this correct?
    #undef usleep
    #define usleep(x) {struct timeval tv={0,(x)};select(0,NULL,NULL,NULL,&tv);}
  #endif
#elif (CLIENT_OS == OS_IRIX)
  #include <unistd.h>
  #if 1
    #ifndef usleep
      #include <limits.h>
      #define usleep(x) sginap((x)*(CLK_TCK/1000000L))
    #endif
  #else
    #ifdef _irix5_
      #define usleep(x) sleep(1) //will use nanosleep() in next revision
      #error please fix "next revision" (see hpux above)
    #endif
  #endif
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
  #error Intentionally left unfixed. *Read* the documentation about using parens!
  }
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <unistd.h>
  }
#elif (CLIENT_OS == OS_DYNIX)
  // DYNIX doesn't have nanosleep() or usleep()
  //#define usleep(x) sleep(1)
  #error FIXME - is this correct?
  #undef usleep
  #define usleep(x) {struct timeval tv={0,(x)};select(0,NULL,NULL,NULL,&tv);}
#elif (CLIENT_OS == OS_ULTRIX)
  #define usleep(x) { struct timeval tv = {0,(x)}; \
                      select(0,NULL,NULL,NULL,&tv); }
#else
  #include <unistd.h> //gcc has both sleep() and usleep()
#endif

#ifndef __SLEEP_FOR_POLLING__
#include "pollsys.h"
#undef  sleep
#define sleep(x) PolledSleep(x)
#undef  usleep
#define usleep(x) PolledUSleep(x)
#endif /* __SLEEP_FOR_POLLING__ */

#endif /* __SLEEPDEF_H__ */

