// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
/* This include file ensures that sleep() and usleep() are valid.
   They MUST actually block/yield for approx. the duration requested.

   Porter notes: Check network.cpp and network.h to remove duplicate
   or conflicting defines or code in the platform specific sections there.

  1. if your platform does not support frac second sleep, try using
     select() as a substitute: For example:
     #define usleep(x) { struct timeval tv = {0,(x)}; \
                         select(0,NULL,NULL,NULL,&tv); }
     (Not all implementations support this: some don't sleep at all, 
     while others sleep forever)
  2. if usleep(x) or sleep(x) are macros, make sure that 'x' is
     enclosed in parens. ie #define sleep(x) myDelay((x)/1000)
*/
//
// $Log: sleepdef.h,v $
// Revision 1.19  1999/02/21 21:44:59  cyp
// tossed all redundant byte order changing. all host<->net order conversion
// as well as scram/descram/checksumming is done at [get|put][net|disk] points
// and nowhere else.
//
// Revision 1.18  1998/12/08 06:00:38  dicamillo
// Add definitions for MacOS.
//
// Revision 1.17  1998/10/30 00:06:05  foxyloxy
//
// Changed sginap() multiplier to be correct.
//
// Revision 1.16  1998/10/26 03:20:54  cyp
// More tags fun.
//
// Revision 1.15  1998/10/19 12:42:17  cyp
// win16 changes
//
// Revision 1.14  1998/10/06 15:19:16  blast
// Changed the sleep() functions used in AmigaOS is the minumum sleep
// we need in the client is 55ms or something like that and the AmigaOS
// Delay() function is 1 clock tick (16/20ms)
//
// Revision 1.13  1998/09/28 02:05:38  cyp
// Modified for use with pollsys. Fixed (?) IRIX's lack of usleep() to use
// sginap().
//
// Revision 1.12  1998/09/25 04:32:15  pct
// DEC Ultrix port changes
//
// Revision 1.11  1998/08/10 10:16:35  cyruspatel
// DOS port changes.
//
// Revision 1.10  1998/07/16 20:12:58  nordquist
// DYNIX port changes.
//
// Revision 1.9  1998/06/29 06:58:12  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.8  1998/06/29 04:22:28  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.7  1998/06/25 20:57:08  foxyloxy
// IRIX5's lack of usleep() rectified temporarily with sleep()... will change
// to nanosleep() soon. (sleepdef.h) -MIPS4 changed to -mips4 (configure)
//
// Revision 1.6  1998/06/22 01:05:03  cyruspatel
// DOS changes. Fixes various compile-time errors: removed extraneous ')' in
// sleepdef.h.
//
// Revision 1.5  1998/06/15 09:12:56  jlawson
// moved more sleep defines into sleepdef.h
//
// Revision 1.4  1998/06/14 11:25:13  ziggyb
// Made the sleep/usleep defines work correctly in OS/2
//
// Revision 1.3  1998/06/14 08:13:10  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.0  1998/05/01 05:01:08  cyruspatel
// Created to support real sleep periods (needed for buffwork revision)
// 

#ifndef __SLEEPDEF_H__
#define __SLEEPDEF_H__

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
  //thats a wrapper around a dpmi yield
  #include "platforms/dos/clidos.h"
  #define sleep(x) dosCliSleep((x))
  #define usleep(x) dosCliUSleep((x))
#elif (CLIENT_OS == OS_OS2)
  #include "platforms/os2cli/os2defs.h"
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
    #define usleep(x) sleep(1)
    #define USLEEP_IS_SLEEP
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
      #define USLEEP_IS_SLEEP
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
  }
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <unistd.h>
  }
#elif (CLIENT_OS == OS_DYNIX)
  // DYNIX doesn't have nanosleep() or usleep()
  #define usleep(x) sleep(1)
  #define USLEEP_IS_SLEEP
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
