// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: sleepdef.h,v $
// Revision 1.6  1998/06/22 01:05:03  cyruspatel
// DOS changes. Fixes various compile-time errors: removed extraneous ')' in
// sleepdef.h, resolved htonl()/ntohl() conflict with same def in client.h
// (is now inline asm), added NONETWORK wrapper around Network::Resolve()
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
// 

// This include file ensures that sleep() and usleep() are valid.
// They MUST actually block/yield for approx. the duration requested.

// include this file from client.h and network.h

#ifndef __SLEEPDEF_H__
#define __SLEEPDEF_H__

#include "cputypes.h"

/* Porter notes: Check network.cpp and network.h to remove duplicate
   or conflicting defines or code in the platform specific sections there.

  1. if your platform does not support frac second sleep, try using
       select() as a substitute: For example:
       #define usleep(x) { struct timeval tv = {0,(x)}; \
                          select(0,NULL,NULL,NULL,&tv); }
     That is ok according to the posix definition, but some bsdsocket
     implementations don't support it and don't sleep or sleep forever.
  2. if usleep(x) or sleep(x) are macros, make sure that 'x' is
     enclosed in parens. ie #define sleep(x) myDelay((x)/1000)
*/

#if (CLIENT_OS == OS_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #ifdef sleep
    #undef sleep
  #endif
  #define sleep(x) Sleep(1000*(x))
  #define usleep(x) Sleep((x)/1000)
#elif (CLIENT_OS == OS_WIN16)
  #if defined(__WATCOMC__) || defined(__TURBOC__)
    #include <dos.h>
    #define usleep(x) delay((x)/1000)
  #else
    #error sleep()/usleep() are still undefined in sleepdef.h
  #endif
#elif (CLIENT_OS == OS_DOS)
  #if (defined(__WATCOMC__) || defined(__TURBOC__))
    #include <dos.h>
    #define usleep(x) delay((x)/1000)
  #elif defined(DJGPP)
    #include <unistd.h>
  #endif
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
  #endif
#elif (CLIENT_OS == OS_AMIGAOS)
  extern "C" {
  #include <unistd.h>
  }
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <unistd.h>
  }
#else
  #include <unistd.h> //gcc has both sleep() and usleep()
#endif



#endif

