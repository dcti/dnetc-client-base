// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: threadcd.h,v $
// Revision 1.11  1998/07/14 00:45:31  cyruspatel
// Added a second define to differenciate between OS_SUPPORTS_THREADING and
// when special steps must be taken to support it, such as linking special
// libraries or whatever.
//
// Revision 1.10  1998/07/13 23:39:37  cyruspatel
// Added functions to format and display raw cpu info for better management
// of the processor detection functions and tables. Well, not totally raw,
// but still less cooked than SelectCore(). All platforms are supported, but
// the data may not be meaningful on all. The info is accessible to the user
// though the -cpuinfo switch.
//
// Revision 1.9  1998/07/13 03:32:00  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// ambiguities. ("declaration/type or an expression")
//
// Revision 1.8  1998/06/30 06:37:39  ziggyb
// OS/2 specific changes for threads.
//
// Revision 1.7  1998/06/29 06:58:14  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.6  1998/06/29 04:22:32  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.5  1998/06/16 05:38:08  dicamillo
// Added #include for BeOS for OS.h
//
// Revision 1.4  1998/06/14 08:13:14  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef __CLITHREAD_H__
#define __CLITHREAD_H__

#include "cputypes.h"

//-----------------------------------------------------------------------
// Note: #ifdef MULTITHREAD is intentionally avoided until
// the #if/#elif falls into the #else, and that too should be
// removed. This is to support platforms that support threading
// independantly of whether the client.cpp is being built with
// multithread support or not.
//
// Once portable thread creation/destruction is guaranteed, it
// isn't a big step to add functionality to the client - For example:
// 1. a 'pipe' proxy (perhaps with IPX/NetBios support).
// 2. a 'lurk' mechanism running asynchronously.
//-----------------------------------------------------------------------

#define OS_SUPPORTS_THREADING  //#undef this IN *YOUR* SECTION if untrue
#define THREADING_IS_AVAILABLE //special define in case OS_SUPPORTS_THREADING
                               //is true, but special libs (or whatever) must 
                               //be linked. see pthread support for example.

#if (CLIENT_OS == OS_WIN32)
  #include <process.h>
  typedef unsigned long THREADID;
  #define OS_SUPPORTS_THREADING
  #define THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_OS2)
//Headers defined elsewhere in a separate file.
  typedef long THREADID;
  #define OS_SUPPORTS_THREADING
  #define THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_NETWARE)
  #include <process.h>
  typedef long THREADID;
  extern int CliWaitForThreadExit( int threadID );
  #define OS_SUPPORTS_THREADING
  #define THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_BEOS)
  #include <OS.h>
  typedef thread_id THREADID;
  #define OS_SUPPORTS_THREADING
  #define THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_DOS)
  typedef int THREADID ; //dummy
  #undef OS_SUPPORTS_THREADING
  #undef THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_MACOS)
  typedef int THREADID ; //dummy
  #undef OS_SUPPORTS_THREADING
  #undef THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  #include <stdlib.h>
  typedef int THREADID ; //dummy
  #undef OS_SUPPORTS_THREADING
  #undef THREADING_IS_AVAILABLE
#elif (CLIENT_OS == OS_RISCOS)
  typedef int THREADID ; //dummy
  #undef OS_SUPPORTS_THREADING
  #undef THREADING_IS_AVAILABLE
#else
  #if !defined(MULTITHREAD)
    typedef int THREADID ; //dummy
    #undef THREADING_IS_AVAILABLE
    #if ((CLIENT_OS != OS_LINUX) && (CLIENT_OS != OS_DGUX) && \
        (CLIENT_OS != OS_SOLARIS) && (CLIENT_OS != OS_FREEBSD))
      #undef OS_SUPPORTS_THREADING
    #endif 
  #else
    #include <pthread.h>
    typedef pthread_t THREADID;
    #define THREADING_IS_AVAILABLE
  #endif
#endif

//-----------------------------------------------------------------------

// create a thread (blocks till running) - returns threadid or NULL if error
extern THREADID CliCreateThread( register void (*proc)(void *), void *param );

// destroy a thread (blocks till dead)
extern int CliDestroyThread( THREADID cliThreadID );

//-----------------------------------------------------------------------

#endif //ifndef __CLITHREAD_H__

