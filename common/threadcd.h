// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: threadcd.h,v $
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

#if (CLIENT_OS == OS_WIN32)
   #include <process.h>
   typedef unsigned long THREADID;
#elif (CLIENT_OS == OS_OS2)
   #error threadcd.h needs an include //replace this with appropriate includes
   typedef long THREADID;
#elif (CLIENT_OS == OS_NETWARE)
   #include <process.h>
   typedef long THREADID;
   extern int CliWaitForThreadExit( int threadID );
#elif (CLIENT_OS == OS_BEOS)
   #error threadcd.h needs an include //replace this with appropriate includes
   typedef thread_id THREADID;
#elif (CLIENT_OS == OS_DOS)
   typedef int THREADID ; //dummy
   #undef OS_SUPPORTS_THREADING
#elif (CLIENT_OS == OS_MACOS)
   typedef int THREADID ; //dummy
   #undef OS_SUPPORTS_THREADING
#elif (CLIENT_OS == OS_WIN16)
   typedef int THREADID ; //dummy
   #undef OS_SUPPORTS_THREADING
#elif (CLIENT_OS == OS_RISCOS)
   typedef int THREADID ; //dummy
   #undef OS_SUPPORTS_THREADING
#else
   #if !defined(MULTITHREAD)
     typedef int THREADID ;
     #undef OS_SUPPORTS_THREADING
   #else
     #include <pthread.h>
     typedef pthread_t THREADID;
   #endif
#endif

//-----------------------------------------------------------------------

// create a thread (blocks till running) - returns threadid or NULL if error
THREADID CliCreateThread( void (*proc)(void *), void *param );

// destroy a thread (blocks till dead)
int CliDestroyThread( THREADID cliThreadID );

//-----------------------------------------------------------------------

#endif //ifndef __CLITHREAD_H__

