// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: threadcd.h,v $
// Revision 1.13  1998/12/01 19:49:14  cyp
// Cleaned up MULT1THREAD #define: The define is used only in cputypes.h (and
// then undefined). New #define based on MULT1THREAD, CLIENT_CPU and CLIENT_OS
// are CORE_SUPPORTS_SMP, OS_SUPPORTS_SMP. If both CORE_* and OS_* support
// SMP, then CLIENT_SUPPORTS_SMP is defined as well. This should keep thread
// strangeness (as foxy encountered it) out of the picture. threadcd.h
// (and threadcd.cpp) are no longer used, so those two can disappear as well.
// Editorial note: The term "multi-threaded" is (and has always been)
// virtually meaningless as far as the client is concerned. The phrase we
// should be using is "SMP-aware".
//
// Revision 1.12  1998/07/16 20:14:37  nordquist
// DYNIX port changes.
//
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

#include "cputypes.h" /* THREADID typedef */

//-----------------------------------------------------------------------
// Once portable thread creation/destruction is guaranteed, it
// isn't a big step to add functionality to the client - For example:
// 1. a 'pipe' proxy (perhaps with IPX/NetBios support).
// 2. a 'lurk' mechanism running asynchronously.
//-----------------------------------------------------------------------

// create a thread (blocks till running) - returns threadid or NULL if error
extern THREADID CliCreateThread( register void (*proc)(void *), void *param );

// destroy a thread (blocks till dead)
extern int CliDestroyThread( THREADID cliThreadID );

//-----------------------------------------------------------------------

#endif //ifndef __CLITHREAD_H__
