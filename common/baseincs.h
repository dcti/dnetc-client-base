// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// This file #includes the common header files that the client needs
// almost everywhere.
//
// ------------------------------------------------------------------
//
// $Log: baseincs.h,v $
// Revision 1.54  1999/01/31 20:19:07  cyp
// Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.53  1999/01/31 14:00:47  snake
// small correction of defines for BSD/OS 4
//
// Revision 1.52  1999/01/29 19:34:08  jlawson
// added limits.h to linux
//
// Revision 1.51  1999/01/29 19:07:16  jlawson
// fixed formatting.  added limits.h to win32
//
// Revision 1.50  1999/01/28 00:16:49  trevorh
// Minor updates for OS/2 with Watcom
//
// Revision 1.49  1999/01/21 05:02:41  pct
// Minor updates for Digital Unix clients.
//
// Revision 1.48  1999/01/19 12:51:01  patrick
// added strings.h for AIX
//
// Revision 1.47  1999/01/18 15:22:55  patrick
// added/changed some OS2 includes to work also for gcc.
// added unistd.h for AIX
//
// Revision 1.46  1999/01/11 11:52:35  snake
// small openbsd fix
//
// Revision 1.45  1999/01/06 22:14:47  dicamillo
// Support PPC prototype machines.
//
// Revision 1.44  1999/01/06 06:04:02  cramer
// cleaned up some of the solaris/sunos updates
//
// Revision 1.43  1999/01/02 07:13:33  dicamillo
// Remove sched.h for BeOS.
//
// Revision 1.42  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.41  1998/12/31 11:28:26  cyp
// Moved inclusion of sys/stat.h and sys_stat.h from buffwork.cpp to baseincs.h
//
// Revision 1.40  1998/12/31 08:06:26  dicamillo
// Add UseMP function for MacOS.
//
// Revision 1.39  1998/12/29 09:27:50  dicamillo
// For MacOS, add macConOut routine.
//
// Revision 1.38  1998/12/22 15:58:24  jcmichot
// *** empty log message ***
//
// Revision 1.37  1998/12/15 07:00:21  dicamillo
// Use "_" instead of "/" in Mac header file names for CVS.
//
// Revision 1.36  1998/12/14 05:09:16  dicamillo
// Fix formatting error in log comment.
//
// Revision 1.35  1998/12/14 05:05:04  dicamillo
// MacOS updates to eliminate MULTITHREAD and have a singe client for MT
// and non-MT machines
//
// Revision 1.34  1998/12/08 05:27:51  dicamillo
// Add includes for MacOS
//
// Revision 1.33  1998/11/25 09:23:26  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.32  1998/11/25 05:59:36  dicamillo
// Header changes for BeOS R4.
//
// Revision 1.31  1998/11/09 01:17:46  remi
// Linux/aout doesn't have <sched.h>
//
// Revision 1.30  1998/10/31 21:59:12  silby
// Added in an OS_FREEBSD include that was missing.
//
// Revision 1.29  1998/10/30 00:14:07  foxyloxy
// Added unistd.h to Irix standard includes.
//
// Revision 1.28  1998/10/26 03:21:26  cyp
// More tags fun.
//
// Revision 1.27  1998/10/19 12:42:14  cyp
// win16 changes
//
// Revision 1.26  1998/10/11 00:36:38  cyp
// include "w32pre.h" for win32
//
// Revision 1.25  1998/10/06 15:08:27  blast
// changed some AmigaOS includes...
//
// Revision 1.24  1998/10/04 01:30:54  silby
// Removed direct references to platforms/win32cli (makefile handles them)
//
// Revision 1.23  1998/10/03 03:47:01  cyp
// added 3 local header files (w32svc, w32cons, lurk) to the win32 section
// and 2 header files to os2 (os2inst, lurk)
//
// Revision 1.22  1998/09/30 08:12:32  snake
// Removed NASM stuff for BSD/OS, nasm does not support the a.out format of
// BSD/OS.
//
// Revision 1.21  1998/09/30 07:41:05  snake
// BSD/OS also needs <errno.h>, maybe we should include it for all BSD
// like OS's
//
// Revision 1.20  1998/09/29 23:14:19  silby
// Fix for the syntax of the last fix. :)
//
// Revision 1.19  1998/09/29 23:11:25  silby
// Change for freebsd (errno.h)
//
// Revision 1.18  1998/09/28 21:04:03  remi
// Added #include <errno.h> for Linux/glibc2.
//
// Revision 1.17  1998/09/25 11:31:14  chrisb
// Added stuff to support 3 cores in the ARM clients.
//
// Revision 1.16  1998/09/20 15:21:56  blast
// AmigaOS changes (added lines somebody cut out before .. grrr)
//
// Revision 1.15  1998/09/07 18:22:51  blast
// Added fcntl.h for AmigaOS
//
// Revision 1.14  1998/08/05 15:29:41  cyruspatel
// Added <ctype.h>
//
// Revision 1.13  1998/08/02 03:16:16  silby
// Major reorganization: Log,LogScreen, and LogScreenf are now in
// logging.cpp, and are global functions - client.h #includes
// logging.h, which is all you need to use those functions.  Lurk
// handling has been added into the Lurk class, which resides in
// lurk.cpp, and is auto-included by client.h if lurk is defined as
// well. baseincs.h has had lurk-specific win32 includes moved to
// lurk.cpp, cliconfig.cpp has been modified to reflect the changes to
// log/logscreen/logscreenf, and mail.cpp uses logscreen now, instead
// of printf. client.cpp has had variable names changed as well, etc.
//
// Revision 1.12  1998/07/29 05:14:31  silby
// Changes to win32 so that LurkInitiateConnection now works -
// required the addition of a new .ini key connectionname=.  Username
// and password are automatically retrieved based on the
// connectionname.
//
// Revision 1.11  1998/07/29 03:21:24  silby
// Added changes for win32 lurk (more needed functions)
//
// Revision 1.10 1998/07/25 06:31:37 silby Added lurk functions to
// initiate a connection and hangup a connection.  win32 hangup is
// functional.
//
// Revision 1.9  1998/07/16 21:47:56  nordquist
// More DYNIX port changes.
//
// Revision 1.8  1998/07/16 21:23:01  nordquist
// More DYNIX port changes.
//
// Revision 1.7  1998/07/15 05:47:42  ziggyb
// added io.h for Watcom, which has some IO functions not in stdio.h
//
// Revision 1.6  1998/07/13 12:40:23  kbracey
// RISC OS update. Added -noquiet option.
//
// Revision 1.5  1998/07/13 00:37:25  silby
// Changes to make MMX_BITSLICE client buildable on freebsd
//
// Revision 1.4  1998/07/12 13:05:11  cyruspatel
// NetWare changes.
//
// Revision 1.3  1998/07/08 05:19:16  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.2  1998/07/07 23:05:20  jlawson
// added time includes for Linux (probably will be needed for others)
//
// Revision 1.1  1998/07/07 21:55:01  cyruspatel
// client.h has been split into client.h and baseincs.h
//

#ifndef __BASEINCS_H__
#define __BASEINCS_H__

#include "cputypes.h"

// --------------------------------------------------------------------------

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#if (CLIENT_OS == OS_MACOS)
#include <sys_types.h> /* in platforms/macos */
#include <sys_stat.h>  /* in platforms/macos */
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif
#include <errno.h>
#include <limits.h>

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS == OS_IRIX)
  #include <unistd.h>
  #include <sys/types.h>
  #include <sys/prctl.h>
  #include <sys/schedctl.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_OS2)
  #include <sys/timeb.h>
  #include <share.h>
  #if defined(__WATCOMC__)
    #include <direct.h>
  #endif
  #include <fcntl.h>
  #include <io.h>
  #include "platforms/os2cli/os2defs.h"
  #if !defined(__EMX__)               // supported in Watcom
  #include <net/if.h>
  #include "platforms/os2cli/dod.h"   // needs to be included after Client
  #endif
  #include "lurk.h"
  #include "platforms/os2cli/os2inst.h" //-install/-uninstall functionality
  #ifndef QSV_NUMPROCESSORS       /* This is only defined in the SMP toolkit */
    #define QSV_NUMPROCESSORS     26
  #endif
#elif (CLIENT_OS == OS_AMIGAOS)
  #include <amiga/amiga.h>
  #include <unistd.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_RISCOS)
  extern "C"
  {
    #include <sys/fcntl.h>
    #include <unistd.h>
    #include <stdarg.h>
    #include <machine/endian.h>
    #include <sys/time.h>
    #include <swis.h>
    extern unsigned int ARMident(), IOMDident();
    extern void riscos_clear_screen();
    extern int riscos_check_taskwindow();
    extern void riscos_backspace();
    extern int riscos_count_cpus();
    extern char *riscos_x86_determine_name();
    extern int riscos_find_local_directory(const char *argv0);
    extern char *riscos_localise_filename(const char *filename);
    extern void riscos_upcall_6(void); //yield
    extern int getch();
    #define fileno(f) ((f)->__file)
    #define isatty(f) ((f) == 0)
  }
  extern s32 guiriscos, guirestart;
  extern int riscos_in_taskwindow;
#elif (CLIENT_OS == OS_VMS)
  #include <fcntl.h>
  #include <types.h>
  #define unlink remove
#elif (CLIENT_OS == OS_SCO)
  #include <fcntl.h>
  #include <sys/time.h>
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN16)
  #if (CLIENT_OS == OS_WIN32) || !defined(__WINDOWS_386__)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winsock.h>      // timeval
  #else
    #include <windows.h>
    #include "w32sock.h"
  #endif
  #include <sys/timeb.h>
  #include <process.h>
  #include <conio.h>
  #include <share.h>
  #include <fcntl.h>
  #include <io.h>
  #include "lurk.h"
  #include "w32svc.h"       // service
  #include "w32cons.h"      // console
  #include "w32pre.h"       // prelude
#elif (CLIENT_OS == OS_DOS)
  #include <sys/timeb.h>
  #include <io.h>
  #include <conio.h>
  #include <share.h>
  #include <fcntl.h>
  #include <dos.h> //for drive functions in pathwork.cpp
  #include "platforms/dos/clidos.h"
  #if defined(__WATCOMC__)
    #include <direct.h> //getcwd
  #elif defined(__TURBOC__)
    #include <dir.h>
  #endif
#elif (CLIENT_OS == OS_BEOS)
  #include <OS.h>
  #include <unistd.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_NETWARE)
  #include <sys/time.h> //timeval
  #include <unistd.h> //isatty, chdir, getcwd, access, unlink, chsize, O_...
  #include <conio.h> //ConsolePrintf(), clrscr()
  #include <share.h> //SH_DENYNO
  #include <nwfile.h> //sopen()
  #include <fcntl.h> //O_... constants
  #include "platforms/netware/netware.h" //for stuff in netware.cpp
#elif (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_SOLARIS)
  #include <fcntl.h>
  #include <unistd.h>
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int);
#elif (CLIENT_OS == OS_AIX)
  #include <unistd.h>		// nice()
  #include <strings.h>		// bzero(), strcase...,
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_BSDI) || (CLIENT_OS == OS_OPENBSD)
  #include <sys/time.h>
  #include <unistd.h>
  #if (((CLIENT_OS == OS_LINUX) && (__GLIBC__ >= 2)) || (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_BSDI))
    #include <errno.h> // glibc2 has errno only here
  #endif
  #if (((CLIENT_OS == OS_LINUX) && defined(__ELF__)) || \
    (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_BSDI))
    #include <sched.h>
  #endif
#elif (CLIENT_OS == OS_NETBSD) && (CLIENT_CPU == CPU_ARM)
  #include <sys/time.h>
#elif (CLIENT_OS == OS_QNX)
  #include <sys/time.h>
  #include <sys/select.h>
  #define strncmpi strncasecmp
#elif (CLIENT_OS == OS_DYNIX)
  #include <unistd.h> // sleep(3c)
  struct timezone
  {
    int  tz_minuteswest;    /* of Greenwich */
    int  tz_dsttime;        /* type of dst correction to apply */
  };
  extern "C" int gethostname(char *, int);
  extern "C" int gettimeofday(struct timeval *, struct timezone *);
#elif (CLIENT_OS == OS_MACOS)
  #include <sys_time.h>
  #include <stat.mac.h>
  #include <machine_endian.h>
  #include <unistd.h>
  #define _UTIME
  #include <unix.mac.h>
  #include "mac_extras.h"
  #include <console.h>
  #include <Multiprocessing.h>
  void macConOut(char *msg);
  void YieldToMain(char force_events);
  u32 GetTimesliceToUse(u32 contestid);
  void tick_sleep(unsigned long tickcount);
  extern Boolean Mac_PPC_prototype;
  extern Boolean haveMP;
  extern short MP_active;
  extern "C" unsigned long mp_sleep(unsigned long seconds);
  extern MPCriticalRegionID MP_count_region;
  extern char useMP(void);
  extern volatile s32 ThreadIsDone[2*MAC_MAXCPUS];
  #if defined(MAC_GUI)
    #include "gui_incs.h"
  #endif
#elif (CLIENT_OS == OS_DEC_UNIX)
  #include <unistd.h>
#endif

// --------------------------------------------------------------------------

#endif //__BASEINCS_H__

