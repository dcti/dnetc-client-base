// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// This file #includes the common header files that the client needs
// almost everywhere.
//
// ------------------------------------------------------------------
//
// $Log: baseincs.h,v $
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
#include <sys/types.h>
#include <errno.h>

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS == OS_IRIX)
  #include <limits.h>
  #include <sys/types.h>
  #include <sys/prctl.h>
  #include <sys/schedctl.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_OS2)
  #include <sys/timeb.h>
  #include <conio.h>
  #include <share.h>
  #include <direct.h>
  #include <fcntl.h>
  #include <io.h>
  #include "platforms/os2cli/os2defs.h"
  #include "platforms/os2cli/dod.h"   // needs to be included after Client
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
    extern bool riscos_check_taskwindow();
    extern int riscos_find_local_directory(const char *argv0);
    extern char *riscos_localise_filename(const char *filename);
    extern void riscos_upcall_6(void); //yield
    extern int getch();

    #define fileno(f) ((f)->__file)
    #define isatty(f) ((f) == 0)
  }
  extern s32 guiriscos, guirestart;
  extern bool riscos_in_taskwindow;
#elif (CLIENT_OS == OS_VMS)
  #include <fcntl.h>
  #include <types.h>
  #define unlink remove
#elif (CLIENT_OS == OS_SCO)
  #include <fcntl.h>
  #include <sys/time.h>
#elif (CLIENT_OS == OS_WIN16)
  #include <sys/timeb.h>
  #include <io.h>
  #include <conio.h>
  #include <dos.h>
  #include <share.h>
  #include <dir.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S)
  #include <sys/timeb.h>
  #include <process.h>
  #include <conio.h>
  #include <share.h>
  #include <fcntl.h>
  #include <io.h>
  #if defined(__TURBOC__)
    #include <dir.h>
  #endif
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <winsock.h>      // timeval
  #include "lurk.h"
  #include "w32svc.h" //service
  #include "w32cons.h" //console
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
// nothing  #include <share.h>
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
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int);
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS==OS_BSDI)
  #include <sys/time.h>
  #include <unistd.h>
  #if (((CLIENT_OS == OS_LINUX) && (__GLIBC__ >= 2)) || (CLIENT_OS==OS_FREEBSD) || (CLIENT_OS==OS_BSDI))
    #include <errno.h> // glibc2 has errno only here
  #endif
#elif (CLIENT_OS == OS_NETBSD) && (CLIENT_CPU == CPU_ARM)
  #include <sys/time.h>
#elif (CLIENT_OS == OS_DYNIX)
  #include <unistd.h> // sleep(3c)
  struct timezone
  {
    int  tz_minuteswest;    /* of Greenwich */
    int  tz_dsttime;        /* type of dst correction to apply */
  };
  extern "C" int gethostname(char *, int);
  extern "C" int gettimeofday(struct timeval *, struct timezone *);
#endif

// --------------------------------------------------------------------------

#ifdef max
#undef max
#endif
#define max(a,b)            (((a) > (b)) ? (a) : (b))

#ifdef min
#undef min
#endif
#define min(a,b)            (((a) < (b)) ? (a) : (b))

// --------------------------------------------------------------------------

#endif //__BASEINCS_H__
