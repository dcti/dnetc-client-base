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
// Revision 1.11  1998/07/29 03:21:24  silby
// Added changes for win32 lurk (more needed functions)
//
// Revision 1.10  1998/07/25 06:31:37  silby
// Added lurk functions to initiate a connection and hangup a connection.  win32 hangup is functional.
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
// RISC OS update.
// Added -noquiet option.
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
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
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
#include <sys/types.h>

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
  #ifndef QSV_NUMPROCESSORS       /* This is only defined in the SMP toolkit */
    #define QSV_NUMPROCESSORS     26
  #endif
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
  #if (CLIENT_OS == OS_WIN32)
    #include <ras.h>
    typedef DWORD (WINAPI *rasenumconnectionsT)(LPRASCONN, LPDWORD, LPDWORD);
    typedef DWORD (WINAPI *rasgetconnectstatusT)(HRASCONN, LPRASCONNSTATUS);
    typedef DWORD (WINAPI *rashangupT)(HRASCONN); 
    typedef DWORD (WINAPI *rasdialT)(LPRASDIALEXTENSIONS, LPSTR,
                   LPRASDIALPARAMS, DWORD, LPVOID, LPHRASCONN);
    typedef DWORD (WINAPI *rasgeterrorstringT)(UINT, LPTSTR, DWORD);
  #endif
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
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)
  #include <sys/time.h>
  #include <unistd.h>
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
