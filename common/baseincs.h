// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: baseincs.h,v $
// Revision 1.30.2.7  1999/01/04 01:50:06  remi
// Synced with :
//
//  Revision 1.43  1999/01/02 07:13:33  dicamillo
//  Remove sched.h for BeOS.
//
//  Revision 1.42  1999/01/01 02:45:14  cramer
//  Part 1 of 1999 Copyright updates...
//
//  Revision 1.41  1998/12/31 11:28:26  cyp
//  Moved inclusion of sys/stat.h and sys_stat.h from buffwork.cpp to baseincs.h
//
//  Revision 1.40  1998/12/31 08:06:26  dicamillo
//  Add UseMP function for MacOS.
//
// Revision 1.30.2.6  1998/12/29 11:19:32  remi
// Synced with :
//
//  Revision 1.39  1998/12/29 09:27:50  dicamillo
//  For MacOS, add macConOut routine.
//
// Revision 1.30.2.5  1998/12/28 15:29:38  remi
// Synced with :
//
//  Revision 1.38  1998/12/22 15:58:24  jcmichot
//  QNX port.
//
//  Revision 1.37  1998/12/15 07:00:21  dicamillo
//  Use "_" instead of "/" in Mac header file names for CVS.
//
//  Revision 1.36  1998/12/14 05:09:16  dicamillo
//  Fix formatting error in log comment.
//
//  Revision 1.35  1998/12/14 05:05:04  dicamillo
//  MacOS updates to eliminate MULTITHREAD and have a singe client for MT
//  and non-MT machines
//
//  Revision 1.34  1998/12/08 05:27:51  dicamillo
//  Add includes for MacOS
//
//  Revision 1.33  1998/11/25 09:23:26  chrisb
//  various changes to support x86 coprocessor under RISC OS
//
//  Revision 1.32  1998/11/25 05:59:36  dicamillo
//  Header changes for BeOS R4.
//
// Revision 1.30.2.4  1998/11/16 09:56:01  remi
// In win32 section, fixed two #include.
//
// Revision 1.30.2.3  1998/11/15 15:36:05  remi
// Synced with :
//  Revision 1.31  1998/11/09 01:17:46  remi
//  Linux/aout doesn't have <sched.h>
//
// Revision 1.30.2.2  1998/11/08 11:38:38  remi
// Added $Log tag
//
// Synchronized with official 1.30

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

#if ((CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS))
}
#endif

#if (CLIENT_OS == OS_IRIX)
  #include <limits.h>
  #include <unistd.h>
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
  extern bool riscos_in_taskwindow;
#elif (CLIENT_OS == OS_VMS)
  #include <fcntl.h>
  #include <types.h>
  #define unlink remove
#elif (CLIENT_OS == OS_SCO)
  #include <fcntl.h>
  #include <sys/time.h>
#elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN32S) || (CLIENT_OS==OS_WIN16)
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
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int);
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS==OS_BSDI)
  #include <sys/time.h>
  #include <unistd.h>
  #if (((CLIENT_OS == OS_LINUX) && (__GLIBC__ >= 2)) || (CLIENT_OS==OS_FREEBSD) || (CLIENT_OS==OS_BSDI))
    #include <errno.h> // glibc2 has errno only here
  #endif
  #if (((CLIENT_OS == OS_LINUX) && defined(__ELF__)) || \
       (CLIENT_OS == OS_FREEBSD))
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
  extern Boolean haveMP;
  extern short MP_active;
  extern "C" unsigned long mp_sleep(unsigned long seconds);
  extern MPCriticalRegionID MP_count_region;
  extern char useMP(void);
  extern volatile s32 ThreadIsDone[2*MAC_MAXCPUS];
  #if defined(MAC_GUI)
    #include "gui_incs.h"
  #endif
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
