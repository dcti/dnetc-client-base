/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __BASEINCS_H__
#define __BASEINCS_H__ "@(#)$Id: baseincs.h,v 1.78 2000/01/08 23:36:03 cyp Exp $"

#include "cputypes.h"

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#if defined(__unix__)
#include <sys/utsname.h> /* uname() */
#endif
}

#if (CLIENT_OS == OS_IRIX)
  #include <unistd.h>
  #include <sys/types.h>
  #include <sys/prctl.h>
  #include <sys/schedctl.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_HPUX)
  #include <unistd.h>
  #include <sys/types.h>
  #include <fcntl.h>
  #include <sys/param.h>
  #include <sys/pstat.h>
#elif (CLIENT_OS == OS_OS2)
  #if defined(__WATCOMC__)
    #include "os2defs.h"
    #include <conio.h>            /* for console functions */
    #include <direct.h>
    #include <process.h>
  #endif
  #define INCL_DOSPROCESS         /* For Disk functions */
  #define INCL_DOSFILEMGR         /* For Dos_Delete */
  #define INCL_ERRORS             /* DOS error values */
  #define INCL_DOSMISC            /* DosQuerySysInfo() */
  #define INCL_WINWORKPLACE       /* Workplace shell objects */
  #define INCL_VIO                /* OS/2 text graphics functions */
  #define INCL_DOS
  #define INCL_SUB
  #include <os2.h>
  #include <sys/timeb.h>
  #include <share.h>
  #include <fcntl.h>
  #include <io.h>
  #include <sys/time.h>         /* timeval */
  #if defined(OS2_PM)
    #include "platforms/os2gui/os2cons.h"
  #endif
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
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
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
  #ifdef _MSC_VER
  // msc equivalents of file perm flags
  #define R_OK 04
  #define W_OK 02
  #define S_IRUSR _S_IREAD
  #define S_IWUSR _S_IWRITE
  #define S_IRGRP _S_IREAD
  #define S_IWGRP _S_IWRITE
  #endif
#elif (CLIENT_OS == OS_DOS)
  #include <sys/timeb.h>
  #include <io.h>
  #include <conio.h>
  #include <share.h>
  #include <fcntl.h>
  #include <dos.h> //for drive functions in pathwork.cpp
  #include "platforms/dos/clidos.h" //gettimeofday(), usleep() etc
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
  #include <poll.h>
  #include <thread.h>
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int);
#elif (CLIENT_OS == OS_AIX)
  #include <unistd.h>		// nice()
  #include <strings.h>		// bzero(), strcase...,
  #include <sys/select.h>	// fd_set on AIX 4.1
#elif (CLIENT_OS == OS_LINUX)
  #include <sys/time.h>
  #include <unistd.h>
  #if defined(__ELF__)
    #include <sched.h>
  #endif
#elif (CLIENT_OS == OS_MACOS)  
  #include <sys/time.h>
  #include <unistd.h>
  #include <sched.h>
  #include <Gestalt.h>
  #include "client_defs.h"
#elif (CLIENT_OS == OS_FREEBSD)  
  #include <sys/time.h>
  #include <unistd.h>
  #include <sched.h>
#elif (CLIENT_OS == OS_OPENBSD)
  #include <sys/time.h>
  #include <unistd.h>
#elif (CLIENT_OS == OS_BSDOS)
  #include <sys/param.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <sched.h>
#elif (CLIENT_OS == OS_NETBSD)
  #include <sys/time.h>
  #include <unistd.h>
#elif (CLIENT_OS == OS_QNX)
  #include <sys/time.h>
  #include <sys/select.h>
  #define strncmpi strncasecmp
#elif (CLIENT_OS == OS_DYNIX)
  #include <unistd.h> // sleep(3c)
  struct timezone { int tz_minuteswest, tz_dsttime; };
  extern "C" int gethostname(char *, int);
  extern "C" int gettimeofday(struct timeval *, struct timezone *);
#elif (CLIENT_OS == OS_DEC_UNIX)
  #include <unistd.h>
#endif


#endif /* __BASEINCS_H__ */
