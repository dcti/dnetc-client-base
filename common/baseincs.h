/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __BASEINCS_H__
#define __BASEINCS_H__ "@(#)$Id: baseincs.h,v 1.65.2.47 2001/01/16 21:00:25 teichp Exp $"

#include "cputypes.h"

#if (CLIENT_OS == OS_RISCOS)
 extern "C" { /* headers are unsafe for c++ */
#endif

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

#if (CLIENT_OS == OS_RISCOS)
} /* End the extern needed to handle unsafe standard headers. */
#endif

/* ------------------------------------------------------------------ */

#if (CLIENT_OS == OS_IRIX)
  #include <unistd.h>
  #include <sys/types.h>
  #include <sys/prctl.h>
  #include <sys/schedctl.h>
  #include <fcntl.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_HPUX)
  #include <unistd.h>
  #include <sys/types.h>
  #include <fcntl.h>
  #include <sys/param.h>
  #include <sys/pstat.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
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
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
  #if defined(OS2_PM)
    #include "platforms/os2gui/os2cons.h"
  #endif
  #ifndef QSV_NUMPROCESSORS       /* This is only defined in the SMP toolkit */
    #define QSV_NUMPROCESSORS     26
  #endif
#elif (CLIENT_OS == OS_AMIGAOS)
  #include "platforms/amiga/amiga.h"
  #include <sys/unistd.h>
  #include <fcntl.h>
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <sys/fcntl.h>
  //#include <unistd.h>
  #include <stdarg.h>
  #include <machine/endian.h>
  #include <sys/time.h>
  #include <socklib.h>
  #include <inetlib.h>
  #include <unixlib.h>
  #include <sys/ioctl.h>
  #include <netdb.h>
  #include <swis.h>
  #include <riscos_sup.h>
  extern int getch();
  #define fileno(f) ((f)->__file)
  #define isatty(f) ((f) == 0)
  #undef time
  #define time riscos_utcbase_time
  #undef gmtime
  #define gmtime riscos_utcbase_gmtime
  #undef localtime
  #define localtime riscos_utcbase_localtime
  #define tzset() /* nothing */
  #define S_IRUSR S_IREAD
  #define S_IWUSR S_IWRITE
  #define S_IRGRP 0
  #define S_IWGRP 0
  } /* extern "C" */
  extern s32 guiriscos, guirestart;
  extern int riscos_in_taskwindow;
#elif (CLIENT_OS == OS_VMS)
  #include <fcntl.h>
  #include <types.h>
  #define unlink remove
  #ifdef __VMS_UCX__
    #include <netinet/in.h> //ntohl/htonl/ntohs/htons
  #elif defined(MULTINET)
    #include "multinet_root:[multinet.include.netinet]in.h"
  #endif
#elif (CLIENT_OS == OS_SCO)
  #include <fcntl.h>
  #include <sys/time.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  #include <windows.h>
  #include <sys/timeb.h>
  #include <process.h>
  #include <conio.h>
  #include <share.h>
  #include <fcntl.h>
  #include <io.h>
  #ifdef __BORLANDC__
    #include <utime.h>
  #else
    #include <sys/utime.h>
  #endif
  #include <fcntl.h>
  #ifndef SH_DENYNO
    #include <share.h>
  #endif
  #include "w32sock.h"      //ntohl/htonl/ntohs/htons/timeval
  #include "w32util.h"
  #include "w32svc.h"       // service
  #include "w32cons.h"      // console
  #include "w32pre.h"       // prelude
  #if defined(_MSC_VER)
    // msc equivalents of file perm flags
    #define R_OK 04
    #define W_OK 02
    #define S_IRUSR _S_IREAD
    #define S_IWUSR _S_IWRITE
    #define S_IRGRP _S_IREAD
    #define S_IWGRP _S_IWRITE
  #elif defined(__BORLANDC__)
    #define R_OK 04
    #define W_OK 02
    //#define S_IRUSR S_IREAD
    //#define S_IWUSR S_IWRITE
    #define S_IRGRP S_IREAD
    #define S_IWGRP S_IWRITE
  #endif
  #ifndef MAX_PATH
    #define MAX_PATH 256
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
  #include <sys/time.h>  // timeval
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_NETWARE)
  #include <sys/time.h> //timeval
  #include <unistd.h> //isatty, chdir, getcwd, access, unlink, chsize, O_...
  #include <conio.h> //ConsolePrintf(), clrscr()
  #include <share.h> //SH_DENYNO
  #include <nwfile.h> //sopen()
  #include <fcntl.h> //O_... constants
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
  #include "platforms/netware/netware.h" //for stuff in netware.cpp
#elif (CLIENT_OS == OS_SUNOS)
  #include <fcntl.h>
  #include <unistd.h>
  #include <stdlib.h>
  #include <stdio.h>
  #define SEEK_SET 0
  #define SEEK_CUR 1
  #define SEEK_END 2
  #include <poll.h>
  #include <sys/time.h>
  #include <sys/socket.h>
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int);
#elif (CLIENT_OS == OS_SOLARIS)
  #include <fcntl.h>
  #include <unistd.h>
  #include <poll.h>
  #include <thread.h>
  extern "C" int nice(int);
  extern "C" int gethostname(char *, int);
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_AIX)
  #include <unistd.h>   // nice()
  #include <strings.h>    // bzero(), strcase...,
  #include <sys/select.h> // fd_set on AIX 4.1
  // clock_gettime is called getclock (used in clitime.cpp)
  #define clock_gettime(a,b) (getclock(a,b))
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_LINUX)
  #include <sys/time.h>
  #include <sys/file.h>
  #include <unistd.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
  #undef NULL    /* some broken header unconditionally */
  #define NULL 0 /* defines NULL to be ((void *)0) */
  #if defined(_MIT_POSIX_THREADS)
    #define sched_yield() pthread_yield()
  #elif defined(HAVE_KTHREADS)
    extern "C" int kthread_join( long );
    extern "C" long kthread_create( void (*)(void *), int , void * );
    extern "C" int kthread_yield(void);
  #elif defined(__ELF__) && !defined(_LINUX_SCHED_H)
    #include <sched.h>
  #endif
  extern "C" int linux_uninstall(const char *basename, int quietly);
  extern "C" int linux_install(const char *basename, int argc,
    const char *argv[], int quietly); /* argv[1..(argc-1)] as start options */
#elif (CLIENT_OS == OS_MACOS)
  #include "client_defs.h"
  #include <Gestalt.h>
  #define CLOCK_MONOTONIC 3
  extern "C" void tzset(void);
  extern "C" int clock_gettime(int clktype, struct timespec *tsp);
  #include <unistd.h>
  #define fileno(f) ((f)->handle)
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_MACOSX)
  //rhapsody is mach 2.x based and altivec unsafe
  #include <sys/time.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_FREEBSD)
  #include <sys/time.h>
  #include <unistd.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #if defined(__FreeBSD__) && (__FreeBSD__ < 3)
    #include <sys/unistd.h>
  #else
    #include <sched.h>
    #include <sys/rtprio.h>
  #endif
#elif (CLIENT_OS == OS_OPENBSD)
  #include <sys/time.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
#elif (CLIENT_OS == OS_BSDOS)
  #include <sys/time.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
  #include <sched.h>
#elif (CLIENT_OS == OS_NETBSD)
  #include <sys/time.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
#elif (CLIENT_OS == OS_QNX)
  #include <sys/time.h>
  #include <sys/select.h>
  #include <process.h>
  #include <env.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_NTO2)
  #include <sys/time.h>
  #include <strings.h>
  #include <unistd.h>
  #include <sched.h>
  #include <sys/syspage.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_DYNIX)
  #include <unistd.h> // sleep(3c)
  struct timezone { int tz_minuteswest, tz_dsttime; };
  extern "C" int gethostname(char *, int);
  extern "C" int gettimeofday(struct timeval *, struct timezone *);
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_DEC_UNIX)
  #include <unistd.h>
  #include <machine/cpuconf.h>
  #include <sys/time.h>
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#elif (CLIENT_OS == OS_NEXTSTEP)
  #include <bsd/sys/time.h>
  #include <sys/types.h>
  #include <fcntl.h>
  #define       S_IRUSR         0x400           /* read permission, */
  #define       S_IWUSR         0x200           /* write permission, */
  #define       S_IRGRP         0x040           /* read permission, group */
  #define       S_IWGRP         0x020           /* write permission, group */
  #define       CLOCKS_PER_SEC          CLK_TCK
  extern "C" int sleep(unsigned int seconds);
  extern "C" int usleep(unsigned int useconds);
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#endif

#endif /* __BASEINCS_H__ */
