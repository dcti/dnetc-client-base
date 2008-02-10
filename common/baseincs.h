/* -*-C++-*-
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __BASEINCS_H__
#define __BASEINCS_H__ "@(#)$Id: baseincs.h,v 1.90 2008/02/10 00:24:30 kakace Exp $"

#include "cputypes.h"

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
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
#endif

/* ------------------------------------------------------------------ */

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
  #include <sched.h>
#elif (CLIENT_OS == OS_OS2)
  #if defined(__WATCOMC__)
    #include "os2defs.h"
    #include <direct.h>
    #include <dos.h>
  #else
    #include "plat/os2/os2defs.h"
  #endif
  #include <conio.h>            /* for console functions */
  #include <process.h>
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
    #include "plat/os2/os2cons.h"
  #endif
  #ifndef QSV_NUMPROCESSORS       /* This is only defined in the SMP toolkit */
    #define QSV_NUMPROCESSORS     26
  #endif
#elif (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_MORPHOS)
  #include "plat/amigaos/amiga.h"
  #ifdef __amigaos4__
  #include <unistd.h>
  #else
  #include <sys/unistd.h>
  #endif
  #include <fcntl.h>
#elif (CLIENT_OS == OS_RISCOS)
  #include <unixlib/local.h>
  #include <sys/fcntl.h>
  #include <netinet/in.h>
  #include <unistd.h>
  #include <stdarg.h>
  #include <endian.h>
  #include <sys/time.h>
  #include <sys/ioctl.h>
  #include <netdb.h>
  #include <kernel.h>
  #include <swis.h>
  #include <riscos_sup.h>
  extern s32 guiriscos, guirestart;
  extern int riscos_in_taskwindow;
#elif (CLIENT_OS == OS_VMS)
  #include <fcntl.h>
  #include <types.h>
  #include <unistd.h>
  #include <timers.h>
  #include <in.h>
#elif (CLIENT_OS == OS_SCO)
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/time.h>
#elif (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
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
  #include "plat/dos/clidos.h" //gettimeofday(), usleep() etc
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
#elif (CLIENT_OS == OS_NETWARE)
  #include <sys/time.h> //timeval
  #include <unistd.h> //isatty, chdir, getcwd, access, unlink, chsize, O_...
  #include <conio.h> //ConsolePrintf(), clrscr()
  #include <share.h> //SH_DENYNO
  #include <nwfile.h> //sopen()
  #include <fcntl.h> //O_... constants
  #include <netinet/in.h> //ntohl/htonl/ntohs/htons
  #include "plat/netware/netware.h" //for stuff in netware.cpp
#elif (CLIENT_OS == OS_NETWARE6)
  #include <sys/time.h>
  #include <sys/times.h>
  #include <sys/file.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/byteorder.h>
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
  //extern "C" int gethostname(char *, int);
#elif (CLIENT_OS == OS_AIX)
  #include <unistd.h>   // nice()
  #include <fcntl.h> /* O_RDWR etc */
  #include <strings.h>    // bzero(), strcase...,
  #include <sys/select.h> // fd_set on AIX 4.1
  // clock_gettime is called getclock (used in clitime.cpp)
  #include <sys/timers.h> /* int getclock */ 
  #define clock_gettime(a,b) (getclock(a,b))
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_PS2LINUX)
  #include <sys/time.h>
  #include <sys/file.h>
  #include <unistd.h>
  #include <fcntl.h> /* O_RDWR etc */
  #include <sys/dir.h> /*scandir*/
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
#elif (CLIENT_OS == OS_MACOSX)
  #include <sys/time.h>
  #include <sys/vmparam.h> //USRSTACK
  #include <sys/sysctl.h>
  #include <unistd.h>
  #include <fcntl.h> /* O_RDWR etc */
#elif (CLIENT_OS == OS_FREEBSD)
  #include <sys/time.h>
  #include <unistd.h>
  #include <fcntl.h> /* O_RDWR etc */
  #if defined (__FreeBSD__) && (__FreeBSD__ < 5)
    #include <sys/param.h>
  #endif
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
  #include <fcntl.h> /* O_RDWR etc */
#elif (CLIENT_OS == OS_BSDOS)
  #include <sys/time.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
  #include <sched.h>
  #include <fcntl.h> /* O_RDWR etc */
#elif (CLIENT_OS == OS_NETBSD)
  #include <sys/time.h>
  #include <sys/param.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
  #include <fcntl.h> /* O_RDWR etc */
#elif (CLIENT_OS == OS_QNX)
  #include <sys/time.h>
  #if defined(__QNXNTO__) /* neutrino */
  #include <sched.h>
  #include <sys/syspage.h>
  #else
  #include <ioctl.h>
  #include <unix.h>
  #include <sys/sched.h>
  #include <sys/select.h>
  #include <process.h>
  #include <env.h>
  #endif
  #include <fcntl.h> /* O_RDWR etc */
#elif (CLIENT_OS == OS_DYNIX)
  #include <unistd.h> // sleep(3c)
  #include <fcntl.h> /* O_RDWR etc */
#elif (CLIENT_OS == OS_DEC_UNIX)
  #include <unistd.h>
  #include <machine/cpuconf.h>
  #include <sys/time.h>
  #include <fcntl.h> /* O_RDWR etc */
  #include <machine/endian.h>
#elif (CLIENT_OS == OS_NEXTSTEP)
  #include <mach/mach.h>  /* host_self, host_kernel_version */
  #include <libc.h>       /* access, geteuid, ... */
  #include <next_sup.h>   /* strdup */

  /* defaults in header are (void (*)())0/1 which make gcc complain */
  #undef  SIG_DFL
  #undef  SIG_IGN
  #define SIG_DFL (void (*)(int))0
  #define SIG_IGN (void (*)(int))1

  #define setsid() setpgrp(0, getpid())         /* cmdline.cpp */

  /* the following are present in NeXTstep but not defined in system
  ** headers or for some reason marked as POSIX-subsystem only */
  #define       S_IRUSR         0x400           /* read permission, */
  #define       S_IWUSR         0x200           /* write permission, */
  #define       S_IRGRP         0x040           /* read permission, group */
  #define       S_IWGRP         0x020           /* write permission, group */

  typedef int pid_t;

  extern "C" int sleep(unsigned int seconds);
  extern "C" void tzset(void);
  extern "C" int getppid(void);            /* triggers.cpp */
  extern "C" int syscall(int number, ...); /* for uname */

  #define SYS_uname       182
  #define _SYS_NAMELEN    32
  struct utsname {
    char sysname[_SYS_NAMELEN];  /* Name of OS */
    char nodename[_SYS_NAMELEN]; /* Name of this node */
    char release[_SYS_NAMELEN];  /* Release level of */
    char version[_SYS_NAMELEN];  /* Version level of */
    char machine[_SYS_NAMELEN];  /* Hardware name */
  };

  #define uname(x) syscall(SYS_uname, x)   /* client.cpp */
#endif

#endif /* __BASEINCS_H__ */
