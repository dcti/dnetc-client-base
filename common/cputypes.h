/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 

#ifndef _CPUTYPES_H_
#define _CPUTYPES_H_ "@(#)$Id: cputypes.h,v 1.56 1999/04/05 18:15:20 cyp Exp $"

/* ----------------------------------------------------------------- */

#if !defined(INTSIZES)
#define INTSIZES 442
#endif

#if (INTSIZES == 422)       // (16-bit DOS/WIN):  long=32, int=16, short=16
  typedef unsigned long u32;
  typedef signed long s32;
  typedef unsigned short u16;
  typedef signed short s16;
#elif (INTSIZES == 442)     // (typical):  long=32, int=32, short=16
  typedef unsigned long u32;
  typedef signed long s32;
  typedef unsigned short u16;
  typedef signed short s16;
#elif (INTSIZES == 842)     // (Alphas and 64bit MIPS): long=64, int=32, short=16
  typedef unsigned int u32;
  typedef signed int s32;
  typedef unsigned short u16;
  typedef signed short s16;
#else
  #error "Invalid INTSIZES"
#endif

typedef unsigned char u8;
typedef signed char s8;
typedef double f64;
typedef float f32;

struct fake_u64 { u32 hi, lo; };
struct fake_s64 { s32 hi, lo; };

typedef struct fake_u64 u64;
typedef struct fake_s64 s64;

struct u128 { u64 hi, lo; };
struct s128 { s64 hi, lo; };

/* ----------------------------------------------------------------- */

// Major CPU architectures, we don't need (or want) very fine resolution
#define CPU_UNKNOWN     0
#define CPU_X86         1
#define CPU_POWERPC     2
#define CPU_MIPS        3
#define CPU_ALPHA       4
#define CPU_PA_RISC     5
#define CPU_68K         6
#define CPU_SPARC       7
#define CPU_JAVA_VM     8
#define CPU_POWER       9
#define CPU_VAX         10
#define CPU_ARM         11
#define CPU_88K         12
#define CPU_KSR1        13
#define CPU_S390        14
#define CPU_MASPAR      15
#define CPU_DESCRACKER  16  // eff descracker

// Major OS Architectures.
#define OS_UNKNOWN      0
#define OS_WIN32        1  // win95 + win98 + winnt
#define OS_DOS          2  // ms-dos, pc-dos, dr-dos, etc.
#define OS_FREEBSD      3
#define OS_LINUX        4
#define OS_BEOS         5
#define OS_MACOS        6
#define OS_IRIX         7
#define OS_VMS          8
#define OS_DEC_UNIX     9
#define OS_UNIXWARE     10
#define OS_OS2          11
#define OS_HPUX         12
#define OS_NETBSD       13
#define OS_SUNOS        14
#define OS_SOLARIS      15
#define OS_OS9          16
#define OS_JAVA_VM      17
#define OS_BSDI         18
#define OS_NEXTSTEP     19
#define OS_SCO          20
#define OS_QNX          21
#define OS_OSF1         22    // oldname for DEC UNIX
#define OS_MINIX        23
#define OS_MACH10       24
#define OS_AIX          25
#define OS_AUX          26
#define OS_RHAPSODY     27
#define OS_AMIGAOS      28
#define OS_OPENBSD      29
#define OS_NETWARE      30
#define OS_MVS          31
#define OS_ULTRIX       32
#define OS_NEWTON       33
#define OS_RISCOS       34
#define OS_DGUX         35
#define OS_WIN32S       36    // windows 3.1, 3.11, wfw (32-bit Win32s)
#define OS_SINIX        37
#define OS_DYNIX        38
#define OS_OS390        39
#define OS_MASPAR       40
#define OS_WIN16        41    // windows 3.1, 3.11, wfw (16-bit)
#define OS_DESCRACKER   42    // eff des cracker

/* ----------------------------------------------------------------- */

// determine current compiling platform
#if defined(WIN32) || defined(__WIN32__) || defined(_Windows) || defined(_WIN32)
  
  #define CLIENT_OS_NAME "Win32"
  #if defined(NTALPHA)
    #define CLIENT_OS     OS_WIN32
    #define CLIENT_CPU    CPU_ALPHA
  #elif defined(ASM_PPC)
    #define CLIENT_OS     OS_WIN32
    #define CLIENT_CPU    CPU_POWERPC
  #elif !defined(WIN32) && !defined(__WIN32__) && !defined(_WIN32)
    // win16 
    #define CLIENT_OS     OS_WIN16
    #define CLIENT_CPU    CPU_X86
    #undef CLIENT_OS_NAME
    #define CLIENT_OS_NAME "Win16"
  #elif defined(__WIN32S__) /* may need to be defined in makefile */
    // win32s gui
    #define CLIENT_OS     OS_WIN32S
    #define CLIENT_CPU    CPU_X86
  #elif defined(_M_IX86)
    #define CLIENT_OS     OS_WIN32
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(DJGPP) || defined(DOS4G) || defined(__MSDOS__)
  #define CLIENT_OS     OS_DOS
  #define CLIENT_CPU    CPU_X86
  #define CLIENT_OS_NAME "x86 DOS"
#elif defined(__NETWARE__)
  #define CLIENT_OS_NAME "NetWare"
  #if defined(_M_IX86)
    #define CLIENT_OS     OS_NETWARE
    #define CLIENT_CPU    CPU_X86
  #elif defined(_M_SPARC)
    #define CLIENT_OS     OS_NETWARE
    #define CLIENT_CPU    CPU_SPARC
  #elif defined(_M_ALPHA)
    #define CLIENT_OS     OS_NETWARE
    #define CLIENT_CPU    CPU_ALPHA
  #endif
#elif defined(__EMX__) || defined(__OS2__)
  #define CLIENT_OS_NAME "OS/2"
  #define CLIENT_OS     OS_OS2
  #define CLIENT_CPU    CPU_X86
#elif defined(linux)
  #define CLIENT_OS_NAME "Linux"
  #if defined(__alpha__) || defined(ASM_ALPHA)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_ALPHA
  #elif defined(__i386__) || defined(ASM_X86)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_X86
  #elif defined(ARM)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_ARM
  #elif defined(__sparc__) || defined(ASM_SPARC)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_SPARC
  #elif defined(ASM_PPC)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_POWERPC
  #elif defined(ASM_68K)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_68K
  #endif
#elif defined(__FreeBSD__)
  #define CLIENT_OS_NAME "FreeBSD"
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_OS     OS_FREEBSD
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(__NetBSD__)
  #define CLIENT_OS_NAME  "NetBSD"
  #define CLIENT_OS       OS_NETBSD
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #elif defined(ARM)
    #define CLIENT_CPU    CPU_ARM
  #elif defined(__alpha__) || defined(ASM_ALPHA)
    #define CLIENT_CPU    CPU_ALPHA
  #elif defined(__vax__) || defined(ASM_VAX)
    #define CLIENT_CPU    CPU_VAX
  #endif
#elif defined(__OpenBSD__) || defined(openbsd)
  #define CLIENT_OS_NAME  "OpenBSD"
  #define CLIENT_OS       OS_OPENBSD
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #elif defined(__alpha__) || defined(ASM_ALPHA)
    #define CLIENT_CPU    CPU_ALPHA
  #elif defined(__sparc__)
    #define CLIENT_CPU    CPU_SPARC
  #endif
#elif defined(__QNX__)
  #define CLIENT_OS_NAME  "QNX"
  #define CLIENT_OS       OS_QNX
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(solaris) || defined(sun)
  #define CLIENT_OS_NAME  "Solaris"
  #define CLIENT_OS       OS_SOLARIS
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #elif defined(__sparc) || defined(ASM_SPARC)
    #define CLIENT_CPU    CPU_SPARC
  #endif
#elif defined(_SUN68K_)
  #define CLIENT_OS_NAME  "SunOS"
  #define CLIENT_OS       OS_SUNOS
  #define CLIENT_CPU      CPU_68K
#elif defined(bsdi)
  #define CLIENT_OS_NAME  "BSD/OS"
  #define CLIENT_OS       OS_BSDI
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(sco5)
  #define CLIENT_OS_NAME  "SCO Unix"
  #define CLIENT_OS       OS_SCO
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(__osf__)
  #define CLIENT_OS_NAME  "DEC Unix"
  #define CLIENT_OS       OS_DEC_UNIX
  #if defined(__alpha)
    #define CLIENT_CPU    CPU_ALPHA
  #endif
#elif defined(sinix)
  #define CLIENT_OS_NAME  "Sinix"
  #define CLIENT_OS       OS_SINIX
  #if defined(ASM_MIPS) || defined(__mips)
    #define CLIENT_CPU    CPU_MIPS
  #endif
#elif defined(ultrix)
  #define CLIENT_OS_NAME  "Ultrix"
  #define CLIENT_OS       OS_ULTRIX
  #if defined(ASM_MIPS) || defined(__mips)
    #define CLIENT_CPU    CPU_MIPS
  #endif
#elif (defined(ASM_MIPS) || defined(__mips))
  #define CLIENT_OS_NAME  "Irix"
  #define CLIENT_OS       OS_IRIX
  #define CLIENT_CPU    CPU_MIPS
#elif defined(__VMS)
  #define CLIENT_OS_NAME  "VMS"
  #define CLIENT_OS       OS_VMS
  #if defined(__ALPHA)
    #define CLIENT_CPU    CPU_ALPHA
  #endif

  #error NONETWORK define is obsolete. (see top of [high up in] network.cpp)

  #if !defined(__VMS_UCX__) && !defined(NONETWORK) && !defined(MULTINET)
    #define MULTINET 1
  #endif

#elif defined(_HPUX) || defined(__hpux) || defined(__hpux__)
  #define CLIENT_OS_NAME  "HP/UX"
  #define CLIENT_OS       OS_HPUX
  #if defined(__hppa) || defined(__hppa__) || defined(ASM_HPPA)
    #define CLIENT_CPU    CPU_PA_RISC
  #elif defined(_HPUX_M68K)
    #define CLIENT_CPU    CPU_68K
  #endif
#elif defined(_DGUX)
  #define CLIENT_OS_NAME  "DG/UX"
  #define CLIENT_OS       OS_DGUX
  #define CLIENT_CPU      CPU_88K
#elif defined(_AIX)
  #define CLIENT_OS_NAME   "AIX"
  #if (defined(_ARCH_PPC) || defined(ASM_PPC))
    #define CLIENT_OS     OS_AIX
    #define CLIENT_CPU    CPU_POWERPC
  #elif (defined(_ARCH_PWR) || defined(_ARCH_PWR2) || defined(ASM_POWER))
    #define CLIENT_OS     OS_AIX
    #define CLIENT_CPU    CPU_POWER
  #endif
#elif defined(macintosh)
  #define CLIENT_OS_NAME   "MacOS"
  #if GENERATINGPOWERPC
    #define CLIENT_OS     OS_MACOS
    #define CLIENT_CPU    CPU_POWERPC
  #elif GENERATING68K
    #define CLIENT_OS     OS_MACOS
    #define CLIENT_CPU    CPU_68K
  #endif
#elif defined(__dest_os) && defined(__be_os) && (__dest_os == __be_os)
  #define CLIENT_OS_NAME   "BeOS"
  #define CLIENT_OS     OS_BEOS
  #if defined(__POWERPC__)
    #define CLIENT_CPU    CPU_POWERPC
  #endif
#elif defined(__BEOS__) && (__BEOS__ == 1)
  #define CLIENT_OS_NAME   "BeOS"
  #define CLIENT_OS     OS_BEOS
  #if defined(__INTEL__) && (__INTEL__ == 1)
    #define CLIENT_CPU CPU_X86
  #endif
#elif defined(AMIGA)
  #define CLIENT_OS_NAME   "AmigaOS"
  #define CLIENT_OS     OS_AMIGAOS
  #ifdef __PPC__
    #define CLIENT_CPU    CPU_POWERPC
  #else
    #define CLIENT_CPU    CPU_68K
  #endif
#elif defined(__riscos)
  #define CLIENT_OS_NAME   "RISC OS"
  #define CLIENT_OS     OS_RISCOS
  #define CLIENT_CPU    CPU_ARM
#elif defined(_NeXT_)
  #define CLIENT_OS_NAME   "NextStep"
  #if defined(ASM_X86)
    #define CLIENT_OS     OS_NEXTSTEP
    #define CLIENT_CPU    CPU_X86
  #elif defined(ASM_68K)
    #define CLIENT_OS     OS_NEXTSTEP
    #define CLIENT_CPU    CPU_68K
  #elif defined(ASM_HPPA)
    #define CLIENT_OS     OS_NEXTSTEP
    #define CLIENT_CPU    CPU_PA_RISC
  #elif defined(ASM_SPARC)
    #define CLIENT_OS     OS_NEXTSTEP
    #define CLIENT_CPU    CPU_SPARC
  #endif
#elif defined(__MVS__)
  #define CLIENT_OS_NAME   "OS390"
  #define CLIENT_OS     OS_OS390
  #define CLIENT_CPU    CPU_S390
#elif defined(_SEQUENT_)
  #define CLIENT_OS     OS_DYNIX
  #define CLIENT_OS_NAME   "Dynix"
  #if defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #endif
#endif

#if !defined(CLIENT_OS)
  #define CLIENT_OS     OS_UNKNOWN
#endif
#if !defined(CLIENT_OS_NAME)
  #define CLIENT_OS_NAME "**Unknown OS**"
#endif  
#if !defined(CLIENT_CPU)
  #define CLIENT_CPU    CPU_UNKNOWN
#endif
#if (CLIENT_OS == OS_UNKNOWN) || (CLIENT_CPU == CPU_UNKNOWN)
  // ignoreunknowncpuos is used by the client's testplat.cpp utility.
  #if !defined(IGNOREUNKNOWNCPUOS)
    #error "Unknown CPU/OS detected in cputypes.h"
  #endif
#endif

/* ----------------------------------------------------------------- */

#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_88K) || \
     (CLIENT_CPU == CPU_SPARC) || (CLIENT_CPU == CPU_POWERPC) || \
     (CLIENT_CPU == CPU_MIPS) || (CLIENT_CPU == CPU_ARM) || \
     ((CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)))
   #define CORES_SUPPORT_SMP
#endif   

#if (CLIENT_OS == OS_WIN32)
  #include <process.h>
  typedef unsigned long THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_OS2)
  #if defined(__EMX__)
    #include <stdlib.h>
  #else
    //Headers defined elsewhere in a separate file.
    typedef long THREADID;
  #endif
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_NETWARE)
  #include <process.h>
  typedef long THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_BEOS)
  #include <OS.h>
  typedef thread_id THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_MACOS)
  #include <Multiprocessing.h>
  typedef MPTaskID THREADID;
  #define OS_SUPPORTS_SMP
#elif defined(MULTITHREAD)
  #include <pthread.h>
  typedef pthread_t THREADID;
  #define OS_SUPPORTS_SMP
  //egcs always includes pthreads.h, so use something other than PTHREAD_H 
  #define _POSIX_THREADS_SUPPORTED

  #if (CLIENT_OS == OS_DGUX)
    #define PTHREAD_SCOPE_SYSTEM PTHREAD_SCOPE_GLOBAL
    #define pthread_sigmask(a,b,c)
  #endif
#else 
  typedef int THREADID;
#endif

// Fix up MULTITHREAD to mean "SMP aware and thread safe"
#if (defined(CORES_SUPPORT_SMP) && defined(OS_SUPPORTS_SMP))
   #define CLIENT_SUPPORTS_SMP
#endif  
#undef MULTITHREAD //undef it to avoid 'unsafe' meaning

/* ----------------------------------------------------------------- */

// Some compilers/platforms don't yet support bool internally.
// When creating new rules here, please try to use compiler-specific
// macro tests since not all compilers on a specific platform (or even
// a newer version of your own compiler) may be missing bool.
//
#if defined(__VMS) || defined(__SUNPRO_CC) || defined(__DECCXX) || defined(__MVS__)
  #define NEED_FAKE_BOOL
#elif defined(_HPUX) || defined(_OLD_NEXT_)
  #define NEED_FAKE_BOOL
#elif defined(__IBMCPP__)
  #define NEED_FAKE_BOOL
#elif defined(__WATCOMC__)
  //nothing - bool is defined
#elif defined(__xlc) || defined(__xlC) || defined(__xlC__) || defined(__XLC121__)
  #define NEED_FAKE_BOOL
#elif (defined(__mips) && __mips < 3 && !defined(__GNUC__))
  #define NEED_FAKE_BOOL
#elif (defined(__TURBOC__) && __TURBOC__ <= 0x400)
  #define NEED_FAKE_BOOL
#elif (defined(_MSC_VER) && _MSC_VER < 1100)
  #define NEED_FAKE_BOOL
#elif (defined(_SEQUENT_) && !defined(__GNUC__))
  #define NEED_FAKE_BOOL
#endif

#if defined(NEED_FAKE_BOOL)
    typedef int bool;
    #define true 1
    #define false 0
#endif

/* ----------------------------------------------------------------- */

#endif /* _CPUTYPES_H_ */

