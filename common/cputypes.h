/* -*-C-*-
 *
 * Copyright distributed.net 1997-2015 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ** header is included by cores, so guard around c++ constructs **
*/

#ifndef __CPUTYPES_H__
#define __CPUTYPES_H__ "@(#)$Id: cputypes.h,v 1.123 2015/06/27 21:52:52 zebe Exp $"

/* ----------------------------------------------------------------- */

/* There are no longer any size limitations storing CLIENT_CPU and CLIENT_OS 
   in the buffer files. So don't recycle any old entries or you will make
   the stats gurus unhappy. */

/* Major CPU architectures, we don't need (or want) very fine resolution */
#define CPU_UNKNOWN     0
#define CPU_X86         1
#define CPU_POWERPC     2
#define CPU_MIPS        3
#define CPU_ALPHA       4
#define CPU_PA_RISC     5
#define CPU_68K         6
#define CPU_SPARC       7
#define CPU_SH4         8  /* was JAVA_VM (never used) */
#define CPU_POWER       9
#define CPU_VAX         10
#define CPU_ARM         11
#define CPU_88K         12 /* DG/UX is no longer supported */
#define CPU_IA64        13 /* was KSR1 */
#define CPU_S390        14
#define CPU_S390X       15 /* was MASPAR (never used) */
#define CPU_DESCRACKER  16 /* eff descracker */
#define CPU_AMD64       17 /* official name */
#define CPU_X86_64      CPU_AMD64 /* old GNU name before AMD announced AMD64 */
#define CPU_CELLBE      18
#define CPU_CUDA        19
#define CPU_ATI_STREAM  20
#define CPU_OPENCL      21
#define CPU_ARM64       22
#define CPU_PPC64       23

/* DO NOT RECYCLE OLD OS SLOTS !!! (including OS_UNUSED_*) */
/* Old OSes will stay in stats forever! */
/* Major OS Architectures. */
#define OS_UNKNOWN      0
#define OS_WIN32        1  /* win95 + win98 + winnt */
#define OS_DOS          2  /* ms-dos, pc-dos, dr-dos, etc. */
#define OS_FREEBSD      3
#define OS_LINUX        4
#define OS_BEOS         5
/* #define OS_MACOS     6 */  /* Obsolete (no longer supported) */
#define OS_IRIX         7
#define OS_VMS          8
#define OS_DEC_UNIX     9
#define OS_UNIXWARE     10
#define OS_OS2          11
#define OS_HPUX         12
#define OS_NETBSD       13
#define OS_SUNOS        14
#define OS_SOLARIS      15
/* #define OS_UNUSED_1  16 */ /* was OS9 (never used) */
/* #define OS_UNUSED_2  17 */ /* was JAVA-VM (never used) */
#define OS_BSDOS        18
#define OS_NEXTSTEP     19
#define OS_SCO          20
#define OS_QNX          21 /* includes QNX Neutrino */
/* #define OS_UNUSED_3  22 */ /* was OSF1, was NTO2 (QNX) */
/* #define OS_UNUSED_4  23 */ /* was MINIX (never used) */
/* #define OS_UNUSED_5  24 */ /* was MACH10 (never used) */
#define OS_AIX          25
/* #define OS_UNUSED_6  26 */ /* was AUX (never used) */
#define OS_MACOSX       27 /* was RHAPSODY. (MACOSX was 43) */
#define OS_AMIGAOS      28
#define OS_OPENBSD      29
#define OS_NETWARE      30
#define OS_MVS          31
#define OS_ULTRIX       32
/* #define OS_UNUSED_8  33 */ /* was NEWTON (never used), was OS400 (never used) */
#define OS_RISCOS       34
#define OS_DGUX         35
/* #define OS_WIN32S    36 */ /* obsolete (32-bit Win32s) w16 client is 32bit, was WIN16 */
#define OS_SINIX        37
#define OS_DYNIX        38
#define OS_OS390        39
/* #define OS_UNUSED_9  40 */ /* was MASPAR (never used) */
#define OS_WIN16        41 /* windows 3.1, 3.11, wfw (was 16bit, now 32bit), was WIN32S */
#define OS_DESCRACKER   42 /* eff des cracker */
/* #define OS_MACOSX    43 */ /* obsolete, is now 27, was PS2LINUX */
#define OS_PS2LINUX     44
#define OS_MORPHOS      45
#define OS_WIN64        46
#define OS_NETWARE6     47
#define OS_DRAGONFLY    48
#define OS_HAIKU        49
#define OS_ANDROID      50
#define OS_IOS          51 /* Apple iOS (iPhone, iPad, etc.) */
/* DO NOT RECYCLE OLD OS SLOTS !!! (including OS_UNUSED_*) */

/* ----------------------------------------------------------------- */

/* determine current compiling platform */
#if defined(_WIN64)
  #define CLIENT_OS        OS_WIN64
  #define CLIENT_OS_NAME   "Win64"
  #if defined(_M_AMD64)
    #define CLIENT_CPU     CPU_AMD64
  #elif defined(_M_IA64)
    #define CLIENT_CPU     CPU_IA64
  #endif
#elif defined(WIN32) || defined(__WIN32__) || defined(_Windows) || defined(_WIN32)
  #define CLIENT_OS        OS_WIN32
  #define CLIENT_OS_NAME   "Win32"
  #if defined(NTALPHA) || defined(_M_ALPHA)
    #define CLIENT_CPU     CPU_ALPHA
  #elif defined(ASM_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined (CUDA)
    #define CLIENT_CPU     CPU_CUDA
  #elif defined(ATI_STREAM)
    #define CLIENT_CPU     CPU_ATI_STREAM
  #elif defined(_M_ARM)
    #define CLIENT_CPU     CPU_ARM
  #elif defined(OPENCL)
    #define CLIENT_CPU     CPU_OPENCL
  #elif (!defined(WIN32) && !defined(__WIN32__) && !defined(_WIN32)) /* win16 */ \
        || (defined(__WINDOWS386__)) /* standard 32bit client built for win16 */
    #define CLIENT_CPU     CPU_X86
    #undef CLIENT_OS
    #undef CLIENT_OS_NAME
    #define CLIENT_OS      OS_WIN16
    #define CLIENT_OS_NAME "Win16"
  #else
    #define CLIENT_CPU     CPU_X86
  #endif
#elif defined(DJGPP) || defined(DOS4G) || defined(__MSDOS__)
  #define CLIENT_OS        OS_DOS
  #define CLIENT_CPU       CPU_X86
  #define CLIENT_OS_NAME   "x86 DOS"
#elif defined(__NETWARE__)
  #define CLIENT_OS_NAME   "NetWare"
  #if defined(_M_IX86)
    #define CLIENT_OS      OS_NETWARE
    #define CLIENT_CPU     CPU_X86
  #elif defined(_M_SPARC)
    #define CLIENT_OS      OS_NETWARE
    #define CLIENT_CPU     CPU_SPARC
  #elif defined(_M_ALPHA)
    #define CLIENT_OS      OS_NETWARE
    #define CLIENT_CPU     CPU_ALPHA
  #endif
#elif defined(_NETWARE6_)
  #define CLIENT_OS     OS_NETWARE6
  #define CLIENT_CPU    CPU_X86
  #define CLIENT_OS_NAME "NetWare 6.x"  
#elif defined(__EMX__) || defined(__OS2__)
  #define CLIENT_OS_NAME   "OS/2"
  #define CLIENT_OS        OS_OS2
  #define CLIENT_CPU       CPU_X86
#elif defined(linux)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #if defined(__ps2linux__)
    #define CLIENT_OS_NAME "PS2 Linux"
    #define CLIENT_OS      OS_PS2LINUX
  #elif defined(__ps3__)
    #define CLIENT_OS_NAME "PS3 Linux"
    #define CLIENT_OS      OS_LINUX
  #else
    #define CLIENT_OS_NAME "Linux"
    #define CLIENT_OS      OS_LINUX
  #endif
  #if defined(CUDA) && (defined(__i386__) || defined(__x86_64__) || defined(__amd64__))
    #define CLIENT_CPU     CPU_CUDA
  #elif defined(ATI_STREAM) && (defined(__i386__) || defined(__x86_64__) || defined(__amd64__))
    #define CLIENT_CPU     CPU_ATI_STREAM
  #elif defined(OPENCL) && (defined(__i386__) || defined(__x86_64__) || defined(__amd64__))
    #define CLIENT_CPU     CPU_OPENCL
  #elif defined(ASM_HPPA) /* cross compile, ergo don't use __hppa/__hppa__ */
    #define CLIENT_CPU     CPU_PA_RISC
  #elif defined(ASM_SH4) /* cross compile, ergo don't use __sh__ */
    #define CLIENT_CPU     CPU_SH4
  #elif defined(ASM_ALPHA) || defined(__alpha__)
    #define CLIENT_CPU     CPU_ALPHA
  #elif defined(ASM_X86) || defined(__i386__)
    #define CLIENT_CPU     CPU_X86
  #elif defined(__S390__) && defined(S390_Z_ARCH)
    #define CLIENT_CPU     CPU_S390   /* like S390 except rotate.h */
    #undef  CLIENT_OS_NAME
    #define CLIENT_OS_NAME "Linux (z/Architecture)"
  #elif defined(__S390__)
    #define CLIENT_CPU     CPU_S390
  #elif defined(__S390X__)
    #define CLIENT_CPU     CPU_S390X
  #elif defined(__IA64__)
    #define CLIENT_CPU     CPU_IA64
  #elif defined(__arm64__) || defined(__aarch64__)
    #define CLIENT_CPU     CPU_ARM64
 #elif defined(__ppc64__) || defined(__PPC64__)
    #define CLIENT_CPU     CPU_PPC64   
#elif defined(ARM) || defined(__arm__)
    #define CLIENT_CPU     CPU_ARM
  #elif defined(ASM_SPARC) || defined(__sparc__)
    #define CLIENT_CPU     CPU_SPARC
  #elif defined(__PPU__) || defined(__SPU__)
    #define CLIENT_CPU     CPU_CELLBE
  #elif defined(ASM_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(ASM_68K)
    #define CLIENT_CPU     CPU_68K
  #elif defined(ASM_MIPS) || defined(__mips)
    #define CLIENT_CPU     CPU_MIPS
  #elif defined(ASM_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #define CLIENT_CPU     CPU_AMD64
  #endif
#elif defined(__FreeBSD__)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "FreeBSD"
  #define CLIENT_OS        OS_FREEBSD
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(__alpha__) || defined(ASM_ALPHA)
    #define CLIENT_CPU     CPU_ALPHA
  #elif defined(__ppc__) || defined(ASM_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(__sparc__) || defined(__sparc_v9__) || defined(ASM_SPARC)
    #define CLIENT_CPU     CPU_SPARC
  #elif defined(__amd64__) || defined(ASM_AMD64)
    #define CLIENT_CPU     CPU_AMD64
  #endif
#elif defined(__NetBSD__)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "NetBSD"
  #define CLIENT_OS        OS_NETBSD
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(__arm64__)
    #define CLIENT_CPU     CPU_ARM64
  #elif defined(__arm32__) || defined(ARM)
    #define CLIENT_CPU     CPU_ARM
  #elif defined(__alpha__) || defined(ASM_ALPHA)
    #define CLIENT_CPU     CPU_ALPHA
  #elif defined(__vax__) || defined(ASM_VAX)
    #define CLIENT_CPU     CPU_VAX
  #elif defined(__m68k__) || defined(ASM_68K)
    #define CLIENT_CPU     CPU_68K
  #elif defined(__mips__) || defined(ASM_MIPS)
    #define CLIENT_CPU     CPU_MIPS
  #elif defined(__powerpc__) || defined(ASM_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(__sparc__) || defined(ASM_SPARC)
    #define CLIENT_CPU     CPU_SPARC
  #elif defined(ASM_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #define CLIENT_CPU     CPU_AMD64
  #endif
#elif defined(__OpenBSD__) || defined(openbsd)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "OpenBSD"
  #define CLIENT_OS        OS_OPENBSD
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(__alpha__) || defined(ASM_ALPHA)
    #define CLIENT_CPU     CPU_ALPHA
  #elif defined(__sparc__)
    #define CLIENT_CPU     CPU_SPARC
  #elif defined(__m68k__) || defined(ASM_68K)
    #define CLIENT_CPU     CPU_68K
  #elif defined(ASM_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #define CLIENT_CPU     CPU_AMD64
  #elif defined(ASM_HPPA) || defined(__hppa__)
    #define CLIENT_CPU     CPU_PA_RISC
  #elif defined(__mips__) || defined(ASM_MIPS)
    #define CLIENT_CPU     CPU_MIPS
  #elif defined(__powerpc__) || defined(ASM_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #endif
#elif defined(bsdi)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "BSD/OS"
  #define CLIENT_OS        OS_BSDOS
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #endif
#elif defined(__DragonFly__)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "DragonFly"
  #define CLIENT_CPU       CPU_X86 /* no other CPU for now, amd64 will come */
  #define CLIENT_OS        OS_DRAGONFLY
#elif defined(__QNX__)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS        OS_QNX
  #if defined(__QNXNTO__)
    #define CLIENT_OS_NAME "QNX6"
  #else
    #define CLIENT_OS_NAME "QNX4"
  #endif
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(ASM_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(ASM_MIPS) || defined(ASM_MIPS)
    #define CLIENT_CPU     CPU_MIPS
  #elif defined(ASM_ARM)
    #define CLIENT_CPU     CPU_ARM
  #elif defined(ASM_SH4) /* cross compile, ergo don't use  __sh__ */
    #define CLIENT_CPU     CPU_SH4
  #endif
#elif defined(solaris) || defined(sun) || defined(_SUN68K_)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #if defined(sunos) || defined(_SUN68K_)
    #define CLIENT_OS_NAME "SunOS"
    #define CLIENT_OS      OS_SUNOS
  #else
    #define CLIENT_OS_NAME "Solaris"
    #define CLIENT_OS      OS_SOLARIS
  #endif
  #if defined(_SUN68K_) || defined(ASM_68K)
    #define CLIENT_CPU     CPU_68K
  #elif defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(__sparc) || defined(ASM_SPARC)
    #define CLIENT_CPU     CPU_SPARC
  #elif defined(ASM_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #define CLIENT_CPU     CPU_AMD64
  #endif
#elif defined(sco5)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "SCO OpenServer"
  #define CLIENT_OS        OS_SCO
  #if defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #endif
#elif defined(__osf__)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "DEC Unix"
  #define CLIENT_OS        OS_DEC_UNIX
  #if defined(__alpha)
    #define CLIENT_CPU     CPU_ALPHA
  #endif
#elif defined(sinix)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "Sinix"
  #define CLIENT_OS        OS_SINIX
  #if defined(ASM_MIPS) || defined(__mips)
    #define CLIENT_CPU     CPU_MIPS
  #endif
#elif defined(ultrix)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "Ultrix"
  #define CLIENT_OS        OS_ULTRIX
  #if defined(ASM_MIPS) || defined(__mips)
    #define CLIENT_CPU     CPU_MIPS
  #endif
#elif defined(IRIX) || defined(Irix) || defined(irix)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  /*
   * Let the Makefile override the presentation name. This is
   * used by the MIPSpro build targets, letting us set the
   * specific platform the build was for (e.g. "Irix (IP19)").
   */
  #ifndef CLIENT_OS_NAME
    #define CLIENT_OS_NAME "Irix"
  #endif /* ! CLIENT_OS_NAME */
  #define CLIENT_OS        OS_IRIX
  #define CLIENT_CPU       CPU_MIPS
#elif defined(__VMS)
  #define CLIENT_OS_NAME   "OpenVMS"
  #define CLIENT_OS        OS_VMS
  #if defined(__ALPHA)
    #define CLIENT_CPU     CPU_ALPHA
  #endif
#elif defined(_HPUX) || defined(__hpux) || defined(__hpux__)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  typedef unsigned long long uint64_t;
  typedef unsigned int uint32_t;
  typedef long long int64_t;
  #define CLIENT_OS_NAME   "HP-UX"
  #define CLIENT_OS        OS_HPUX
  #if defined(__hppa) || defined(__hppa__) || defined(ASM_HPPA)
    #define CLIENT_CPU     CPU_PA_RISC
  #elif defined(_HPUX_M68K)
    #define CLIENT_CPU     CPU_68K
  #endif
#elif defined(_DGUX)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "DG/UX"
  #define CLIENT_OS        OS_DGUX
  #define CLIENT_CPU       CPU_88K
#elif defined(_AIX)
  #ifndef __unix__ /* should already be defined */
  #define __unix__
  #endif
  #define CLIENT_OS_NAME   "AIX"
  #define CLIENT_OS        OS_AIX

  #if defined(_ARCH_PPC)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(_ARCH_PWR) || defined(_ARCH_PWR2)
    #define CLIENT_CPU     CPU_POWER
  #endif

  /* make sure we are only using threads if the compiler supports it,
  ** for egcs, we have to use -mthreads, for gcc > 3.1 use -pthread and
  ** for xlc, use cc_r */
  #if defined(_THREAD_SAFE)
    #define HAVE_POSIX_THREADS
  #endif
#elif defined(__APPLE__)
  #if defined (IOS)
    #define CLIENT_OS_NAME  "iOS"
    #define CLIENT_OS       OS_IOS
  #else
  #define CLIENT_OS       OS_MACOSX
  #if defined(__arm64__)
    #define CLIENT_OS_NAME  "Mac OS X" // Added this incase we want to identify this as macOS 11 in the future.
  #else
    #define CLIENT_OS_NAME  "Mac OS X"
#endif
#endif
  #ifndef __unix__
    #define __unix__
  #endif
  #if defined(CUDA) && (defined(__i386__) || defined(__x86_64__))
    #define CLIENT_CPU    CPU_CUDA
  #elif defined(OPENCL) && (defined(__i386__) || defined(__x86_64__) || defined(__arm64__))
    #define CLIENT_CPU    CPU_OPENCL
  #elif defined(__ppc__) || defined(__ppc64__)
    #define CLIENT_CPU    CPU_POWERPC
  #elif defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU    CPU_X86
  #elif defined(ASM_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #define CLIENT_CPU    CPU_AMD64
  #elif defined(__arm64__)
    #define CLIENT_CPU    CPU_ARM64
  #elif defined(__arm__)
    #define CLIENT_CPU    CPU_ARM
  #endif
#elif defined(__BEOS__) || defined(__be_os)
  #ifndef __unix__ /* 4.4bsd compatible or not? */
  #define __unix__ /* it ain't that special! */
  #endif
  #define CLIENT_OS_NAME   "BeOS"
  #define CLIENT_OS        OS_BEOS
  #if defined(__POWERPC__) || defined(__PPC__)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(__INTEL__)
    #define CLIENT_CPU     CPU_X86
  #endif
#elif defined(__HAIKU__)
  /* Haiku is nearly the same as BeOS, but with better POSIX support */
  #ifndef __unix__
  #define __unix__ /* seems to be good practice */
  #endif
  #define CLIENT_OS_NAME   "Haiku"
  #define CLIENT_OS        OS_HAIKU
  #if defined(__POWERPC__) || defined(__PPC__)
    #define CLIENT_CPU     CPU_POWERPC
  #elif defined(__INTEL__)
    #define CLIENT_CPU     CPU_X86
  #endif
#elif defined(__MORPHOS__)
  #define CLIENT_OS_NAME   "MorphOS"
  #define CLIENT_OS        OS_MORPHOS
  #define CLIENT_CPU       CPU_POWERPC
#elif defined(AMIGA)
  #define CLIENT_OS_NAME   "AmigaOS"
  #define CLIENT_OS        OS_AMIGAOS
  #ifdef __PPC__
    #define CLIENT_CPU     CPU_POWERPC
  #else
    #define CLIENT_CPU     CPU_68K
  #endif
#elif defined(__riscos)
  #define CLIENT_OS_NAME   "RISC OS"
  #define CLIENT_OS        OS_RISCOS
  #define CLIENT_CPU       CPU_ARM
#elif defined(__NeXT__)
  #ifndef __unix__
  #define __unix__ /* just in case */
  #endif
  #define CLIENT_OS_NAME   "NeXTstep"
  #define CLIENT_OS        OS_NEXTSTEP
  #if defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(ASM_68K)
    #define CLIENT_CPU     CPU_68K
  #elif defined(ASM_HPPA)
    #define CLIENT_CPU     CPU_PA_RISC
  #elif defined(ASM_SPARC)
    #define CLIENT_CPU     CPU_SPARC
  #endif
#elif defined(__MVS__)
  #define CLIENT_OS_NAME   "OS/390"
  #define CLIENT_OS        OS_OS390
  #define CLIENT_CPU       CPU_S390
#elif defined(_SEQUENT_)
  #ifndef __unix__
  #define __unix__
  #endif
  #define CLIENT_OS        OS_DYNIX
  #define CLIENT_OS_NAME   "Dynix"
  #if defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #endif
#elif defined(_ANDROID_)
  #define CLIENT_OS_NAME  "Android"
  #if defined(__arm64__)
    #define CLIENT_CPU     CPU_ARM64
  #elif defined(__i386__) || defined(ASM_X86)
    #define CLIENT_CPU     CPU_X86
  #elif defined(__arm32__) || defined(ARM)
    #define CLIENT_CPU     CPU_ARM
  #else
    #define CLIENT_CPU     CPU_ARM
  #endif
  #define CLIENT_OS        OS_ANDROID
#endif

#if !defined(CLIENT_OS)
  #define CLIENT_OS        OS_UNKNOWN
#endif
#if !defined(CLIENT_OS_NAME)
  #define CLIENT_OS_NAME   "**Unknown OS**"
#endif
#if !defined(CLIENT_CPU)
  #define CLIENT_CPU       CPU_UNKNOWN
#endif
#if defined(ASM_NONE)
  #undef CLIENT_CPU
  #define CLIENT_CPU       CPU_UNKNOWN
#elif (CLIENT_OS == OS_UNKNOWN) || (CLIENT_CPU == CPU_UNKNOWN)
  /* ignoreunknowncpuos is used by the client's testplat.cpp utility. */
  #if !defined(IGNOREUNKNOWNCPUOS)
    #error "Unknown CPU/OS detected in cputypes.h"
    #error "fix common/cputypes.h and/or compiler command line -Defines"
  #endif
#endif

#if (CLIENT_CPU == CPU_CUDA)
  #include <cuda.h>
  #if defined(EXPECTED_CUDA_VERSION) && ((EXPECTED_CUDA_VERSION) != (CUDA_VERSION))
    #error "CUDA version mismatch"
  #endif
  #if (CUDA_VERSION == 2000)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 2.0 on " CLIENT_OS_NAME
  #elif (CUDA_VERSION == 2010)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 2.1 on " CLIENT_OS_NAME
  #elif (CUDA_VERSION == 2020)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 2.2 on " CLIENT_OS_NAME
  #elif (CUDA_VERSION == 2030)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 2.3 on " CLIENT_OS_NAME
  #elif (CUDA_VERSION == 3000)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 3.0 on " CLIENT_OS_NAME
  #elif (CUDA_VERSION == 3010)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 3.1 on " CLIENT_OS_NAME
  #elif (CUDA_VERSION == 5050)
    #define CLIENT_OS_NAME_EXTENDED "CUDA 5.5 on " CLIENT_OS_NAME
  #else
    #define CLIENT_OS_NAME_EXTENDED "CUDA on " CLIENT_OS_NAME
  #endif
#endif

#if (CLIENT_CPU == CPU_ATI_STREAM)
  #if (0)
  #else
    #define CLIENT_OS_NAME_EXTENDED "ATI Stream on " CLIENT_OS_NAME
  #endif
#endif

#if (CLIENT_CPU == CPU_OPENCL)
  #if (0)
  #else
    #define CLIENT_OS_NAME_EXTENDED "OpenCL on " CLIENT_OS_NAME
  #endif
#endif

#if !defined(CLIENT_OS_NAME_EXTENDED)
  #define CLIENT_OS_NAME_EXTENDED   CLIENT_OS_NAME
#endif

/* ----------------------------------------------------------------- */

#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64) || \
     (CLIENT_CPU == CPU_68K) || (CLIENT_CPU == CPU_88K) || \
     (CLIENT_CPU == CPU_SPARC) || (CLIENT_CPU == CPU_ARM64) || \
     (CLIENT_CPU == CPU_POWER) || (CLIENT_CPU == CPU_POWERPC) || \
     (CLIENT_CPU == CPU_MIPS) || (CLIENT_CPU == CPU_ARM) || \
     (CLIENT_CPU == CPU_AMD64) || (CLIENT_CPU == CPU_CUDA) || \
     (CLIENT_CPU == CPU_ATI_STREAM) || (CLIENT_CPU == CPU_OPENCL) || \
     ((CLIENT_CPU == CPU_ALPHA) && ((CLIENT_OS == OS_WIN32) || \
     (CLIENT_OS == OS_DEC_UNIX))))
   #define CORES_SUPPORT_SMP
#endif

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
  #include <process.h>
  typedef unsigned long THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_OS2)
  /* Headers defined elsewhere in a separate file. */
  typedef long THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_NETWARE)
  #include <process.h>
  typedef long THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_NETWARE6)
  #define HAVE_POSIX_THREADS
  #define _POSIX_THREADS_SUPPORTED
  #include <pthread.h>
  typedef pthread_t THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_BEOS)
  #include <OS.h>
  typedef thread_id THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_HAIKU)
  /* behave like BeOS for now, but Haiku does support pthreads properly as well */
  #include <OS.h>
  typedef thread_id THREADID;
  #define OS_SUPPORTS_SMP
#elif  ((CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS))
  /* Uses LWP instead of pthreads */
  #include <sched.h>
  #include <thread.h>
  typedef thread_t THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_FREEBSD)
  /* Uses rfork() instead of pthreads */
  typedef int /*pid_t*/ THREADID;
  #define OS_SUPPORTS_SMP
  #include <sys/wait.h>     /* wait() */
  #include <sys/time.h>     /* required for resource.h */
  #include <sys/resource.h> /* WIF*() macros */
  #include <sys/sysctl.h>   /* sysctl()/sysctlbyname() */
  #include <sys/mman.h>     /* minherit() */
#elif (((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_MACOSX) || \
  (CLIENT_OS == OS_IOS) || (CLIENT_OS == OS_ANDROID)) && \
  ((CLIENT_CPU == CPU_CUDA) || (CLIENT_CPU == CPU_ATI_STREAM) || (CLIENT_CPU == CPU_OPENCL)))
  /* Necessary for streams to work correctly */
  #define HAVE_POSIX_THREADS
  #define _POSIX_THREADS_SUPPORTED
  #include <pthread.h>
  typedef pthread_t THREADID;
  #define OS_SUPPORTS_SMP
#elif (CLIENT_OS == OS_LINUX) && \
  !defined(HAVE_POSIX_THREADS) && (CLIENT_CPU == CPU_X86) && 0 /* DISABLED! */
  #define HAVE_KTHREADS /* platforms/linux/li_kthread.c */
  /* uses clone() instead of pthreads ... */
  /* ... but first thread is polled ... */
  #define OS_SUPPORTS_SMP
  typedef long THREADID;
#elif (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_MORPHOS)
  typedef long THREADID;
  #define OS_SUPPORTS_SMP
#elif defined(HAVE_POSIX_THREADS)
  /*
  Q: can't we simply use if defined(_POSIX_THREADS), as this is often defined
     if POSIX threads are supported. Patrick Hildenbrand (patrick@mail4you.de)
  A: yes and no. :)
     no: _POSIX_THREADS may have been defined by an include file, but the
         compiler might generate unsafe code. This is usually made clear with
         the absence/presence of _THREAD_SAFE.
     yes: to keep the number of downloadable binaries to a minimum, clients
          should always be built with thread support (the only reason why two
          binaries might be necessary is when threading depends on a lib that
          is not installed everywhere and a static build is not possible).
          Users can always force non-threadedness by setting numcpu=0.  -cyp
  */
  #include <pthread.h>
  typedef pthread_t THREADID;
  #define OS_SUPPORTS_SMP
  /* egcs always includes pthreads.h, so use something other than PTHREAD_H */
  #define _POSIX_THREADS_SUPPORTED
  #if (CLIENT_OS == OS_DGUX)
    #define PTHREAD_SCOPE_SYSTEM PTHREAD_SCOPE_GLOBAL
    #define pthread_sigmask(a,b,c)
  #elif (CLIENT_OS == OS_LINUX) && defined(_MIT_POSIX_THREADS)
    #define pthread_sigmask(a,b,c) /*no*/
  #elif (CLIENT_OS == OS_AIX)
    #define pthread_sigmask(a,b,c) sigthreadmask(a,b,c)

    /* we need to switch threads to system scheduling scope otherwise
    ** they'll all run on one processor, _POSIX_THREAD_PRIORITY_
    ** SCHEDULING is defined in unistd.h when _AIX_PTHREADS_D7 is
    ** defined, no special libs are involved, xlC_r7 would add this
    ** define but gcc doesn't have it so we do it by hand, actually
    ** we're supposed to link against libpthread_compat.a but it works
    ** fine without it */
    #define _AIX_PTHREADS_D7 1
  #endif
#elif defined(__unix__) && !defined(SINGLE_CRUNCHER_ONLY)
  typedef int /*pid_t*/ THREADID;
  #define OS_SUPPORTS_SMP
  #define HAVE_MULTICRUNCH_VIA_FORK
  #include <sys/wait.h>     /* wait() */
  #include <sys/time.h>     /* required for resource.h */
  #include <sys/resource.h> /* WIF*() macros */
  #if (CLIENT_CPU != CPU_CELLBE)
  #include <sys/mman.h>     /* minherit() */
  #endif
  #ifndef CORES_SUPPORT_SMP 
  #define CORES_SUPPORT_SMP /* no shared data, so cores become smp safe */
  #endif
  #if (CLIENT_CPU == CPU_CUDA)
  #error "FORK and CUDA don't work together - can't use multiple GPUs"
  #elif (CLIENT_CPU == CPU_ATI_STREAM)
  #error "FORK and ATI Stream probably don't work together"
  #elif (CLIENT_CPU == CPU_OPENCL)
  #error "FORK and OpenCL probably don't work together"
  #endif
#else
  typedef int THREADID;
#endif

#if (defined(CORES_SUPPORT_SMP) && defined(OS_SUPPORTS_SMP))
   #define CLIENT_SUPPORTS_SMP
#endif

/* ----------------------------------------------------------------- */

#if defined(PROXYTYPE) /* only for proxy */
  /*
  ** This is only required for proxy builds since the client source is
  ** designed for maximum portability and don't need/use 'bool'
  ** IT IS NOT SUFFICIENT TO 'typedef int bool'!!
  */
  #if (defined(__GNUC__)     && (__GNUC__ < 2)         ) || \
      (defined(__WATCOMC__)  && (__WATCOMC__ < 1100)   ) || \
      (defined(_MSC_VER)     && (_MSC_VER < 1100)      ) || \
      ((defined(__xlc) || defined(__xlC) || \
        defined(__xlC__) || defined(__XLC121__))       ) || \
      (defined(__SUNPRO_CC)                            ) || \
      (defined(__IBMCPP__)                             ) || \
      (defined(__TURBOC__)   && (__TURBOC__ <= 0x400)  )
    /*
     Some compilers don't yet support bool internally.
     *** When creating new rules here, USE COMPILER-SPECIFIC TESTS ***
     (Do not use operating system or cpu type comparisons, since not all
     compilers on a specific platform or even a newer version of your
     own compiler may require the hack).
     *** When creating new rules here, USE COMPILER-SPECIFIC TESTS ***
    */
    #error "To build a proxy you must use a compiler that has intrinsic support for 'bool'"
    /* IT IS NOT SUFFICIENT TO 'typedef int bool'!! */
  #endif
#endif

/* ----------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif
#include <limits.h>
#ifdef __cplusplus
}
#endif

#if !defined(SIZEOF_LONG) || !defined(SIZEOF_SHORT) || !defined(SIZEOF_INT)
  #if (!defined(UINT_MAX) || !defined(ULONG_MAX))
    #error your limits.h appears to be borked (UINT_MAX or ULONG_MAX are undefined)
  #elif (ULONG_MAX < UINT_MAX)
    #error your limits.h is borked. ULONG_MAX can never be less than UINT_MAX
  #else
    #if (!defined(USHRT_MAX) && defined(USHORT_MAX))
      #define USHRT_MAX USHORT_MAX
    #endif
    #if !defined(SIZEOF_SHORT) && defined(USHRT_MAX)
      #if (USHRT_MAX == 0xFF)
        #define SIZEOF_SHORT 1
      #elif (USHRT_MAX == 0xFFFFUL)
        #define SIZEOF_SHORT 2
      #elif (USHRT_MAX == 0xFFFFFFFFUL)
        #define SIZEOF_SHORT 4
      #elif (USHRT_MAX == 0xFFFFFFFFFFFFFFFFUL)
        #define SIZEOF_SHORT 8
      #else
        #error fixme: sizeof(unsigned short) !=1 and !=2 and !=4 and !=8?
      #endif
    #endif
    #if defined(SIZEOF_INT)
      #ifndef SIZEOF_SHORT
        #if (SIZEOF_INT < 4)
          #define SIZEOF_SHORT SIZEOF_INT
        #elif (SIZEOF_INT > 4)
          #define SIZEOF_SHORT (SIZEOF_INT>>1)
        #else
          #define SIZEOF_SHORT 2
        #endif
      #endif
      #ifndef SIZEOF_LONG
        #define SIZEOF_LONG (SIZEOF_INT<<1)
      #endif
    #elif (UINT_MAX == 0xFFUL)
      #ifndef SIZEOF_SHORT
        #define SIZEOF_SHORT 1
      #endif
      #define SIZEOF_INT   1
      #ifndef SIZEOF_LONG
        #if (ULONG_MAX != UINT_MAX)
          #define SIZEOF_LONG  2
        #else
          #define SIZEOF_LONG  1
        #endif
      #endif
    #elif (UINT_MAX == 0xFFFFUL)
      #ifndef SIZEOF_SHORT
        #define SIZEOF_SHORT 2
      #endif
      #define SIZEOF_INT   2
      #ifndef SIZEOF_LONG
        #if (ULONG_MAX != UINT_MAX)
          #define SIZEOF_LONG  4
        #else
          #define SIZEOF_LONG  2
        #endif
      #endif
    #elif (UINT_MAX == 0xFFFFFFFFUL)
      #ifndef SIZEOF_SHORT
        #define SIZEOF_SHORT 2
      #endif
      #define SIZEOF_INT   4
      #ifndef SIZEOF_LONG
        #if (ULONG_MAX != UINT_MAX)
          #define SIZEOF_LONG  8
        #else
          #define SIZEOF_LONG  4
        #endif
      #endif
    #elif (UINT_MAX == 0xFFFFFFFFFFFFFFFFUL)
      #ifndef SIZEOF_SHORT
        #define SIZEOF_SHORT 4
      #endif
      #define SIZEOF_INT   8
      #ifndef SIZEOF_LONG
        #if (ULONG_MAX != UINT_MAX)
          #define SIZEOF_LONG  16
        #else
          #define SIZEOF_LONG  8
        #endif
      #endif
    #else
      #error fixme: sizeof(int) > 8? what would sizeof(short) be?
    #endif
  #endif /* ULONG_MAX >= UINT_MAX */
#endif

#if (defined(SIZEOF_SHORT) && (SIZEOF_SHORT == 2))
  typedef unsigned short u16;
  typedef signed short s16;
#elif (defined(SIZEOF_SHORT) && (SIZEOF_INT == 2))
  typedef unsigned int u16;
  typedef signed int s16;
#elif (defined(SIZEOF_LONG) && (SIZEOF_LONG == 2))
  typedef unsigned long u16;
  typedef signed long s16;
#else
  #error types u16 is undefined (try wchar_t)
#endif
#if (defined(SIZEOF_SHORT) && (SIZEOF_SHORT == 4))
  typedef unsigned short u32;
  typedef signed short s32;
#elif (defined(SIZEOF_INT) && (SIZEOF_INT == 4))
  typedef unsigned int u32;
  typedef signed int s32;
#elif (defined(SIZEOF_LONG) && (SIZEOF_LONG == 4))
  typedef unsigned long u32;
  typedef signed long s32;
#else
  #error types u32/s32 is undefined
#endif
#if (defined(SIZEOF_SHORT) && (SIZEOF_SHORT == 8))
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned short ui64;
  typedef signed short si64;
#elif (defined(SIZEOF_INT) && (SIZEOF_INT == 8))
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned int ui64;
  typedef signed int si64;
#elif (defined(SIZEOF_LONG) && (SIZEOF_LONG == 8))
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned long ui64;
  typedef signed long si64;
#elif defined(__GCC__) || defined(__GNUC__)
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned long long ui64;
  typedef signed long long si64;
#elif (defined(__WATCOMC__) && (__WATCOMC__ >= 1100))
  #if 1/*(CLIENT_OS != OS_NETWARE)*/ /* only if intrinsic (no [vfs]printf etc)*/
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned __int64 ui64;
  typedef __int64 si64;
  #endif
#elif (defined(_MSC_VER) && (_MSC_VER >= 11)) /* VC++ >= 5.0 */
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned __int64 ui64;
  typedef __int64 si64;
#elif defined(__MWERKS__)
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned long long ui64;
  typedef signed long long si64;
#elif defined(__SUNPRO_CC)
  #define HAVE_I64
  #define SIZEOF_LONGLONG 8
  typedef unsigned long long ui64;
  typedef signed long long si64;
#endif

typedef unsigned char u8;

/* ----------------------------------------------------------------- */

#endif /* __CPUTYPES_H__ */
