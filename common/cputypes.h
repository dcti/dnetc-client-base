// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#if ( !defined(_CPU_32BIT_) && !defined(_CPU_64BIT_) )
#define _CPU_32BIT_
#endif

#ifndef _CPUTYPES_H_
#define _CPUTYPES_H_

typedef unsigned long u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed long s32;
typedef signed short s16;
typedef signed char s8;
typedef double f64;
typedef float f32;

struct fake_u64 { u32 hi, lo; };
struct fake_s64 { s32 hi, lo; };

#if defined(_CPU_32BIT_)
typedef struct fake_u64 u64;
typedef struct fake_s64 s64;
#elif defined(_CPU_64BIT_)
typedef unsigned long long u64;
typedef signed long long s64;
#endif

struct u128 { u64 hi, lo; };
struct s128 { s64 hi, lo; };

// Major CPU architectures, we don't need (or want) very fine resolution
// do not just add numbers here email beberg@distributed.net for an assignment
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
#define CPU_STRONGARM   11
#define CPU_88K         12
#define CPU_KSR1        13
#define CPU_S390        14

// Major OS Architectures, we'll need a port for each
// do not just add numbers here email beberg@distributed.net for an assignment
#define OS_UNKNOWN      0
#define OS_WIN32        1  // 95 + NT + win32
#define OS_DOS          2  // dos (win31 now separate)
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
#define OS_WIN16        36
#define OS_SINIX        37
#define OS_DYNIX        38
#define OS_OS390        39

// determine current compiling platform
#if defined(WIN32) || defined(__WIN32__) || defined(_Windows) || defined(_WIN32)
  #if defined(NTALPHA)
    #define CLIENT_OS     OS_WIN32
    #define CLIENT_CPU    CPU_ALPHA
  #elif defined(ASM_PPC)
    #define CLIENT_OS     OS_WIN32
    #define CLIENT_CPU    CPU_POWERPC
  #elif !defined(WIN32) && !defined(__WIN32__) && !defined(_WIN32)
    #define CLIENT_OS     OS_WIN16
    #define CLIENT_CPU    CPU_X86
  #elif defined(WIN32s)
    #define CLIENT_OS     OS_WIN16
    #define CLIENT_CPU    CPU_X86
  #elif defined(_M_IX86)
    #define CLIENT_OS     OS_WIN32
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(DJGPP) || defined(DOS4G) || defined(__MSDOS__)
  #define CLIENT_OS     OS_DOS
  #define CLIENT_CPU    CPU_X86
#elif defined(__NETWARE__) && defined(_M_IX86)
  #define CLIENT_OS     OS_NETWARE
  #define CLIENT_CPU    CPU_X86
#elif defined(__OS2__)
  #define CLIENT_OS     OS_OS2
  #define CLIENT_CPU    CPU_X86
#elif defined(linux)
  #if defined(ASM_ALPHA)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_ALPHA
  #elif defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_X86
  #elif defined(ARM)
    #define CLIENT_OS     OS_LINUX
    #define CLIENT_CPU    CPU_STRONGARM
  #elif defined(ASM_SPARC) || defined(SPARCLINUX)
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
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_FREEBSD
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(__NetBSD__)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_NETBSD
    #define CLIENT_CPU    CPU_X86
  #elif defined(ARM)
    #define CLIENT_OS     OS_NETBSD
    #define CLIENT_CPU    CPU_STRONGARM
  #endif
#elif defined(__OpenBSD__) || defined(openbsd)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_OPENBSD
    #define CLIENT_CPU    CPU_X86
  #elif defined(ASM_ALPHA)
    #define CLIENT_OS     OS_OPENBSD
    #define CLIENT_CPU    CPU_ALPHA
  #endif
#elif defined(__QNX__)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_QNX
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(solaris)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_SOLARIS
    #define CLIENT_CPU    CPU_X86
  #elif defined(ASM_SPARC)
    #define CLIENT_OS     OS_SOLARIS
    #define CLIENT_CPU    CPU_SPARC
  #endif
#elif defined(_SUN68K_)
  #define CLIENT_OS         OS_SUNOS
  #define CLIENT_CPU        CPU_68K
#elif defined(bsdi)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_BSDI
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(sco5)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
    #define CLIENT_OS     OS_SCO
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(__osf__)
  #if defined(__alpha)
    #define CLIENT_OS     OS_DEC_UNIX
    #define CLIENT_CPU    CPU_ALPHA
  #endif
#elif defined(sinix)
  #if defined(ASM_MIPS) || defined(__mips)
    #define CLIENT_OS     OS_SINIX
    #define CLIENT_CPU    CPU_MIPS
  #endif
#elif (defined(ASM_MIPS) || defined(__mips)) && !defined(sinix)
  #define CLIENT_OS     OS_IRIX
  #define CLIENT_CPU    CPU_MIPS
#elif defined(__VMS)
  #if defined(__ALPHA)
    #define CLIENT_OS     OS_VMS
    #define CLIENT_CPU    CPU_ALPHA
  #endif
  #if !defined(__VMS_UCX__) && !defined(NONETWORK) && !defined(MULTINET)
    #define MULTINET 1
  #endif
#elif defined(_HPUX)
  #if defined(ASM_HPPA)
    #define CLIENT_OS     OS_HPUX
    #define CLIENT_CPU    CPU_PA_RISC
  #endif
#elif defined(_DGUX)
  #define CLIENT_OS     OS_DGUX
  #define CLIENT_CPU    CPU_88K
  #define PTHREAD_SCOPE_SYSTEM PTHREAD_SCOPE_GLOBAL
  #define pthread_sigmask(a,b,c)
#elif defined(_AIX)
  #if defined(_ARCH_PPC)
    #define CLIENT_OS     OS_AIX
    #define CLIENT_CPU    CPU_POWERPC
  #endif
#elif defined(macintosh)
  #if GENERATINGPOWERPC
    #define CLIENT_OS     OS_MACOS
    #define CLIENT_CPU    CPU_POWERPC
  #elif GENERATING68K
    #define CLIENT_OS     OS_MACOS
    #define CLIENT_CPU    CPU_68K
  #endif
#elif defined(__dest_os) && defined(__be_os) && (__dest_os == __be_os)
  #if defined(__POWERPC__)
    #define CLIENT_OS     OS_BEOS
    #define CLIENT_CPU    CPU_POWERPC
  #elif defined(__INTEL__)
    #define CLIENT_OS     OS_BEOS
    #define CLIENT_CPU    CPU_X86
  #endif
#elif defined(AMIGA)
  #define CLIENT_OS     OS_AMIGAOS
  #ifdef __PPC__
    #define CLIENT_CPU    CPU_POWERPC
  #else
    #define CLIENT_CPU    CPU_68K
  #endif
#elif defined(__riscos)
  #define CLIENT_OS     OS_RISCOS
  #define CLIENT_CPU    CPU_STRONGARM
#elif defined(_NeXT_)
  #if defined(ASM_486) || defined(ASM_P5) || defined(ASM_P6) || defined(ASM_K5) || defined(ASM_K6) || defined(ASM_6x86)
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
  #define CLIENT_OS     OS_OS390
  #define CLIENT_CPU    CPU_S390
#endif

#if !defined(CLIENT_OS) || !defined(CLIENT_CPU) || (CLIENT_OS == OS_UNKNOWN) || (CLIENT_CPU == CPU_UNKNOWN)
  #define CLIENT_OS     OS_UNKNOWN
  #define CLIENT_CPU    CPU_UNKNOWN
  #error "Unknown CPU/OS detected in cputypes.h"
#endif

// Some platforms don't yet support bool internally
#if defined(__VMS) || defined(__SUNPRO_CC) || defined(__DECCXX) || defined(__MVS__)
  #define NEED_FAKE_BOOL
#elif defined(_HPUX) || defined(_OLD_NEXT_) 
  #define NEED_FAKE_BOOL
#if (CLIENT_OS == OS_OS2)
  #define NEED_FAKE_BOOL
#elif defined(__xlc) || defined(__xlC) || defined(__xlC__) || defined(__XLC121__)
  #define NEED_FAKE_BOOL
#elif (defined(__mips) && __mips < 3 && !defined(__GNUC__))
  #define NEED_FAKE_BOOL
#elif (defined(__TURBOC__) && __TURBOC__ <= 0x400)
  #define NEED_FAKE_BOOL
#elif (defined(_MSC_VER) && _MSC_VER < 1100)
  #define NEED_FAKE_BOOL
#endif

#if defined(NEED_FAKE_BOOL)
  typedef char bool;
  #define true 1
  #define false 0
#endif


#endif

