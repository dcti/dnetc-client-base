#ifndef _AMIGA_VERSION_C
#define _AMIGA_VERSION_C

#include "version.h"
#include "common/cputypes.h"

#if (CLIENT_OS == OS_AMIGAOS)
  #ifdef __SASC
    #if (CLIENT_CPU == CPU_68K)
    const char versionstring[] = "\0$VER: RC5DES_68K " CLIENT_VERSIONSTRING " " __AMIGADATE__;
    #elif (CLIENT_CPU == CPU_POWERPC)
    const char versionstring[] = "\0$VER: RC5DES_PPC " CLIENT_VERSIONSTRING " " __AMIGADATE__;
    #else
    #error "An AmigaOS machine with a different CPU ? Can't be right!"
    #endif // (CLIENT_CPU == CPU_68K)
  #else
    #if (CLIENT_CPU == CPU_68K)
    const char *versionstring = "\0$VER: dnetc_68k " CLIENT_VERSIONSTRING " (" __AMIGADATE__ ")";
    #elif (CLIENT_CPU == CPU_POWERPC)
    const char *versionstring = "\0$VER: dnetc_ppc " CLIENT_VERSIONSTRING " (" __AMIGADATE__ ")";
    #else
    #error "An AmigaOS machine with a different CPU ? Can't be right!"
    #endif // (CLIENT_CPU == CPU_68K)
  #endif // __SASC
#endif // (CLIENT_OS == OS_AMIGAOS)

#endif // _AMIGA_VERSION_C
