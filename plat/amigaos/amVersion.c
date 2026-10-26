/*
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: amVersion.c,v 1.5 2008/12/30 20:58:44 andreasb Exp $
 *
 * ----------------------------------------------------------------------
 * AmigaOS version string/tag
 * ----------------------------------------------------------------------
*/

#ifndef _AMIGA_VERSION_C
#define _AMIGA_VERSION_C

#include "version.h"
#include "common/cputypes.h"

#if (CLIENT_CPU == CPU_68K)
  const char *versionstring = "\0$VER: dnetc_68k " CLIENT_VERSIONSTRING " (" __AMIGADATE__ ")";
#elif (CLIENT_CPU == CPU_POWERPC)
  #if defined(__amigaos4__) || defined(__MORPHOS__)
    const char *versionstring = "\0$VER: dnetc " CLIENT_VERSIONSTRING " (" __AMIGADATE__ ")";
  #else
    const char *versionstring = "\0$VER: dnetc_ppc " CLIENT_VERSIONSTRING " (" __AMIGADATE__ ")";
  #endif
#else
  #error "An AmigaOS machine with a different CPU ? Can't be right!"
#endif

#endif // _AMIGA_VERSION_C
