/* 
 * Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * Generic strcmpi() and strncmpi() macros
 * ----------------------------------------------------------------------
*/ 
#ifndef __CMPIDEFS_H__
#define __CMPIDEFS_H__ "@(#)$Id: cmpidefs.h,v 1.20 1999/04/05 13:22:30 cyp Exp $"


#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN16)
  #if defined(__TURBOC__)
  // Borland already knows strcmpi
  // Borland already knows strncmpi
  #elif defined(_MSC_VER)
    #define strcmpi(x,y)  _stricmp(x,y)
    #define strncmpi(x,y,n)  _strnicmp(x,y,n)
  #elif defined(__WATCOMC__)
    // already knows strcmpi
    #define strncmpi(x,y,n)  strnicmp(x,y,n)
  #else
    //nada. Let the compiler generate the error if needed
  #endif
#elif (CLIENT_OS == OS_DOS) && defined(__WATCOMC__)
  // already knows strcmpi
  #define strncmpi(x,y,n)  strnicmp(x,y,n)
#elif (CLIENT_OS == OS_OS2)
  #if defined(__EMX__)
    #define strcmpi(x,y)  _stricmp(x,y)
    #define strncmpi(x,y,n)  _strnicmp(x,y,n)
  #elif defined(__WATCOMC__)
    // already knows strcmpi
    #define strncmpi(x,y,n)  strnicmp(x,y,n)
  #elif defined(__IBMCPP__)
    #define strncmpi(x,y,n)  strnicmp(x,y,n)
  #else
    //nada. Let the compiler generate the error if needed
  #endif
#elif (CLIENT_OS == OS_NETWARE)
  #define strcmpi(x,y)  stricmp(x,y)
  #define strncmpi(x,y,n)  strnicmp(x,y,n)
  // SDK knows strcmpi but not strncmpi
#elif (CLIENT_OS == OS_QNX)
  #define strcmpi(x,y)  strcasecmp(x,y)
  #define strncmpi(x,y,n)  strncasecmp(x,y,n)
#elif (CLIENT_OS == OS_VMS)
  // strcmpi() has no equivalent in DEC C++ 5.0  (not true if based
  // on MS C)  #define NO_STRCASECMP
  #define NO_STRCASECMP
  #define strcmpi(x,y)  strcasecmp(x,y)
  #define strncmpi(x,y,n)  strncasecmp(x,y,n)
#elif (CLIENT_OS == OS_AMIGAOS)
  // SAS/C already knows strcmpi
  // but doesn't know strncmpi, translated to strnicmp
  #define strncmpi(x,y,n) strnicmp(x,y,n)
#elif (CLIENT_OS == OS_RISCOS)
  extern "C" {
  #include <unixlib.h>
  #include <sys/types.h>
  }
  #define strcmpi(x,y)  strcasecmp(x,y)
  #define strncmpi(x,y,n)  strncasecmp(x,y,n)
#elif (CLIENT_OS == OS_SUNOS)
  #include <sys/types.h>
  #if (CLIENT_CPU == CPU_68K)
    #define strcmpi(x,y)  strcasecmp(x,y)
    #define strncmpi(x,y,n)  strncasecmp(x,y,n)
    extern "C" int strcasecmp(char *s1, char *s2); // Keep g++ happy.
    extern "C" int strncasecmp(char *s1, char *s2, size_t); // Keep g++ happy.
  #endif
#elif (CLIENT_OS == OS_MACOS)
  #include <stat.mac.h>
  #include "mac_extras.h"
  #define strcmpi(x,y)  stricmp(x,y)
  #define strncmpi(x,y,n)  strnicmp(x,y,n)
#else
  #if defined(__MVS__)
    #include <strings.h>
  #endif
  #include <unistd.h>
  #define strcmpi(x,y)  strcasecmp(x,y)
  #define strncmpi(x,y,n)  strncasecmp(x,y,n)
  #if (CLIENT_OS == OS_DYNIX)
    extern "C" int strcasecmp(const char *s1, const char *s2);
    extern "C" int strncasecmp(const char *s1, const char *s2, size_t);
  #elif (CLIENT_OS == OS_ULTRIX)
    extern "C" int strcasecmp(const char *s1, const char *s2);
    extern "C" int strncasecmp(const char *s1, const char *s2, int);
  #endif
#endif 

#endif /* __CMPIDEFS_H__ */
