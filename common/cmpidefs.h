// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
/*
* Generic strcmpi and strncmpi macros
*/
//  
// $Log: cmpidefs.h,v $
// Revision 1.14  1999/01/19 09:41:32  patrick
//
// added OS2-EMX defines for strcmpi...
// moved AIX include for strings.h to basincs.h
//
// Revision 1.13  1999/01/09 19:31:14  cyp
// synchronized cmpidefs.h in client and proxy source trees
//
// Revision 1.12  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.11  1998/12/22 15:58:24  jcmichot
// QNX port changes
//
// Revision 1.10  1998/12/08 05:38:18  dicamillo
// Add defines for MacOS.
//
// Revision 1.9  1998/09/25 04:29:12  pct
// DEC Ultrix port changes
//
// Revision 1.8  1998/08/27 00:43:28  cyp
// DOS change: strncmpi() now defined.
//
// Revision 1.7  1998/07/16 21:23:03  nordquist
// More DYNIX port changes.
//
// Revision 1.6  1998/07/13 12:40:28  kbracey
// RISC OS update. Added -noquiet option.
//
// Revision 1.5  1998/07/05 07:04:26  jlawson
// changes for Win32s
//
// Revision 1.4  1998/07/01 03:12:45  blast
// AmigaOS changes...
//
// Revision 1.3  1998/06/29 07:58:39  ziggyb
// strncmpi(x,y,n) doesn't seem to be defined in my older version of watcom.
//
// Revision 1.2  1998/06/26 10:02:49  cyruspatel
// Fixed cmpidefs.h to be platform-centric instead of compiler-centric.
//
// Revision 1.1  1998/06/26 07:11:55  daa
// move macro defs for strcmpi and strncmpi to a seperate header file
//

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
  #if (CLIENT_OS == OS_AIX) || defined(__MVS__)
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
