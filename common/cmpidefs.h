// Hey, Emacs, this a -*-C++-*- file !
// 
// $Log: cmpidefs.h,v $
// Revision 1.3  1998/06/29 07:58:39  ziggyb
// strncmpi(x,y,n) doesn't seem to be defined in my older version of watcom.
//
// Revision 1.2  1998/06/26 10:02:49  cyruspatel
// Fixed cmpidefs.h to be platform-centric instead of compiler-centric.
//
// Revision 1.1  1998/06/26 07:11:55  daa
// move macro defs for strcmpi and strncmpi to a seperate header file
// 

/*
  Generic strcmpi and strncmpi macros        

  version 1.0 
  by DCTI 
  Copyright 1998 Distributed Computing Technologies Inc. 
*/

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_OS2)
  #if defined(__TURBOC__)
  // Borland already knows strcmpi
  // Borland already knows strncmpi
  #elif defined(_MSC_VER)
    #define strcmpi(x,y)  _stricmp(x,y)
    #define strncmpi(x,y,n)  _strnicmp(x,y,n)
  #elif defined(__WATCOMC__)
    #define strncmpi(x,y,n)  strnicmp(x,y,n)
    // watcom knows both
    // Older version of watcom does do strncmpi
  #else
    //nada. Let the compiler generate the error if needed
  #endif
#elif (CLIENT_OS == OS_DOS) && defined(__WATCOMC__)
  // watcom knows both
#elif (CLIENT_OS == OS_NETWARE)
  #define strcmpi(x,y)  stricmp(x,y)
  #define strncmpi(x,y,n)  strnicmp(x,y,n)
  // SDK knows strcmpi but not strncmpi
#elif (CLIENT_OS == OS_QNX)  
  // already knows strcmpi
  // already knows strncmpi
#elif (CLIENT_OS == OS_VMS)
  // strcmpi() has no equivalent in DEC C++ 5.0  (not true if based 
  // on MS C)  #define NO_STRCASECMP
  #define NO_STRCASECMP
  #define strcmpi(x,y)  strcasecmp(x,y)
  #define strncmpi(x,y,n)  strncasecmp(x,y,n)
#elif (CLIENT_OS == OS_AMIGAOS)
  // SAS/C already knows strcmpi
#elif (CLIENT_OS == OS_RISCOS)
  #include <unixlib.h>
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
#else
  #if (CLIENT_OS == OS_AIX) || defined(__MVS__)
    #include <strings.h>
  #endif
  #include <unistd.h>
  #define strcmpi(x,y)  strcasecmp(x,y)
  #define strncmpi(x,y,n)  strncasecmp(x,y,n)
#endif
