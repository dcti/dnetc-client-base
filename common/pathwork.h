// Hey, Emacs, this is a -*-C++-*- file !

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: pathwork.h,v $
// Revision 1.4  1998/08/19 17:00:25  cyruspatel
// dos2unix'ified.
//
// Revision 1.3  1998/08/10 20:22:23  cyruspatel
// Moved path related defines from client.h. Created macros so pathwork.cpp
// functions are do-nothing functions if DONT_USE_PATHWORK is defined.
//
// Revision 1.2  1998/07/06 01:47:05  cyruspatel
// Added signature so that emacs recognizes this to be a c++ file.
//
// Revision 1.1  1998/07/05 13:09:09  cyruspatel
// Created - see pathwork.cpp for documentation 
//

#ifndef __PATHWORK_H__
#define __PATHWORK_H__

#ifndef DONT_USE_PATHWORK

// -------------------------------------------------------------------
// Get the working directory previously initialized with 
// InitWorkingDirectoryFromSamplePaths(). returns NULL if buffer is too
// small otherwise returns buffer. The directory returned can be 
// strcat()ed with a filename (ie it always ends with '/' or whatever)

const char *GetWorkingDirectory( char *buffer, unsigned int maxlen );

// -------------------------------------------------------------------
// the working directory is the app's directory, unless the ini filename
// contains a dirspec, in which case the path to the ini file is used.
// Exception to the rule: win32/win16 which always use the app's directory.

int InitWorkingDirectoryFromSamplePaths( const char *inipath, const char *apppath );

// -------------------------------------------------------------------
// prepends the working directory to 'filename' if the filename is
// not already pathified. Returns a pointer to the pathified filename
// or to a zero length string on error (which will cause fopen() to fail)

const char *GetFullPathForFilename( const char *filename );

// -------------------------------------------------------------------

#if (CLIENT_OS == OS_RISCOS)
  #define EXTN_SEP   "/"
#else
  #define EXTN_SEP   "."
#endif

#endif //#ifndef DONT_USE_PATHWORK

// *******************************************************************

#ifdef DONT_USE_PATHWORK

#define GetFullPathForFilename( x ) ( x )
#define InitWorkingDirectoryFromSamplePaths( ip, ap ) //nothing
//#define GetWorkingDirectory( b, z ) 

  #if (CLIENT_OS == OS_NETWARE)
  //#define PATH_SEP   "\\"   //left undefined so I can see
  //#define PATH_SEP_C '\\'   //where the bad references are
  #define EXTN_SEP   "."
  #define EXTN_SEP_C '.'
  #elif ((CLIENT_OS == OS_DOS) || CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_OS2)
  #define PATH_SEP   "\\"
  #define PATH_SEP_C '\\'
  #define ALT_PATH_SEP '/'
  #define ALT_PATH_SEP_C '/'
  #define DRIVE_SEP ':'
  #define DRIVE_SEP_C ':'
  #define EXTN_SEP   "."
  #define EXTN_SEP_C '.'
  #elif (CLIENT_OS == OS_MACOS)
  #define PATH_SEP   ":"
  #define PATH_SEP_C ':'
  #define EXTN_SEP   "."
  #define EXTN_SEP_C '.'
  #elif (CLIENT_OS == OS_RISCOS)
  #define PATH_SEP   "."
  #define PATH_SEP_C '.'
  #define EXTN_SEP   "/"
  #define EXTN_SEP_C '/'
  #else
  #define PATH_SEP   "/"
  #define PATH_SEP_C '/'
  #define EXTN_SEP   "."
  #define EXTN_SEP_C '.'
  #endif
#endif //DONT_USE_PATHWORK

#endif //__PATHWORK_H__
