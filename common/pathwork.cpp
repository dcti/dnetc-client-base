/*
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This module contains functions for setting the "working directory"
 * and pathifying a filename that has no dirspec. Functions need to be
 * initialized from main() with InitWorkingDirectoryFromSamplePaths();
 * [ ...( inipath, apppath) where inipath should not have been previously
 * merged from argv[0] + default, although it doesn't hurt if it has.]
 *
 * The "working directory" is assumed to be the app's directory, unless the
 * ini filename contains a dirspec, in which case the path to the ini file
 * is used. The exception here is win32/win16 which, for reasons of backward
 * compatability, always use the app's directory.
 *
 * GetFullPathForFilename() is ideally intended for use in (or just prior to)
 * a call to fopen(). This obviates the necessity of having to pre-parse
 * filenames or maintain duplicate filename buffers. In addition, each
 * platform has its own code sections to avoid cross-platform assumptions
 * altogether.
 *
 * --------------------------------------------------------------------
 *
 * $Log: pathwork.cpp,v $
 * Revision 1.13  1999/04/01 03:04:37  cyp
 * Created GetFullPathForFilenameAndDir(). Like GetFullPathForFilename() but
 * can use a directory other than the client's wd.
 *
 * Revision 1.12  1999/01/23 21:29:48  patrick
 * OS2-EMX does not need special care here
 *
 * Revision 1.11  1999/01/08 21:20:38  sugalskd
 * Fixed VMS path-finding code
 *
 * Revision 1.10  1998/10/04 11:35:49  remi
 * Id tags fun.
 *
 * Revision 1.9  1998/08/05 16:49:32  cberry
 * commented out #define DEBUG that KB had put in for some reason.
 *
 * Revision 1.8  1998/07/30 02:11:52  blast
 * AmigaOS update
 *
 * Revision 1.7  1998/07/13 12:40:31  kbracey
 * RISC OS update. Added -noquiet option.
 *
 * Revision 1.6  1998/07/05 23:32:12  cyruspatel
 * Fixed missing semicolon in *nix'ish code section.
 *
 * Revision 1.3  1998/07/05 20:27:02  jlawson
 * headers for win32s and win16 and borland/dos
 *
 * Revision 1.2  1998/07/05 20:13:41  jlawson
 * modified headers for Win32
 *
 * Revision 1.1  1998/07/05 13:09:07  cyruspatel
 * Created - see documentation above. - my apologies to all non-win32 people,
 * for the sake of one platform, we bork the rest. :(
 *
*/

#if (!defined(lint) && defined(__showids__))
const char *pathwork_cpp(void) {
return "@(#)$Id: pathwork.cpp,v 1.13 1999/04/01 03:04:37 cyp Exp $"; }
#endif

#include <stdio.h>
#include <string.h>
#include "cputypes.h"
#include "pathwork.h"

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || (CLIENT_OS == OS_WIN16)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#elif (CLIENT_OS == OS_DOS) ||  ( (CLIENT_OS == OS_OS2) && !defined(__EMX__) )
  #include <dos.h>  //drive functions
  #include <ctype.h> //toupper
  #if defined(__WATCOMC__)
    #include <direct.h> //getcwd
  #elif defined(__TURBOC__)
    #include <dir.h>
  #endif
#elif (CLIENT_OS == OS_RISCOS)
  #include <swis.h>
#endif

#if (CLIENT_OS == OS_RISCOS)
#define MAX_FULLPATH_BUFFER_LENGTH (1024)
#else
#define MAX_FULLPATH_BUFFER_LENGTH (512)
#endif

/* ------------------------------------------------------------------------ */

/* get the offset of the filename component in fully qualified path */
/* previously called IsFilenamePathified() */
unsigned int GetFilenameBaseOffset( const char *fullpath )
{
  char *slash;
  if ( !fullpath )
    return 0;
  #if (CLIENT_OS == OS_MACOS)
    slash = strrchr( fullpath, ':' );
  #elif (CLIENT_OS == OS_VMS)
    slash = strrchr( fullpath, ':' );
    char *slash2 = strrchr( fullpath, '$' );
    if (slash2 > slash) slash = slash2;
  #elif (CLIENT_OS == OS_RISCOS)
    slash = strrchr( fullpath, '.' );
  #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || \
    (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || \
    (CLIENT_OS == OS_OS2)
    slash = strrchr( (char*) fullpath, '\\' );
    char *slash2 = strrchr( (char*) fullpath, '//' );
    if (slash2 > slash) slash = slash2;
    slash2 = strrchr( (char*) fullpath, ':' );
    if (slash2 > slash) slash = slash2;
  #elif (CLIENT_OS == OS_NETWARE)
    slash = strrchr( fullpath, '\\' );
    char *slash2 = strrchr( fullpath, '//' );
    if (slash2 > slash) slash = slash2;
    slash2 = strrchr( fullpath, ':' );
    if (slash2 > slash) slash = slash2;
  #elif (CLIENT_OS == OS_AMIGAOS)
    slash = strrchr( fullpath, '/' );
    char *slash2 = strrchr( fullpath, ':' );
    if (slash2 > slash) slash = slash2;
  #else
    slash = strrchr( fullpath, '/' );
  #endif
  return (( slash == NULL ) ? (0) : (( slash - fullpath )+1) );
}

/* ------------------------------------------------------------------------ */

static char cwdBuffer[MAX_FULLPATH_BUFFER_LENGTH+1];
static unsigned int cwdBufferLength = 0xFFFFFFFFL;

/* ------------------------------------------------------------------------ */
/* the working directory is the app's directory, unless the ini filename    */
/* contains a dirspec, in which case the path to the ini file is used.      */
/* Exception to the rule: win32/win16 which always use the app's directory. */
/* ------------------------------------------------------------------------ */

int InitWorkingDirectoryFromSamplePaths( const char *inipath, const char *apppath )
{
  if ( inipath == NULL ) inipath = "";
  if ( apppath == NULL ) apppath = "";

  #if (CLIENT_OS == OS_MACOS)
  {
    strcpy( cwdBuffer, inipath );
    char *slash = strrchr(cwdBuffer, ':');
    if (slash != NULL) *(slash+1) = 0;   //<peterd> On the Mac, the current
    else cwdBuffer[0] = 0; // directory is always the apps directory at startup.
  }
  #elif (CLIENT_OS == OS_VMS)
  {
    strcpy( cwdBuffer, inipath );
    char *slash, *bracket, *dirend;
    slash = strrchr(cwdBuffer, ':');
    bracket = strrchr(cwdBuffer, ']');
    dirend = (slash > bracket ? slash : bracket);
    if (dirend == NULL && apppath != NULL && strlen( apppath ) > 0)
    {
      strcpy( cwdBuffer, apppath );
      slash = strrchr(cwdBuffer, ':');
      bracket = strrchr(cwdBuffer, ']');
      dirend = (slash > bracket ? slash : bracket);
    }
    if (dirend != NULL) *(dirend+1) = 0;
    else cwdBuffer[0] = 0;  //current directory is also always the apps dir
  }
  #elif (CLIENT_OS == OS_NETWARE)
  {
    strcpy( cwdBuffer, inipath );
    char *slash = strrchr(cwdBuffer, '/');
    char *slash2 = strrchr(cwdBuffer, '\\');
    if (slash2 > slash) slash = slash2;
    if ( slash == NULL )
    {
      if ( strlen( cwdBuffer ) < 2 || cwdBuffer[1] != ':' )
      {
        strcpy( cwdBuffer, apppath );
        slash = strrchr(cwdBuffer, '/');
        slash2 = strrchr(cwdBuffer, '\\');
        if (slash2 > slash) slash = slash2;
      }
      if ( slash == NULL && strlen( cwdBuffer ) >= 2 && cwdBuffer[1] == ':' )
        slash = &( cwdBuffer[1] );
    }
    if (slash != NULL) *(slash+1) = 0;
    else cwdBuffer[0] = 0;
  }
  #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || \
        (CLIENT_OS == OS_WIN16)
  {
    strcpy( cwdBuffer, inipath );
    char *slash = strrchr(cwdBuffer, '/');
    char *slash2 = strrchr(cwdBuffer, '\\');
    if (slash2 > slash) slash = slash2;
    slash2 = strrchr(cwdBuffer, ':');
    if (slash2 > slash) slash = slash2;
    if ( slash == NULL )
    {
      strcpy( cwdBuffer, apppath );
      if ( *apppath == 0 )
        GetModuleFileName(NULL, cwdBuffer, sizeof(cwdBuffer));
      slash = strrchr(cwdBuffer, '\\');
    }
    if (slash != NULL) *(slash+1) = 0;
    else cwdBuffer[0] = 0;
  }
  #elif (CLIENT_OS == OS_DOS) || ( (CLIENT_OS == OS_OS2) && !defined(__EMX__) )
  {
    strcpy( cwdBuffer, inipath );
    char *slash = strrchr(cwdBuffer, '/');
    char *slash2 = strrchr(cwdBuffer, '\\');
    if (slash2 > slash) slash = slash2;
    slash2 = strrchr(cwdBuffer, ':');
    if (slash2 > slash) slash = slash2;
    if ( slash == NULL )
    {
      strcpy( cwdBuffer, apppath );
      slash = strrchr(cwdBuffer, '\\');
    }
    if ( slash == NULL )
    {
      cwdBuffer[0] = cwdBuffer[ sizeof( cwdBuffer )-2 ] = 0;
      if ( getcwd( cwdBuffer, sizeof( cwdBuffer )-2 )==NULL )
        strcpy( cwdBuffer, ".\\" ); //don't know what else to do
      else if ( cwdBuffer[ strlen( cwdBuffer )-1 ] != '\\' )
        strcat( cwdBuffer, "\\" );
    }
    else
    {
      *(slash+1) = 0;
      if ( ( *slash == ':' ) && ( strlen( cwdBuffer )== 2 ) )
      {
        char buffer[256];
        buffer[0] = buffer[ sizeof( buffer )-1 ] = 0;
        #if (defined(__WATCOMC__))
        {
          unsigned int drive1, drive2, drive3;
          _dos_getdrive( &drive1 );   /* 1 to 26 */
          drive2 = ( toupper( *cwdBuffer ) - 'A' )+1;  /* 1 to 26 */
          _dos_setdrive( drive2, &drive3 );
          _dos_getdrive( &drive3 );
          if (drive2 != drive3 || getcwd( buffer, sizeof(buffer)-1 )==NULL)
            buffer[0]=0;
          _dos_setdrive( drive1, &drive3 );
        }
        #else
          #error FIXME: need to get the current directory on that drive
          //strcat( cwdBuffer, ".\\" );  does not work
        #endif
        if (buffer[0] != 0 && strlen( buffer ) < (sizeof( cwdBuffer )-2) )
        {
          strcpy( cwdBuffer, buffer );
          if ( cwdBuffer[ strlen( cwdBuffer )-1 ] != '\\' )
            strcat( cwdBuffer, "\\" );
        }
      }
    }
  }
  #elif ( CLIENT_OS == OS_RISCOS )
  {
    const char *runpath = NULL;
    if (*inipath == '\0')
    {
      inipath = apppath;
      runpath = "Run$Path";
    }
    _swi(OS_FSControl, _INR(0,5), 37, inipath, cwdBuffer,
                         runpath, NULL, sizeof cwdBuffer);
    char *slash = strrchr( cwdBuffer, '.' );
    if ( slash != NULL )
      *(slash+1) = 0;
    else cwdBuffer[0]=0;
  }
  #elif (CLIENT_OS == OS_AMIGAOS)
  {
    strcpy( cwdBuffer, inipath );
    char *slash = strrchr(cwdBuffer, ':');
    char *slash2 = strrchr(cwdBuffer, '/');
    if (slash2 > slash) slash = slash2;
    if (slash == NULL && apppath != NULL && strlen( apppath ) > 0)
    {
      strcpy( cwdBuffer, apppath );
      slash = strrchr(cwdBuffer, ':');
    }
    if (slash != NULL) *(slash+1) = 0;
    else cwdBuffer[0] = 0; // Means we're started from the dir the things are in...
  }
  #else
  {
    strcpy( cwdBuffer, inipath );
    char *slash = strrchr( cwdBuffer, '/' );
    if (slash == NULL)
    {
      strcpy( cwdBuffer, apppath );
      slash = strrchr( cwdBuffer, '/' );
    }
    if ( slash != NULL )
      *(slash+1) = 0;
    else
      strcpy( cwdBuffer, "./" );
  }
  #endif
  cwdBufferLength = strlen( cwdBuffer );

  #ifdef DEBUG_PATHWORK
  printf( "Working directory is \"%s\"\n", cwdBuffer );
  #endif
  return 0;
}

/* --------------------------------------------------------------------- */

const char *GetWorkingDirectory( char *buffer, unsigned int maxlen )
{
  if ( buffer == NULL )
    return NULL;  //could do a malloc() here. naah.
  if ( maxlen == 0 )
    return "";
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || \
     (CLIENT_OS == OS_WIN32S)
  if ( cwdBufferLength == 0xFFFFFFFFL )
    InitWorkingDirectoryFromSamplePaths( "", "" );
  #endif
  if ( cwdBufferLength == 0xFFFFFFFFL ) //not initialized
    buffer[0] = 0;
  else if ( cwdBufferLength > maxlen )
    return NULL;
  else
    strcpy( buffer, cwdBuffer );
  return buffer;
}

/* --------------------------------------------------------------------- */

static char pathBuffer[MAX_FULLPATH_BUFFER_LENGTH+2];

const char *GetFullPathForFilename( const char *filename )
{
  const char *outpath;

  if ( filename == NULL || filename[0] == 0 )
    outpath = "";
  else if ( GetFilenameBaseOffset( filename ) != 0 )
    outpath = filename;
  else if ( GetWorkingDirectory( pathBuffer, sizeof( pathBuffer ) )==NULL )
    outpath = "";
  else if (strlen(pathBuffer)+strlen(filename)+1 > sizeof(pathBuffer))
    outpath = "";
  else
    outpath = strcat(pathBuffer, filename);

  #ifdef DEBUG_PATHWORK
  printf( "got \"%s\" returning \"%s\"\n", filename, outpath );
  #endif

  return ( outpath );
}

/* --------------------------------------------------------------------- */

const char *GetFullPathForFilenameAndDir( const char *fname, const char *dir )
{
  if (!fname)
    fname = "";
  else if ( GetFilenameBaseOffset( fname ) != 0 )
    return fname;
  if (!dir)
    return GetFullPathForFilename( fname );
  if ( ( strlen( dir ) + strlen( fname ) + 2 ) > sizeof(pathBuffer) )
    return "";
  strcpy( pathBuffer, dir );
  if ( strlen( pathBuffer ) > GetFilenameBaseOffset( pathBuffer ) )
  {
    /* dirname is not terminated with a directory separator - so add one */
    #if (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_VMS)
      strcat( pathBuffer, ":" );
    #elif (CLIENT_OS == OS_RISCOS)
      strcat( pathBuffer, "." );
    #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || \
      (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S) || \
      (CLIENT_OS == OS_OS2)
      strcat( pathBuffer, "\\" );
    #else
      strcat( pathBuffer, "/" );
    #endif
  }
  strcat( pathBuffer, fname );
  return pathBuffer;
}  
