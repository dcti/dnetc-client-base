/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de> to be able to throw
 * away some very ugly hackery in buffer open code.
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
*/
const char *pathwork_cpp(void) {
return "@(#)$Id: pathwork.cpp,v 1.15.2.6 2001/04/15 18:23:52 oliver Exp $"; }

#include <stdio.h>
#include <string.h>
#include "cputypes.h"
#include "pathwork.h"

#if (CLIENT_OS == OS_DOS) ||  ( (CLIENT_OS == OS_OS2) && !defined(__EMX__) )
  #include <dos.h>  //drive functions
  #include <ctype.h> //toupper
  #if defined(__WATCOMC__)
    #include <direct.h> //getcwd
  #elif defined(__TURBOC__)
    #include <dir.h>
  #endif
#elif (CLIENT_OS == OS_RISCOS)
  #include <swis.h>
#elif defined(__unix__)
  #include <unistd.h>    /* geteuid() */
  #include <pwd.h>       /* getpwnam(), getpwuid(), struct passwd */
  #define HAVE_UNIX_TILDE_EXPANSION
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
    (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
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

static const char *__finalize_fixup(char *path, unsigned int maxlen)
{
  maxlen = maxlen;  /* shaddup compiler */
#if defined(HAVE_UNIX_TILDE_EXPANSION)
  if (*path == '~')
  {
    char username[64];
    unsigned int usernamelen = 0;
    const char *homedir = (const char *)0;
    char *rempath = &path[1];
    while (*rempath && *rempath != '/' && 
           usernamelen < (sizeof(username)-1))
    {
      username[usernamelen++] = *rempath++;
    } 
    if (usernamelen < (sizeof(username)-1))
    {
      struct passwd *pw = (struct passwd *)0;
      username[usernamelen] = '\0';
      if (usernamelen == 0)
        pw = getpwuid(geteuid());
      else 
        pw = getpwnam(username);
      if (pw)
        homedir = pw->pw_dir;
    }
    if (homedir)
    {
      unsigned int dirlen = strlen(homedir);
      unsigned int remlen = strlen(rempath);
      if (*rempath == '/' && dirlen > 0 && homedir[dirlen-1] == '/')
      {
        rempath++; 
        remlen--;  
      }  
      if ((remlen+1+dirlen+1) < maxlen)
      {
        memmove( &path[dirlen], rempath, remlen+1 );
        memcpy( &path[0], homedir, dirlen);
      }
    }   
  }
#endif
  return path;
}


static char __cwd_buffer[MAX_FULLPATH_BUFFER_LENGTH+1];
static int __cwd_buffer_len = -1; /* not initialized */

/* ------------------------------------------------------------------------ */
/* the working directory is the app's directory, unless the ini filename    */
/* contains a dirspec, in which case the path to the ini file is used.      */
/* ------------------------------------------------------------------------ */

int InitWorkingDirectoryFromSamplePaths( const char *inipath, const char *apppath )
{
  if ( inipath == NULL ) inipath = "";
  if ( apppath == NULL ) apppath = "";

  __cwd_buffer[0] = '\0';
  #if (CLIENT_OS == OS_MACOS)
  {
    strcpy( __cwd_buffer, inipath );
    char *slash = strrchr(__cwd_buffer, ':');
    if (slash != NULL) *(slash+1) = 0;   //<peterd> On the Mac, the current
    else __cwd_buffer[0] = 0; // directory is always the apps directory at startup.
  }
  #elif (CLIENT_OS == OS_VMS)
  {
    strcpy( __cwd_buffer, inipath );
    char *slash, *bracket, *dirend;
    slash = strrchr(__cwd_buffer, ':');
    bracket = strrchr(__cwd_buffer, ']');
    dirend = (slash > bracket ? slash : bracket);
    if (dirend == NULL && apppath != NULL && strlen( apppath ) > 0)
    {
      strcpy( __cwd_buffer, apppath );
      slash = strrchr(__cwd_buffer, ':');
      bracket = strrchr(__cwd_buffer, ']');
      dirend = (slash > bracket ? slash : bracket);
    }
    if (dirend != NULL) *(dirend+1) = 0;
    else __cwd_buffer[0] = 0;  //current directory is also always the apps dir
  }
  #elif (CLIENT_OS == OS_NETWARE)
  {
    strcpy( __cwd_buffer, inipath );
    char *slash = strrchr(__cwd_buffer, '/');
    char *slash2 = strrchr(__cwd_buffer, '\\');
    if (slash2 > slash) slash = slash2;
    if ( slash == NULL )
    {
      if ( strlen( __cwd_buffer ) < 2 || __cwd_buffer[1] != ':' ) /* dos partn */
      {
        strcpy( __cwd_buffer, apppath );
        slash = strrchr(__cwd_buffer, '/');
        slash2 = strrchr(__cwd_buffer, '\\');
        if (slash2 > slash) slash = slash2;
      }
      if ( slash == NULL && strlen( __cwd_buffer ) >= 2 && __cwd_buffer[1] == ':' )
        slash = &( __cwd_buffer[1] );
    }
    if (slash != NULL) *(slash+1) = 0;
    else __cwd_buffer[0] = 0;
  }
  #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  {
    strcpy( __cwd_buffer, inipath );
    char *slash = strrchr(__cwd_buffer, '/');
    char *slash2 = strrchr(__cwd_buffer, '\\');
    if (slash2 > slash) slash = slash2;
    slash2 = strrchr(__cwd_buffer, ':');
    if (slash2 > slash) slash = slash2;
    if ( slash == NULL )
    {
      strcpy( __cwd_buffer, apppath );
      slash = strrchr(__cwd_buffer, '/');
      slash2 = strrchr(__cwd_buffer, '\\');
      if (slash2 > slash) slash = slash2;
      slash2 = strrchr(__cwd_buffer, ':');
      if (slash2 > slash) slash = slash2;
    }
    if (slash != NULL) *(slash+1) = 0;
    else __cwd_buffer[0] = 0;
  }
  #elif (CLIENT_OS == OS_DOS) || ( (CLIENT_OS == OS_OS2) && !defined(__EMX__) )
  {
    strcpy( __cwd_buffer, inipath );
    char *slash = strrchr(__cwd_buffer, '/');
    char *slash2 = strrchr(__cwd_buffer, '\\');
    if (slash2 > slash) slash = slash2;
    slash2 = strrchr(__cwd_buffer, ':');
    if (slash2 > slash) slash = slash2;
    if ( slash == NULL )
    {
      strcpy( __cwd_buffer, apppath );
      slash = strrchr(__cwd_buffer, '\\');
    }
    if ( slash == NULL )
    {
      __cwd_buffer[0] = __cwd_buffer[ sizeof( __cwd_buffer )-2 ] = 0;
      if ( getcwd( __cwd_buffer, sizeof( __cwd_buffer )-2 )==NULL )
        strcpy( __cwd_buffer, ".\\" ); //don't know what else to do
      else if ( __cwd_buffer[ strlen( __cwd_buffer )-1 ] != '\\' )
        strcat( __cwd_buffer, "\\" );
    }
    else
    {
      *(slash+1) = 0;
      if ( ( *slash == ':' ) && ( strlen( __cwd_buffer )== 2 ) )
      {
        char buffer[256];
        buffer[0] = buffer[ sizeof( buffer )-1 ] = 0;
        #if (defined(__WATCOMC__))
        {
          unsigned int drive1, drive2, drive3;
          _dos_getdrive( &drive1 );   /* 1 to 26 */
          drive2 = ( toupper( *__cwd_buffer ) - 'A' )+1;  /* 1 to 26 */
          _dos_setdrive( drive2, &drive3 );
          _dos_getdrive( &drive3 );
          if (drive2 != drive3 || getcwd( buffer, sizeof(buffer)-1 )==NULL)
            buffer[0]=0;
          _dos_setdrive( drive1, &drive3 );
        }
        #else
          #error FIXME: need to get the current directory on that drive
          //strcat( __cwd_buffer, ".\\" );  does not work
        #endif
        if (buffer[0] != 0 && strlen( buffer ) < (sizeof( __cwd_buffer )-2) )
        {
          strcpy( __cwd_buffer, buffer );
          if ( __cwd_buffer[ strlen( __cwd_buffer )-1 ] != '\\' )
            strcat( __cwd_buffer, "\\" );
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
    _swi(OS_FSControl, _INR(0,5), 37, inipath, __cwd_buffer,
                         runpath, NULL, sizeof __cwd_buffer);
    char *slash = strrchr( __cwd_buffer, '.' );
    if ( slash != NULL )
      *(slash+1) = 0;
    else __cwd_buffer[0]=0;
  }
  #elif (CLIENT_OS == OS_AMIGAOS)
  {
    strcpy( __cwd_buffer, inipath );
    char *slash = strrchr(__cwd_buffer, ':');
    char *slash2 = strrchr(__cwd_buffer, '/');
    if (slash2 > slash) slash = slash2;
    if (slash == NULL && apppath != NULL && strlen( apppath ) > 0)
    {
      strcpy( __cwd_buffer, apppath );
      slash = strrchr(__cwd_buffer, ':');
    }
    if (slash != NULL) *(slash+1) = 0;
    else __cwd_buffer[0] = 0; // Means we're started from the dir the things are in...
  }
  #else
  {
    strcpy( __cwd_buffer, inipath );
    char *slash = strrchr( __cwd_buffer, '/' );
    if (slash == NULL)
    {
      strcpy( __cwd_buffer, apppath );
      slash = strrchr( __cwd_buffer, '/' );
    }
    if ( slash != NULL )
      *(slash+1) = 0;
    else
      strcpy( __cwd_buffer, "./" );
    __finalize_fixup( __cwd_buffer, sizeof(__cwd_buffer) );
  }
  #endif
  __cwd_buffer_len = strlen( __cwd_buffer );

  #ifdef DEBUG_PATHWORK
  printf( "Working directory is \"%s\"\n", __cwd_buffer );
  #endif
  return 0;
}

/* --------------------------------------------------------------------- */

static int __is_filename_absolute(const char *fname)
{
  #if (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_VMS)
  return (*fname == ':');
  #elif (CLIENT_OS == OS_RISCOS)
  return (*fname == '.');
  #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || \
      (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
  return (*fname == '\\' || *fname == '/' || (*fname && fname[1]==':'));       
  #elif (CLIENT_OS == OS_NETWARE)
  return (*fname == '\\' || *fname == '/' || (strchr(fname,':')));       
  #elif (CLIENT_OS == OS_AMIGAOS)
  return (strchr(fname,':') != NULL);       
  #else
  return (*fname == '/');
  #endif
}

/* --------------------------------------------------------------------- */

/* get working directory (may be relative to system's current directory),
** including trailing directory separator. Returns NULL on error.
*/
const char *GetWorkingDirectory( char *buffer, unsigned int maxlen )
{
  if ( buffer == NULL )
    return buffer; //could do a malloc() here. naah.
  else if ( maxlen == 0 )
    return "";
  else if ( __cwd_buffer_len <= 0 ) /* not initialized or zero len */
    buffer[0] = 0;
  else if ( ((unsigned int)__cwd_buffer_len) > maxlen )
    return NULL;
  else
    strcpy( buffer, __cwd_buffer );
  /* __cwd_buffer is already fixed up, so __finalize_fixup() is not needed */
  return buffer;
}

/* --------------------------------------------------------------------- */

static char __path_buffer[MAX_FULLPATH_BUFFER_LENGTH+2];

/* GetFullPathForFilename( const char *fname )
**   is a misnomer - should be called GetWorkingPathForFilename()
**   to reflect that returned paths may be relative.
** returns "" on error.
** returns <fname> if <fname> is absolute
** otherwise returns <X> + <fname>
**   where <X> is "" if <fname> is not just a plain basename (has a dir spec)
**   where <X> is __cwd_buffer + <dirsep> if <fname> is just a plain basename
*/   
const char *GetFullPathForFilename( const char *filename )
{
  const char *outpath;

  if ( filename == NULL )
    outpath = "";
  else if (*filename == '\0' || __is_filename_absolute(filename))
    outpath = filename;
  else if (strlen(filename) >= sizeof(__path_buffer))
    outpath = "";
  else if ( GetFilenameBaseOffset( filename ) != 0 ) /* already pathified */
    outpath = __finalize_fixup(strcpy(__path_buffer, filename),sizeof(__path_buffer));
  else if ( !GetWorkingDirectory( __path_buffer, sizeof( __path_buffer ) ) )
    outpath = "";
  else if (strlen(__path_buffer)+strlen(filename) >= sizeof(__path_buffer))
    outpath = "";
  else
    outpath = __finalize_fixup(strcat(__path_buffer, filename),sizeof(__path_buffer));

  #ifdef DEBUG_PATHWORK
  printf( "got \"%s\" returning \"%s\"\n", filename, outpath );
  #endif

  return ( outpath );
}

/* --------------------------------------------------------------------- */

/* GetFullPathForFilenameAndDir( const char *fname, const char *dir )
** returns <fname> if <fname> is absolute
** otherwise returns <dir> + <dirsep> + <fname>
**      if <dir> is NULL, use __cwd_buffer as <dir>; 
**      if <dir> is "", use ""
**      if <fname> is NULL, use "" as <fname>
*/   
const char *GetFullPathForFilenameAndDir( const char *fname, const char *dir )
{
  if (!fname)
    fname = "";
  else if (__is_filename_absolute(fname))
    return fname;
  if (!dir)
    dir = ((__cwd_buffer_len > 0) ? (__cwd_buffer) : "");

  if ( ( strlen( dir ) + 1 + strlen( fname ) + 1 ) >= sizeof(__path_buffer) )
    return "";
  strcpy( __path_buffer, dir );
  if ( strlen( __path_buffer ) > GetFilenameBaseOffset( __path_buffer ) )
  {
    /* dirname is not terminated with a directory separator - so add one */
    #if (CLIENT_OS == OS_MACOS) || (CLIENT_OS == OS_VMS)
      strcat( __path_buffer, ":" );
    #elif (CLIENT_OS == OS_RISCOS)
      strcat( __path_buffer, "." );
    #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16) || \
      (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
      strcat( __path_buffer, "\\" );
    #else
      strcat( __path_buffer, "/" );
    #endif
  }
  strcat( __path_buffer, fname );
  return __finalize_fixup(__path_buffer, sizeof(__path_buffer));
}  

