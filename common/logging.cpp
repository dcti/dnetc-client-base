// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "baseincs.h"
#include "client.h"
#include "pathwork.h"

// --------------------------------------------------------------------------
// Global variables:

  s32 quietmode;
  char logstr[1024];
#ifdef DONT_USE_PATHWORK
  char ini_logname[128];// Logfile name as is in the .ini
#else
  char logname[128];// Logfile name
#endif

// ------------------------------------------------------------------------

#if !defined(NEEDVIRTUALMETHODS)

// gui clients will override this function in their derrived classes
void LogScreen ( const char *text)
{
  if (!quietmode)
  {
#if (CLIENT_OS == OS_OS2)
    DosSetPriority(PRTYS_THREAD, PRTYC_REGULAR, 0, 0);
    // Give prioirty boost so that text appears faster
#endif
    fwrite(text, 1, strlen(text), stdout);
    fflush(stdout);
#if (CLIENT_OS == OS_OS2)
    SetNiceness();
#endif
  }
}

#endif
// ---------------------------------------------------------------------------

void LogScreenf ( const char *format, ...)
{
#if defined(NEEDVIRTUALMETHODS)
  // the gui clients depend on the overridden LogScreen for output
  va_list argptr;
  va_start(argptr, format);
  vsprintf(logstr, format, argptr);
  LogScreen(logstr);
  va_end(argptr);
#else
  if (!quietmode)
  {
#if (CLIENT_OS == OS_OS2)
    DosSetPriority(PRTYS_THREAD, PRTYC_REGULAR, 0, 0);
    // Give prioirty boost so that text appears faster
#endif
    va_list argptr;
    va_start(argptr, format);
    vprintf( format, argptr); // display it
    fflush( stdout );
    va_end(argptr);
#if (CLIENT_OS == OS_OS2)
    SetNiceness();
#endif
  }
#endif
}

// ---------------------------------------------------------------------------

void Log( const char *format, ...)
{
  va_list argptr;

  // format the buffer
  va_start(argptr, format);
  vsprintf(logstr, format, argptr);
  va_end(argptr);

  //add to mail
  mailmessage.addtomessage( logstr );

  // print it out and log it
  LogScreen(logstr);

  if ( IsFilenameValid( logname ) )
    {
#ifdef DONT_USE_PATHWORK
      FILE *file = fopen ( logname, "a" );
#else
      FILE *file = fopen ( GetFullPathForFilename( logname ), "a" );
#endif
      if (file != NULL)
        {
          fprintf( file, "%s", logstr );
          fclose( file );
        }
    }
}

