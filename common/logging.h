// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// Logging functions

  void LogScreenf( const char *format, ... );
    // logs message to screen only

  void Log( const char *format, ... );
    // logs message to screen and file (append mode)
    // if logname isn't set, then only to screen

#if defined(NEEDVIRTUALMETHODS)
  virtual void LogScreen ( const char *text );
    // logs preformated message to screen only.  can be overriden.
#else
  void LogScreen ( const char *text );
    // logs preformated message to screen only.  can be overriden.
#endif

// --------------------------------------------------------------------------
// Logging variables:

  extern s32 quietmode;
  extern char logstr[1024];
#ifdef DONT_USE_PATHWORK
  extern char ini_logname[128];// Logfile name as is in the .ini
#else
  extern char logname[128];// Logfile name
#endif

