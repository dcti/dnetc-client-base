// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: logstuff.h,v $
// Revision 1.3  1998/10/03 04:04:04  cyp
// changed prototype for CliClearScreen() into a #redefinition of ConClear()
//
// Revision 1.2  1998/08/03 21:13:40  cyruspatel
// Fixed many bugs (especially percentbar related ones). New log file
// types work correctly now. Added some functionality, eg a function for
// obtaining the name of the last used log file (used for win32 gui graphing).
//
// Revision 1.1  1998/08/02 16:00:44  cyruspatel
// Created. Check the FIXMEs! Please get in touch with me before implementing
// support for the extended file logging types (rotate/fifo/restart types).
//
//

#ifndef __LOGSTUFF_H__
#define __LOGSTUFF_H__

//for compatability until it can be changed everywhere.
#include "console.h"
#define CliScreenClear() ConClear()

//Flush mail and if last screen write didn't end with a LF then do that now. 
extern void LogFlush( int forceflush );

//Log message to screen only. Make adjustments, like fixing a missing datestamp
extern void LogScreen( const char *format, ... ); //identical to LogScreenf()

//Legacy function. Same as LogScreen(...)
//extern void LogScreenf( const char *format, ... ); 
#define LogScreenf LogScreen

//Log to mail+file+screen. Make adjustments.
extern void Log( const char *format, ... );

//Log message in raw form (no adjustments) to screen only.
extern void LogScreenRaw( const char *format, ... );

//Log to mail+file+screen. No adjustments.
extern void LogRaw( const char *format, ... );

//display percent bar. (bar is now always compound form)
extern void LogScreenPercent( unsigned int load_problem_count );

//Return name of last accessed logfile, or NULL if not logging to file, 
//or "" if logfile hasn't been accessed yet.
extern const char *LogGetCurrentLogFilename( void );

#endif //__LOGSTUFF_H__

