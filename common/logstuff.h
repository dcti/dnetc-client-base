// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: logstuff.h,v $
// Revision 1.8  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.7  1998/11/28 19:44:34  cyp
// InitializeLogging() and DeinitializeLogging() are no longer Client class
// methods.
//
// Revision 1.6  1998/10/06 21:29:14  cyp
// Removed prototype for LogSetTimeStampingMode()
//
// Revision 1.5  1998/10/05 01:58:07  cyp
// Implemented automatic time stamping. Added LogSetTimeStampingMode(int) to
// enable timestamps once the ::Run has started.
//
// Revision 1.4  1998/10/03 17:00:30  sampo
// Finished the ConClear() replacement of CliScreenClear.
//
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

#define LOGFILETYPE_NONE    0 //
#define LOGFILETYPE_NOLIMIT 1 //unlimited (or limit == -1)
#define LOGFILETYPE_ROTATE  2 //then logLimit is in days
#define LOGFILETYPE_RESTART 4 //then logLimit is in KByte 
#define LOGFILETYPE_FIFO    8 //then logLimit is in KByte (minimum 100K)

#define LOGTO_NONE       0x00 
#define LOGTO_SCREEN     0x01
#define LOGTO_FILE       0x02
#define LOGTO_MAIL       0x04
#define LOGTO_RAWMODE    0x80    

#define MAX_LOGENTRY_LEN 1024 //don't make this smaller than 1K!

#define ASSERT_WIDTH_80     //show where badly formatted lines are cropping up

/* ---------------------------------------------------- */

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

//init/deinit prototypes
void DeinitializeLogging(void);
void InitializeLogging( int noscreen, int nopercent, const char *logfilename, 
                        unsigned int logfiletype, int logfilelimit, 
                        long mailmsglen, const char *smtpsrvr, 
                        unsigned int smtpport, const char *smtpfrom, 
                        const char *smtpdest, const char *id );
#endif //__LOGSTUFF_H__

