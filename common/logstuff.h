/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __LOGSTUFF_H__
#define __LOGSTUFF_H__ "@(#)$Id: logstuff.h,v 1.8.2.1 1999/04/13 19:45:23 jlawson Exp $"

#define LOGFILETYPE_NONE    0x00 //
#define LOGFILETYPE_NOLIMIT 0x01 //unlimited (or limit == -1)
#define LOGFILETYPE_ROTATE  0x02 //then logLimit is in days
#define LOGFILETYPE_RESTART 0x04 //then logLimit is in KByte 
#define LOGFILETYPE_FIFO    0x08 //then logLimit is in KByte (minimum 100K)

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
extern void LogScreen( const char *format, ... );

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

