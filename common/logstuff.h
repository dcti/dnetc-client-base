/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __LOGSTUFF_H__
#define __LOGSTUFF_H__ "@(#)$Id: logstuff.h,v 1.12 2000/06/02 06:24:56 jlawson Exp $"

#define LOGFILETYPE_NONE    0 //
#define LOGFILETYPE_NOLIMIT 1 //unlimited (or limit == -1)
#define LOGFILETYPE_RESTART 2 //then logLimit is in KByte 
#define LOGFILETYPE_FIFO    3 //then logLimit is in KByte (minimum 100K)
#define LOGFILETYPE_ROTATE  4 //then logLimit is in days

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

//Log to mail+file+screen. Make adjustments.
extern void Log( const char *format, ... );

//Log message in raw form (no adjustments) to screen only.
extern void LogScreenRaw( const char *format, ... );

//Log to mail+file+screen. No adjustments.
extern void LogRaw( const char *format, ... );

//Log to LOGTO_* flags (RAW implies screen)
extern void LogTo( int towhat, const char *format, ... );

//display percent bar. (bar is now always compound form)
extern void LogScreenPercent( unsigned int load_problem_count );

//Return name of last accessed logfile, or NULL if not logging to file, 
//or "" if logfile hasn't been accessed yet.
extern const char *LogGetCurrentLogFilename( void );

//init/deinit prototypes
void DeinitializeLogging(void);
void InitializeLogging( int noscreen, int nopercent, const char *logfilename, 
                        const char *logfiletype, const char *logfilelimit, 
                        long mailmsglen, const char *smtpsrvr, 
                        unsigned int smtpport, const char *smtpfrom, 
                        const char *smtpdest, const char *id );
#endif //__LOGSTUFF_H__

