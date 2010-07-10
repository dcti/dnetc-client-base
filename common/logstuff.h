/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __LOGSTUFF_H__
#define __LOGSTUFF_H__ "@(#)$Id: logstuff.h,v 1.20 2010/07/10 17:35:31 stream Exp $"

/* this is shared with Configure() */
#define LOGFILETYPE_NONE    0 //no logging to file
#define LOGFILETYPE_NOLIMIT 1 //unlimited (or limit == -1)
#define LOGFILETYPE_RESTART 2 //then logLimit is in KByte
#define LOGFILETYPE_FIFO    3 //then logLimit is in KByte (minimum 100K)
#define LOGFILETYPE_ROTATE  4 //then logLimit is in days

/* this is shared with anything that uses LogTo() */
#define LOGTO_NONE       0x00
#define LOGTO_SCREEN     0x01
#define LOGTO_FILE       0x02
#define LOGTO_MAIL       0x04
#define LOGTO_RAWMODE    0x80

/* log modes used in corresponding functions for simple usage in LogTo() */
#define LOGAS_LOG        (LOGTO_SCREEN | LOGTO_FILE | LOGTO_MAIL)  // same as Log()
#define LOGAS_LOGSCREEN  LOGTO_SCREEN                              // same as LogScreen()

/* ---------------------------------------------------- */

#if defined(__GNUC__)
#define __CHKFMT_PRINTF __attribute__((format(printf,1,2)))
#define __CHKFMT_LOGTO  __attribute__((format(printf,2,3)))
#else
#define __CHKFMT_PRINTF
#define __CHKFMT_LOGTO
#endif

//Flush mail and if last screen write didn't end with a LF then do that now.
extern void LogFlush( int forceflush );

//Log message to screen only. Make adjustments, like fixing a missing datestamp
extern void LogScreen( const char *format, ... ) __CHKFMT_PRINTF;

//Log to mail+file+screen. Make adjustments.
extern void Log( const char *format, ... ) __CHKFMT_PRINTF;

//Log message in raw form (no adjustments) to screen only.
extern void LogScreenRaw( const char *format, ... ) __CHKFMT_PRINTF;

//Log to mail+file+screen. No adjustments.
extern void LogRaw( const char *format, ... ) __CHKFMT_PRINTF;

//Log to LOGTO_* flags (RAW implies screen)
extern void LogTo( int towhat, const char *format, ... ) __CHKFMT_LOGTO;

//display percent bar. (bar is now always compound form)
extern void LogScreenPercent( unsigned int load_problem_count );

//Return name of current logfile, or NULL if not logging to file,
//or "" if logfile hasn't been accessed yet.
extern const char *LogGetCurrentLogFilename(char *buffer, unsigned int len);

//init/deinit prototypes
void DeinitializeLogging(void);
void InitializeLogging( int noscreen, int nopercent, int nopercbaton,
                        const char *logfilename, int rotateUTC,
                        const char *logfiletype, const char *logfilelimit,
                        long mailmsglen, const char *smtpsrvr,
                        unsigned int smtpport, const char *smtpfrom,
                        const char *smtpdest, const char *id );

#define PROJECT_NOT_HANDLED(cID) Log("PROJECT: %d NOT HANDLED in %s line %d", cID, __FILE__, __LINE__)
// PROJECT_NOT_HANDLED is also used to mark all places to be fixed
// when a new project needs to be integrated into source

#endif //__LOGSTUFF_H__

