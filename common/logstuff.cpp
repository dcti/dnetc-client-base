/* Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel (cyp@fb14.uni-mainz.de)
 *
 * ------------------------------------------------------
 * Pardon, oh, pardon, that my sould should make
 * Of all the strong divineness which I know
 * For thine and thee, an image only so
 * Formed of the sand, and fit to shift and break.
 * ------------------------------------------------------
*/
//#define TRACE

const char *logstuff_cpp(void) {
return "@(#)$Id: logstuff.cpp,v 1.37.2.45 2000/12/15 00:33:47 oliver Exp $"; }

#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "client.h"    // CONTEST_COUNT
#include "mail.h"      // MailMessage
#include "clitime.h"   // CliGetTimeString(NULL,1)
#include "pathwork.h"  // GetFullPathForFilename(), GetWorkingDirectory()
#include "problem.h"   // Problem object for logscreenpercent
#include "probman.h"   // GetProblemPointerFromIndex() for logscreenpercent
#include "bench.h"     // BenchGetBestRate() for logscreenpercent
#include "cpucheck.h"  // GetNumberOfDetectedProcessors() for logscreenpercent
#include "clicdata.h"  // CliGetContestNameFromID() for logscreenpercent
#include "console.h"   // for ConOut(), ConIsScreen(), ConIsGUI()
#include "util.h"      // TRACE
#include "logstuff.h"  // keep the prototypes in sync

//-------------------------------------------------------------------------
#ifndef LOGTO_NONE       /* the following are declared in logstuff.h */
#define LOGTO_NONE       0x00
#define LOGTO_SCREEN     0x01
#define LOGTO_FILE       0x02
#define LOGTO_MAIL       0x04
#define LOGTO_RAWMODE    0x80
#endif

#ifndef LOGFILETYPE_NONE    /* the following are declared in logstuff.h */
#define LOGFILETYPE_NONE    0 //no logging to file
#define LOGFILETYPE_NOLIMIT 1 //unlimited (or limit == -1)
#define LOGFILETYPE_RESTART 2 //then logLimit is in KByte
#define LOGFILETYPE_FIFO    3 //then logLimit is in KByte (minimum 100K)
#define LOGFILETYPE_ROTATE  4 //then logLimit is in days
#endif

#define ASSERT_WIDTH_80     //show where badly formatted lines are cropping up
#define ASSUMED_SCREEN_WIDTH 80 //... until all platforms support ConGetSize()
// ========================================================================

static struct
{
  int initlevel;
  int loggingTo;            // LOGTO_xxx bitfields
  int spoolson;             // mail/file logging and time stamping is on/off.
  int crunchmeter;          // progress ind style (-1=def,0=off,1=abs,2=rel)
  int percbaton;            // percent baton is enabled

  void *mailmessage;         //handle returned from smtp_construct_message()
  char basedir[256];         //filled if user's 'logfile' is not qualified
  char logfile[128];         //filename from user
  FILE *logstream;           //open logfile if /dev/* and "no-limit"
  int  logfileType;          //rotate, restart, fifo, none
  unsigned int logfileLimit; //days when rotating or kbyte when fifo/restart
  unsigned int logfilestarted; // non-zero after the first logfile write 
                             //also used to mark len of the log fname without 
                             // ROTATE suffix

  int stdoutisatty;         //log screen can handle lines not ending in '\n'
  int stableflag;           //last log screen did end in '\n'
  int lastwasperc;          //last log screen was a percentbar
  unsigned int perc_callcount; //#of times LogScreenPercent() printed something
  unsigned int lastperc_done; //percentage of last LogScreenPercent() dot

} logstatics = {
  0,      // initlevel
  LOGTO_NONE,   // loggingTo
  0,      // spoolson
  0,      // crunchmeter
  0,      // percbaton
  NULL,   // *mailmessage
  {0},    // basedir[]
  {0},    // logfile[]
  NULL,   // logstream
  LOGFILETYPE_NONE, // logfileType
  0,      // logfileLimit
  0,      // logfilestarted
  0,      // stdoutisatty
  0,      // stableflag
  0,      // lastwasperc
  0,      // perc_callcount
  0       // lastperc_done
};      

// ========================================================================

static void InternalLogScreen( const char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  if ((logstatics.loggingTo & LOGTO_SCREEN) != 0)
  {
    if ( msglen && (msgbuffer[msglen-1] == '\n' || logstatics.stdoutisatty ) )
    {
      if (strlen( msgbuffer ) == msglen) //we don't do binary data
      {                                  //which shouldn't happen anyway.
        ConOut( msgbuffer );             
      }  
    }
    else
      ConOut( "" ); //flush.
  }
  return;
}

// ------------------------------------------------------------------------

static FILE *__fopenlog( const char *fn, const char *mode )
{
  FILE *file = (FILE *)0;
  unsigned int len = strlen(logstatics.basedir);
  if ((len + strlen(fn) +1) < sizeof(logstatics.basedir))
  {
    strcat( logstatics.basedir, fn );
    file = fopen( logstatics.basedir, mode );
    logstatics.basedir[len] = '\0';
  }
  return file;
}

/* returns NULL if logging is (either implicitely or explicitely) disabled */
/* logfiletype is LOGFILETYPE_[ROTATE|RESTART|FIFO|NOLIMIT|NONE] */
/* loglimit is in kilobytes for RESTART and FIFO, in days for ROTATE. */
/* Note to people writing log parsers: the 'limit' in the .ini may be */
/* on a different scale, for example in Mb, you'll need to convert if so. */
static const char *__get_logname( int logfileType, int logfileLimit,
                                  char *buffer, unsigned int buflen )
{
  if (buffer && buflen)
  {
    int gotit = 0;
    switch (logfileType)
    {
      case LOGFILETYPE_ROTATE:
      {
        int tm_year = 0, tm_mon = 0, tm_mday = 0, tm_yday = 0;
        if (logfileLimit > 0 )
        {
          time_t ttime = time(NULL);
          struct tm *currtm = localtime( &ttime );
          if (currtm)
          {
            if (currtm->tm_mon  >= 0  && currtm->tm_mon  < 12 &&
                currtm->tm_year >= 70 && currtm->tm_year <= (9999-1900) &&
                currtm->tm_mday >= 1  && currtm->tm_mday <= 31 &&
                currtm->tm_yday >= 0  && currtm->tm_yday <= 366)
            {
              tm_year = currtm->tm_year; /* years since 1900*/  
              tm_mon = currtm->tm_mon;   /* months since January [0,11]*/ 
              tm_mday = currtm->tm_mday; /* day of the month [1,31]*/ 
              tm_yday = currtm->tm_yday; /* days since January 1 [0, 365]*/ 
            }
          }
        }
        if (tm_mday != 0)
        {
          unsigned int len;
          char rotsuffix[20];
          rotsuffix[0] = '\0';
        
          if (logfileLimit >= 28 && logfileLimit <= 31) /* monthly */
          {
            static const char *monnames[] = {
               "jan","feb","mar","apr","may","jun",
               "jul","aug","sep","oct","nov","dec" };
            sprintf( rotsuffix, "%02d%s", (tm_year%100), monnames[tm_mon] );
          }
          else if (logfileLimit == 365) /* annually */
          {
            sprintf( rotsuffix, "%04d", tm_year+1900 );
          }
          else if (logfileLimit == 7) /* weekly */
          {
            /* we intentionally don't use tm_wday. Technically, week 1 is 
            ** the first week with a monday in it, but we don't care about 
            ** that here.
            */
            sprintf( rotsuffix, "%02dw%02d", (tm_year%100), (tm_yday+1+6)/7 );
          }
          else /* anything else: daily = 1, fortnightly = 14 etc */
          {
            int year  = tm_year+ 1900;
            int month = tm_mon + 1;
            int day   = tm_mday;
            if (logfileLimit > 1) /* not daily - so date stamp may be in the */
            {                     /* past. Use jdn to move back to that date.*/
              /*
              ** What is 'jdn'?: Julian Day Numbers (JDN) are used by
              ** astronomers as a date/time measure independent of calendars and
              ** convenient for computing the elapsed time between dates.  The JDN
              ** for any date/time is the number of days (including fractional
              ** days) elapsed since noon, 1 Jan 4713 BC.  Julian Day Numbers were
              ** originated by Joseph Justus Scaliger (1540-1609) in 1582 and named
              ** after his father Julius, not after Julius Caesar.  They are not 
              ** related to the Julian calendar. Note that some people use the 
              ** term "Julian day number" to refer to any numbering of days. The US
              ** government, for example, uses the term to denote the number of days 
              ** since 1 January of the current year.
              **
              ** The macros used below assume a Gregorian calendar.
              **
              ** Based on formulae originally posted by
              ** Tom Van Flandern metares@well.sf.ca.us in sci.astro
              ** http://www.deja.com/dnquery.xp?QRY=julian+sci.astro+calculating&ST=MS
              ** http://world.std.com/FAQ/Other/calendar-FAQ.txt
              **
              ** Check date is 1 Jan 2000 ==> JDN 2451545
              */ 
              #define _ymd2jdn( yy, mm, dd, __j ) {   \
                long __m = (mm); /*signed!*/          \
                long __a = (14-(__m))/12;             \
                long __y = (yy)+4800-__a;             \
                __m = __m + 12*__a - 3;               \
                __j = ((long)(dd)) + (153*__m+2)/5 +  \
                      __y*365 + __y/4 - __y/100 +     \
                      __y/400 - 32045L;               \
                }
              #define _jdn2ymd( __j, yy, mm, dd ) {   \
                  long __a = (__j) + 32044L;          \
                  long __b = (4*__a+3)/146097L;       \
                  long __c = __a - (__b*146097L)/4;   \
                  long __d = (4*__c+3)/1461L;         \
                  long __e = __c - (1461L*__d)/4;     \
                  long __f = (5*__e+2)/153L;          \
                  dd = __e - (153L*__f+2)/5 + 1;      \
                  mm = __f + 3 - 12*(__f/10);         \
                  yy = __b*100 + __d - 4800 + __f/10; \
              }
              unsigned long jdn;
              _ymd2jdn( year, month, day, jdn );
              jdn -= (jdn % logfileLimit); /* backoff to beginning of period */
              _jdn2ymd( jdn, year, month, day );
            } /* not daily */
            sprintf( rotsuffix, "%02d%02d%02d", (year%100), month, day );
          }
          strcat(rotsuffix, EXTN_SEP"log");
          strncpy(buffer, logstatics.logfile, buflen );
          buffer[buflen-1] = '\0';
          len = strlen(buffer);
          strncpy(&buffer[len], rotsuffix, buflen-len );
          buffer[buflen-1] = '\0';
          gotit = 1;
        } /* if (tm_mday != 0) */
        break;
      } /* case LOGFILETYPE_ROTATE */
      case LOGFILETYPE_RESTART:
        if (logfileLimit < 1)
          break;
      /* fallthrough */
      case LOGFILETYPE_FIFO: /* limit defaults to 100K if < 100K */
      case LOGFILETYPE_NOLIMIT: 
        if (logstatics.logfile[0])
        {
          strncpy(buffer, logstatics.logfile, buflen );
          buffer[buflen-1] = '\0';
          gotit = 1;
        }
        break;
      default:
        break;
    } /* switch */
    if (gotit)
      return buffer;
  }
  return (const char *)0;
}

//this can ONLY be called from LogWithPointer.
static void InternalLogFile( const char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  #if (CLIENT_OS == OS_NETWARE || CLIENT_OS == OS_DOS || \
       CLIENT_OS == OS_OS2 || CLIENT_OS == OS_WIN16 || \
       CLIENT_OS == OS_WIN32)
    #define ftruncate( fd, sz )  chsize( fd, sz )
  #elif (CLIENT_OS == OS_VMS)
    #define ftruncate( fd, sz ) //nada, not supported
    #define FTRUNCATE_NOT_SUPPORTED
  #endif
  int logfileType = logstatics.logfileType;
  unsigned int logfileLimit = logstatics.logfileLimit;
  char logname[sizeof(logstatics.logfile)+32];

  if ( !msglen || msgbuffer[msglen-1] != '\n' ||
       !logstatics.spoolson || (logstatics.loggingTo & LOGTO_FILE) == 0)
  {
    logfileType = LOGFILETYPE_NONE;
  }

  if ( logfileType == LOGFILETYPE_NONE)
  {
    ; /* nothing */    
  }
  else if ( logfileType == LOGFILETYPE_RESTART)
  {
    if (__get_logname( logfileType, logfileLimit, logname, sizeof(logname) ))
    {
      FILE *file = __fopenlog( logname, "a" );
      if ( file )
      {
        long filelen;
        fseek( file, 0, SEEK_END );
        filelen = (long)ftell( file );
        if (filelen != (long)(-1))
        {
          //filelen += msglen;
          if ( ((unsigned int)(filelen >> 10)) > logfileLimit )
          {
            fclose( file );
            file = __fopenlog( logname, "w" );
            if (file)
            {
              fprintf( file,
                 "[%s] Log file exceeded %uKbyte limit. Restarted...\n\n",
                  CliGetTimeString( NULL, 1 ),
                 (unsigned int)( logstatics.logfileLimit ));
            }
          }
        }
      }
      if (file)
      {
        logstatics.logfilestarted = 1;
        fwrite( msgbuffer, sizeof( char ), msglen, file );
        fclose( file );
        file = NULL;
      }
    }
  }
  else if ( logfileType == LOGFILETYPE_ROTATE )
  {
    if (__get_logname( logfileType, logfileLimit, logname, sizeof(logname) ))
    {
      FILE *file = __fopenlog( logname, "a" );
      if (file)
      {
        fwrite( msgbuffer, sizeof( char ), msglen, file );
        fclose( file );
        file = NULL;
      }
    }
  }
  #ifndef FTRUNCATE_NOT_SUPPORTED
  else if ( logfileType == LOGFILETYPE_FIFO )
  {
    if (__get_logname( logfileType, logfileLimit, logname, sizeof(logname) ))
    {
      FILE *file = __fopenlog( logname, "a" );
      unsigned long filelen = 0;
      if ( logfileLimit < 100 )
        logfileLimit = 100;

      if ( file )
      {
        fwrite( msgbuffer, sizeof( char ), msglen, file );
        if (((long)(filelen = ftell( file ))) == ((long)(-1)))
          filelen = 0;
        fclose( file );
        file = NULL;
      }
      if ( filelen > (((unsigned long)(logfileLimit))<<10) )
      {    /* careful: file must be read/written without translation - cyp */
        unsigned int maxswapsize = 1024*4; //assumed dpage/sector size
        char *swapbuffer = (char *)malloc( maxswapsize );
        if (swapbuffer)
        {
          #if (CLIENT_OS == OS_AMIGAOS)
          // buggy fopen doesn't understand "r+b" (b ignored on AmigaOS anyway)
          file = __fopenlog( logname, "r+" );
          #else
          file = __fopenlog( logname, "r+b" );
          #endif
          if ( file )
          {
            unsigned long next_top = filelen - /* keep last 90% */
                                   ((((unsigned long)(logfileLimit))<<10)*9)/10;
            if ( fseek( file, next_top, SEEK_SET ) == 0 &&
              ( msglen = fread( swapbuffer, sizeof( char ), maxswapsize,
                        file ) ) != 0 )
            {
              /* skip to the beginning of the next line */
              filelen = 0;
              while (filelen < (msglen-1))
              {     
                if (swapbuffer[filelen]=='\r' || swapbuffer[filelen]=='\n')
                {
                  while (filelen < (msglen-1) &&
                    (swapbuffer[filelen]=='\r' || swapbuffer[filelen]=='\n'))
                    filelen++;
                  next_top += filelen;
                  break;
                }
                filelen++;
              }
     
              filelen = 0;
              while ( fseek( file, next_top, SEEK_SET ) == 0 &&
                 ( msglen = fread( swapbuffer, sizeof( char ), maxswapsize,
                        file ) ) != 0 &&
                    fseek( file, filelen, SEEK_SET ) == 0 &&
                 ( msglen == fwrite( swapbuffer, sizeof( char ), msglen,
                         file ) ) )
              {
                next_top += msglen;
                filelen += msglen;
              }
              ftruncate( fileno( file ), filelen );
            }
            fclose( file );
            file = NULL;
          }
          free((void *)swapbuffer);
        }
      }
      logstatics.logfilestarted = 1;
    }
  }
  #endif
  else /* logfileType == LOGFILETYPE_NOLIMIT or no ftruncate() */
  {
    if (__get_logname( logfileType, logfileLimit, logname, sizeof(logname) ))
    {
      if (!logstatics.logstream)
        logstatics.logstream = __fopenlog( logname, "a" );
      if (logstatics.logstream)
      {
        fwrite( msgbuffer, sizeof( char ), msglen, logstatics.logstream );
        #if defined(__unix__) /* don't close it if /dev/tty* */
        if (isatty(fileno(logstatics.logstream)))
        {
          fflush( logstatics.logstream );
        }
        else
        #endif
        {
          fclose( logstatics.logstream );
          logstatics.logstream = NULL;
        }
      }
      logstatics.logfilestarted = 1;
    }
  }
  return;
}

// ------------------------------------------------------------------------

static void InternalLogMail( const char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  if ( msglen && logstatics.mailmessage && logstatics.spoolson &&
                                  (logstatics.loggingTo & LOGTO_MAIL) != 0)
  {
    static int recursion_check = 0; //network stuff sometimes Log()s.
    if ((++recursion_check) == 1)
      smtp_append_message( logstatics.mailmessage, msgbuffer );
    --recursion_check;
  }
  return;
}

// ------------------------------------------------------------------------

// On NT/Alpha (and maybe some other platforms) the va_list type is a struct,
// not an int or a pointer-type.  Hence, NULL is not a valid va_list.  Pass
// a (va_list *) instead to avoid this problem
void LogWithPointer( int loggingTo, const char *format, va_list *arglist )
{
  char msgbuffer[1024]; //min 1024!!, but also think of other OSs stack!!
  unsigned int msglen = 0, sel;
  char *buffptr, *obuffptr;
  int old_loggingTo = loggingTo;

  msgbuffer[0]=0;
  loggingTo &= (logstatics.loggingTo|LOGTO_RAWMODE);

  if ( !format || !*format )
    loggingTo = LOGTO_NONE;

  if ( loggingTo != LOGTO_NONE )
  {
    if ( arglist == NULL )
      strcat( msgbuffer, format );
    else
      vsprintf( msgbuffer, format, *arglist );
    msglen = strlen( msgbuffer );

    if ( msglen == 0 )
      loggingTo = LOGTO_NONE;
    else if (msgbuffer[msglen-1] != '\n')
      loggingTo &= LOGTO_SCREEN|LOGTO_RAWMODE;  //screen only obviously
  }

  if (loggingTo != LOGTO_NONE && logstatics.spoolson /* timestamps */ &&
      (old_loggingTo & LOGTO_RAWMODE) == 0 )
  {
    buffptr = &msgbuffer[0];
    sel = 1;
    do
    {
      while (*buffptr == '\r' || *buffptr=='\n' )
        buffptr++;
      if (*buffptr == ' ' || *buffptr == '\t')
      {
        obuffptr = buffptr;
        while (*obuffptr == ' ' || *obuffptr == '\t')
          obuffptr++;
        memmove( buffptr, obuffptr, strlen( obuffptr )+1 );
      }
      if (*buffptr && *buffptr!='[' && *buffptr!='\r' && *buffptr!='\n' )
      {
        const char *timestamp = CliGetTimeString( NULL, sel );
        memmove( buffptr+(strlen(timestamp)+3), buffptr, strlen( buffptr )+1 );
        *buffptr++=((sel)?('['):(' '));
        while (*timestamp)
          *buffptr++ = *timestamp++;
        *buffptr++=((sel)?(']'):(' '));
        *buffptr=' ';
      }
      sel = 0;
      while (*buffptr && *buffptr != '\n' && *buffptr != '\r')
        buffptr++;
    } while (*buffptr);
    msglen = strlen( msgbuffer );
  }

  if (logstatics.spoolson && (loggingTo & (LOGTO_FILE|LOGTO_MAIL)) != 0 )
  {
    sel = msglen;
    buffptr = &msgbuffer[0];
    while (sel > 0 && *buffptr == '\r')
    {
      buffptr++;
      sel--;
    }
    if (sel > 0)
    {
      if ((loggingTo & LOGTO_FILE) != 0 )
        InternalLogFile( buffptr, sel, 0 );
      if ((loggingTo & LOGTO_MAIL) != 0 )
        InternalLogMail( buffptr, sel, 0 );
    }
  }

  if (( loggingTo & LOGTO_SCREEN ) != 0)
  {
    int scrwidth = ASSUMED_SCREEN_WIDTH; /* assume this for consistancy */
    ConGetSize(&scrwidth,NULL); /* gets set to 80 or untouched, if not supported */

    #ifdef ASSERT_WIDTH_80  //"show" where badly formatted lines are cropping up
    //if (logstatics.stdoutisatty)
    {
      buffptr = &msgbuffer[0];
      do{
        while (*buffptr == '\r' || *buffptr == '\n' )
           buffptr++;
        obuffptr = buffptr;
        while (*buffptr && *buffptr != '\r' && *buffptr != '\n' )
          buffptr++;
        if ((buffptr-obuffptr) >= scrwidth)
        {
          if (scrwidth > 5)
          {
            obuffptr[(scrwidth-5)] = ' ';
            obuffptr[(scrwidth-4)] = '.';
            obuffptr[(scrwidth-3)] = '.';
            obuffptr[(scrwidth-2)] = '.';
          }
          memmove( obuffptr+(scrwidth-1), buffptr, strlen(buffptr)+1 );
          buffptr = obuffptr+(scrwidth-1);
        }
      } while (*buffptr);
      msglen = strlen( msgbuffer );
    }
    #endif

    buffptr = &msgbuffer[0];
    if ((loggingTo & LOGTO_RAWMODE)==0)
    {
      if (logstatics.stableflag) /* previous print ended with '\n'|'\r' */
      {
        if (*buffptr=='\n') /* remove extraneous leading '\n' */
        {
          msglen--;
          buffptr++;
        }
      }    
      else  /* a linefeed is pending */
      {
        if (*buffptr == '\r')  /* curr print expects to overwrites previous */
        {                      /* so ensure the old line is clear */
          memmove( &msgbuffer[scrwidth], msgbuffer, msglen+1 );
          msglen += scrwidth;
          memset( msgbuffer, ' ', scrwidth );
          msgbuffer[0] = '\r';
        }
        else if (*buffptr!='\n') /* curr print expects to be on a newline */
        {                        /* so ensure it is */
          msglen++;
          memmove( msgbuffer+1, msgbuffer, msglen );
          msgbuffer[0] = '\n';
          logstatics.stableflag = 1;
        }  
      }
    }
    if (msglen)
    {
      logstatics.lastwasperc = 0; //perc bar looks for this
      logstatics.stableflag = ( buffptr[(msglen-1)] == '\n' || 
                                buffptr[(msglen-1)] == '\r' );
      InternalLogScreen( buffptr, msglen, 0 );
    }
  }

  return;
}

// ------------------------------------------------------------------------

void LogFlush( int forceflush )
{
  if (( logstatics.loggingTo & LOGTO_SCREEN ) != 0)
  {
    //if ( logstatics.stableflag == 0 )
    //  LogWithPointer( LOGTO_SCREEN, "\n", NULL ); //LF if needed then fflush()
  }
  if (( logstatics.loggingTo & LOGTO_MAIL ) != 0)
  {
    if ( logstatics.mailmessage )
    {
      logstatics.loggingTo &= ~LOGTO_MAIL;
      if (forceflush)
        smtp_send_message( logstatics.mailmessage );
      else   
        smtp_send_if_needed( logstatics.mailmessage );
      logstatics.loggingTo |= LOGTO_MAIL;
    }
  }
  return;
}

// ------------------------------------------------------------------------

#if (CLIENT_OS == OS_WIN16) && defined(__WATCOMC__)
#define MAKE_VA_LIST_PTR(__va_list) ((va_list *)(&(__va_list[0])))
#else
#define MAKE_VA_LIST_PTR(__va_list) (&__va_list)
#endif

void LogScreen( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN, format, MAKE_VA_LIST_PTR(argptr) );
  va_end(argptr);
  return;
}

void LogScreenRaw( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_RAWMODE|LOGTO_SCREEN, format, MAKE_VA_LIST_PTR(argptr));
  va_end(argptr);
  return;
}

void Log( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN|LOGTO_FILE|LOGTO_MAIL, format, MAKE_VA_LIST_PTR(argptr));
  va_end(argptr);
  return;
}

void LogRaw( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_RAWMODE|LOGTO_SCREEN|LOGTO_FILE|LOGTO_MAIL, format, MAKE_VA_LIST_PTR(argptr));
  va_end(argptr);
  return;
}

void LogTo( int towhat, const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( towhat, format, MAKE_VA_LIST_PTR(argptr) );
  va_end(argptr);
  return;
}

/* return NULL if logging to file is implicitely or explicitely disabled,
** or "" if logging hasn't started yet, else ptr to buffer 
*/
const char *LogGetCurrentLogFilename(char *buffer, unsigned int buflen)
{
  if ((logstatics.loggingTo & LOGTO_FILE) == 0 ||
    logstatics.logfileType == LOGFILETYPE_NONE )
    return NULL;
  if ( !logstatics.logfilestarted )
    return "";
  return __get_logname( logstatics.logfileType, logstatics.logfileLimit,
                         buffer, buflen );
}

// ---------------------------------------------------------------------------

/* the following could be anywhere since it doesn't touch static data - */ 
/* it doesn't really belong in logstuff.cpp, but its currently only used */
/* from here (the matching public is LogGetContestLiveRate()), and has */
/* special handling when used from LogScreenPercent() */
static int __ContestGetLiveRate(unsigned int contest_i,
                                int called_from_logscreenpercent,
                                int *some_doneP,
                                u32 *ratehiP, u32 *rateloP,
                                u32 *walltime_hiP, u32 *walltime_loP,
                                u32 *coretime_hiP, u32 *coretime_loP)
{
  int probcount = -1;
  if (contest_i < CONTEST_COUNT)
  {
    int numprobs = ProblemCountLoaded(-1); /* total */
    if (numprobs > 0)
    {
      struct timeval tv;
      if (CliGetMonotonicClock(&tv) == 0)
      {            
        int numdone = 0;
        int prob_i, tab_sel = ((called_from_logscreenpercent)?(0):(1));
        u32 oldest_sec = 0, oldest_usec = 0;
        u32 tccount_hi = 0, tccount_lo = 0;
        u32 tctime_hi = 0,  tctime_lo = 0;
        u32 c_sec, c_usec;
        probcount = 0;

        for (prob_i = 0; prob_i < numprobs; prob_i++)
        {
          Problem *selprob = GetProblemPointerFromIndex(prob_i);
          if (selprob)
          { 
            int isinit = ProblemIsInitialized(selprob);
            if (!isinit)
              continue;
            if (isinit > 0) /* completed (any contest) */
              numdone++;
            if (selprob->pub_data.contest == contest_i)
            {
              u32 ccounthi, ccountlo;
              if (ProblemGetInfo(selprob, 0, 0,
                                 0, 0,
                                 0, 0, 
                                 0, 
                                 0, 0, 
                                 0, 
                                 0, 0, 
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 &ccounthi, &ccountlo, 0, 0,
                                 0, 0, 0, 0 ) >= 0)
              {
                int gotit = 0;
                u32 last_ccounthi = 0, last_ccountlo = 0; 
                u32 last_ctimehi = 0, last_ctimelo = 0; 
                u32 last_utimehi = 0, last_utimelo = 0; 
                c_sec = selprob->pub_data.runtime_sec;
                c_usec = selprob->pub_data.runtime_usec;
                if (selprob->pub_data.live_rate[tab_sel].init)
                {
                  last_ccounthi= selprob->pub_data.live_rate[tab_sel].ccounthi; 
                  last_ccountlo= selprob->pub_data.live_rate[tab_sel].ccountlo; 
                  last_ctimehi = selprob->pub_data.live_rate[tab_sel].ctimehi; 
                  last_ctimelo = selprob->pub_data.live_rate[tab_sel].ctimelo; 
                  last_utimehi = selprob->pub_data.live_rate[tab_sel].utimehi;   
                  last_utimelo = selprob->pub_data.live_rate[tab_sel].utimelo;   
                  gotit = 1;
                }
                selprob->pub_data.live_rate[tab_sel].ccounthi = ccounthi; 
                selprob->pub_data.live_rate[tab_sel].ccountlo = ccountlo; 
                selprob->pub_data.live_rate[tab_sel].ctimehi = c_sec; 
                selprob->pub_data.live_rate[tab_sel].ctimelo = c_usec; 
                selprob->pub_data.live_rate[tab_sel].utimehi = tv.tv_sec;   
                selprob->pub_data.live_rate[tab_sel].utimelo = tv.tv_usec;   
                selprob->pub_data.live_rate[tab_sel].init = 1;
                if (gotit)
                {
                  u32 temp = ccountlo;
                  ccountlo -= last_ccountlo;
                  if (ccountlo > temp)
                    ccounthi--;   
                  ccounthi -= last_ccounthi;  

                  tccount_hi += ccounthi;
                  temp = tccount_lo + ccountlo;
                  if (temp < tccount_lo)
                    tccount_hi++;
                  tccount_lo = temp;

                  if (probcount == 0 || last_utimehi < oldest_sec || 
                     (last_utimehi == oldest_sec && 
                      last_utimelo == oldest_usec))
                  {
                    oldest_sec = last_utimehi;
                    oldest_usec = last_utimelo;
                  }
                  if (c_usec < last_ctimelo)
                  {
                    c_sec--;
                    c_usec += 1000000;
                  }
                  tctime_hi += (c_sec - last_ctimehi);
                  tctime_lo += (c_usec - last_ctimelo); 
                  if (tctime_lo >= 1000000)
                  {
                    tctime_hi++;
                    tctime_lo -= 1000000;
                  }
                  probcount++;
                }
              } /* if ProblemGetInfo() */
            } /* if (selprob->pub_data.contest == contest_i) */
          } /* if (selprob) */
        } /* for (prob_i = 0; prob_i < numprobs; prob_i++) */
        if (probcount > 0)
        {
          c_sec = tv.tv_sec;
          c_usec = tv.tv_usec;
          if (c_sec < oldest_sec || 
             (c_sec == oldest_sec && c_usec < oldest_usec))
          {
            probcount = -1;
          }
          else 
          {
            if (c_usec < oldest_usec)
            {
              c_sec--;
              c_usec += 1000000;
            }
            c_sec -= oldest_sec;
            c_usec -= oldest_usec;
            if (some_doneP)
              *some_doneP = numdone;
            if (coretime_hiP)
              *coretime_hiP = tctime_hi;
            if (coretime_loP)
              *coretime_loP = tctime_lo;
            if (walltime_hiP)
              *walltime_hiP = c_sec;
            if (walltime_loP)
              *walltime_loP = c_usec;
            if (ratehiP || rateloP)
            {
              ProblemComputeRate( contest_i, c_sec, c_usec, 
                                  tccount_hi, tccount_lo, 
                                  ratehiP, rateloP, 0, 0);
            } /* if (ratehiP || rateloP) */
          } /* time is valid */
        } /* if (probcount > 0) */
      } /* if (CliGetMonotonicClock(&tv) == 0) */
    } /* if (numprobs > 0) */
  } /* if (contest_i < CONTEST_COUNT) */
  return probcount;
}

int LogGetContestLiveRate(unsigned int contest_i,
                          u32 *ratehiP, u32 *rateloP,
                          u32 *walltime_hiP, u32 *walltime_loP,
                          u32 *coretime_hiP, u32 *coretime_loP)
{
  return __ContestGetLiveRate(contest_i, 0, 0, ratehiP, rateloP, 
                              walltime_hiP, walltime_loP, 
                              coretime_hiP, coretime_loP );
}

// ---------------------------------------------------------------------------


//#define NO_PERCENTOMATIC_BATON

void LogScreenPercent( unsigned int load_problem_count )
{
  unsigned int percent, restartperc, endperc, prob_i, cont_i;
  unsigned int selprob_i = logstatics.perc_callcount % load_problem_count;
  char buffer[128]; unsigned char pbuf[52]; /* 'a'-'z','A'-'Z' */
  int disp_format, active_contests = 0;
  unsigned int prob_count[CONTEST_COUNT];

  if (!logstatics.crunchmeter || ( logstatics.loggingTo & LOGTO_SCREEN ) == 0 )
    return;

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
  {
    prob_count[cont_i] = ProblemCountLoaded(cont_i); /* -1=all contests */
    if (prob_count[cont_i]) 
      active_contests++;
  }
  if (active_contests == 0)
    return;

  #define DISPFORMAT_AUTO  -1
  #define DISPFORMAT_PERC   0
  #define DISPFORMAT_COUNT  1
  #define DISPFORMAT_RATE   2

  disp_format = DISPFORMAT_PERC;
  if (logstatics.stdoutisatty)
  {
    if (logstatics.crunchmeter == 1) /* absolute */
      disp_format = DISPFORMAT_COUNT; 
    else if (logstatics.crunchmeter == 3) /* rate */
      disp_format = DISPFORMAT_RATE;
    else if (logstatics.crunchmeter < 0 &&
     (prob_count[OGR] > 0 || load_problem_count >= sizeof(pbuf)))
      disp_format = DISPFORMAT_RATE; //or DISPFORMAT_AUTO for count;
    /* anything else is percent */
  }

  buffer[0] = '\0';
  endperc = restartperc = 0;

  if (disp_format == DISPFORMAT_RATE) /* live rate */
  {
    u32 ratehi, ratelo, wtimehi, wtimelo, ctimehi, ctimelo;
    int some_done; unsigned int i, x;
    cont_i = logstatics.perc_callcount % active_contests;

    for (x = 0, i = 0; x < CONTEST_COUNT; x++)
    {
      if (prob_count[x]) 
      {
        if (i == cont_i)
        {
          cont_i = x;
          break;
        }
        i++;
      }
    }
    if (__ContestGetLiveRate(cont_i, 1, /* <= YES, called from LogScreen... */
                             &some_done, &ratehi, &ratelo, 
                             &wtimehi, &wtimelo, &ctimehi, &ctimelo) < 1)
    {
      return; /* no rate available yet */
    }
    if (some_done)
    {
      endperc = 100;
    }
    else
    {
      i = sprintf(buffer,"\r%s: rate: ", CliGetContestNameFromID(cont_i));
      ProblemComputeRate( cont_i, 0, 0, ratehi, ratelo, 0, 0,
                          &buffer[i], sizeof(buffer)-i );
      strcat(buffer, "/sec");
#if 0
      //if (CliGetThreadUserTime(0)==0) /* thread time supported */
      { 
        unsigned long efficiency = 0;
        wtimelo = (wtimelo / 1000)+(wtimehi * 1000);
        ctimelo = ((ctimelo+499) / 1000)+(ctimehi * 1000);
        if (wtimelo)
        {
          /* note that efficiency can be greater than 100% 
          ** due to scheduler hickups or SMP or OS's clock rounding.
          ** This is perfectly normal or all OSs.
          */
          unsigned long effmax = GetNumberOfDetectedProcessors()*1000;
          efficiency = (((unsigned long)ctimelo) * 1000ul)/wtimelo;
          if (efficiency > effmax)
            efficiency = effmax;
        }
        sprintf(&buffer[strlen(buffer)], " (%lu.%01lu%% efficient)", 
                                          efficiency/10, efficiency%10);
      }
      #if 0 /* is this useful/meaningful? */
      else 
      {
        unsigned long benchrate = BenchGetBestRate(cont_i);
        int bestperm = -1;
        if (benchrate)
        {
          #if (ULONG_MAX > 0xfffffffful)
          unsigned long r = (((unsigned long)ratehi)<<32)+ratelo)*100;
          bestperm = (int)(r/benchrate);
          #else
          if (!ratehi)
          {
            unsigned long r = ratelo * 100;
            if (r < ratelo)
            {
              r = ratelo;
              benchrate /= 100;
            }
            bestperm = (int)(r/benchrate);
          }
          #endif
        }
        if (bestperm >= 0)
        {
          sprintf(&buffer[strlen(buffer)], " (%d.%02d%%)", 
                                            bestperm/100, bestperm%100);
        }
      }    
      #endif
#endif
    }   
  }
  else
  {
    for (prob_i = 0; prob_i < load_problem_count; prob_i++)
    {
      Problem *selprob = GetProblemPointerFromIndex(prob_i);
      pbuf[prob_i] = 0;
      if (selprob)
      {
        unsigned int permille = 0, startpermille = 0;  int girc = -1;
        cont_i = 0;
 
        if (disp_format != DISPFORMAT_PERC &&
            (!buffer[0] || prob_i == selprob_i))
        {
          char blkdone[32], blksig[32]; const char *contname;
          girc = ProblemGetInfo(selprob, &cont_i, &contname, 0, 0, 0, 0, 0, 
                           &permille, &startpermille, 0,
                           blksig, sizeof(blksig), 0, 0, 0, 0, 0,0,0, 0,0,0,
                           0, 0, 0, 0, blkdone, sizeof(blkdone) );
          if (permille == 1000 && disp_format == DISPFORMAT_AUTO)
            disp_format = DISPFORMAT_PERC;
          else if (girc != -1)
          {
            sprintf(buffer, "#%u: %s:%s [%s]", 
                    prob_i+1, contname, blksig, blkdone );
            //there isn't enough space for percent so don't even think about it
          }
        }
        else
        {
          girc = ProblemGetInfo(selprob, &cont_i, 0, 0, 0, 0, 0, 0,
                                         &permille, &startpermille, 0,
                                         0, 0, 0, 0, 0, 0, 
                                         0,0,0, 0,0,0, 0, 0, 0, 0, 0, 0 );
        }
        if (girc != -1)
        {
          percent = (permille+((permille < 995)?(5):(0)))/10;
          if (load_problem_count == 1 && percent != 100)
          {   /* don't do 'R' if multiple-problems */
            restartperc = (startpermille)/10;
            restartperc = (!restartperc || percent == 100) ? 0 :
                ( restartperc - ((restartperc > 90) ? (restartperc & 1) :
                (1 - (restartperc & 1))) );
          }
          if (percent > endperc)
          {
            endperc = percent;
            if (endperc == 100 && disp_format == DISPFORMAT_AUTO)
              disp_format = DISPFORMAT_PERC;
          }
          if (percent && ((percent>90)?((percent&1)!=0):((percent&1)==0)))
          {
            percent--;  /* make sure that it is visible */
          }
          if (prob_i < sizeof(pbuf)) /* a-z,A-Z */
          {
            pbuf[prob_i] = (unsigned char)(percent);
            prob_count[cont_i]++;
          }
        }
      }      
    }
  }

  if (buffer[0] && disp_format != DISPFORMAT_PERC)
  {
    LogScreen( "\r%s", buffer, NULL );
    logstatics.stableflag = 0; //(endperc == 0);  //cursor is not at column 0
    logstatics.lastwasperc = 1; //(endperc != 0); //percbar requires reset
    /* simple, eh? :) */
  }
  else
  {
    char *bufptr = &buffer[0];
    unsigned int multiperc = 0;

    if (!logstatics.lastwasperc || logstatics.stdoutisatty)
      logstatics.lastperc_done = 0;

    multiperc = 0;
    if (load_problem_count > 1 && logstatics.lastperc_done == 0 && 
        logstatics.stdoutisatty && endperc < 100)
    {
      multiperc = load_problem_count;
      if (multiperc > sizeof(pbuf))
        multiperc = sizeof(pbuf);
    }      

    if (logstatics.lastperc_done==0 && endperc > 0 && logstatics.stdoutisatty)
      *bufptr++ = '\r';

    percent = logstatics.lastperc_done+1;
    logstatics.lastperc_done = endperc;    
    for (; percent <= endperc; percent++)
    {
      if ( percent >= 100 )
      { strcpy( bufptr, "100" ); bufptr+=sizeof("100"); /*endperc=0;*/ break;}
      else if ( ( percent % 10 ) == 0 )
      { sprintf( bufptr, "%d%%", (int)(percent) ); bufptr+=3; }
      else if ( restartperc == percent)
      { *bufptr++='R'; }
      else if (((percent&1)?(percent<90):(percent>90)))
      {
        char ch = '.';
        #if (CLIENT_OS == OS_OS2)
        ch = ((percent+1 >= endperc)?(219):(176)); /* oooh! fanschy! */
        #endif
        if (multiperc)
        {
          unsigned int equals = 0;
          /* multiperc is min(load_problem_count,sizeof(pbuf)) */
          for ( prob_i=0; prob_i < multiperc; prob_i++ )
          {
            if ( pbuf[prob_i] == (unsigned char)(percent) )
            {
              ch = (char)('a'+prob_i);
              if (ch > 'z')
                ch = (char)('A'+(prob_i-('z'-'a')));
              if ( (++equals)>selprob_i )
                break;
            }    
          }
        }
        *bufptr++ = ch;
      }
    }
    #ifndef NO_PERCENTOMATIC_BATON
    if (endperc < 100 && logstatics.percbaton)
    { /* implies conistty and !gui (window repaints are _expensive_) */
      static const char batonchars[] = {'|','/','-','\\'};
      if (bufptr == &buffer[0]) /* didn't prepend '\r' */
        *bufptr++ = '\r';       /* so do it now */
      *bufptr++=(char)batonchars[logstatics.perc_callcount%sizeof(batonchars)];
    }  
    #endif
    *bufptr = '\0';

    if ( (buffer[0]==0) || (buffer[0]=='\r' && buffer[1]==0) )
      ;
    else
    {
      static char lastbuffer[sizeof(buffer)] = {0};
      if (!logstatics.lastwasperc || strcmp( lastbuffer, buffer ) != 0)
      {
        strcpy( lastbuffer, buffer );
        LogWithPointer( LOGTO_SCREEN|LOGTO_RAWMODE, buffer, NULL );
        logstatics.stableflag = 0; //(endperc == 0);  //cursor is not at column 0
        logstatics.lastwasperc = 1; //(endperc != 0); //percbar requires reset
      }
    }
  } /* percent based dotdotdot */
  logstatics.perc_callcount++;
  return;
}

// ------------------------------------------------------------------------

void DeinitializeLogging(void)
{
  if (logstatics.mailmessage)
  {
    void *mailmessage = logstatics.mailmessage;
    logstatics.mailmessage = NULL;
    if ((logstatics.loggingTo & LOGTO_MAIL)!=0)
    {
      logstatics.loggingTo &= ~LOGTO_MAIL;
      smtp_send_message( mailmessage );
      smtp_clear_message( mailmessage );
    }  
    smtp_destruct_message( mailmessage );
  }
  if ( logstatics.logstream )
  {
    fclose( logstatics.logstream );
    logstatics.logstream = NULL;
  }
  memset((void *)&logstatics, 0, sizeof(logstatics));
  logstatics.loggingTo = LOGTO_NONE;
  logstatics.logfileType = LOGFILETYPE_NONE;
  return;
}

// ---------------------------------------------------------------------------

static int fixup_logfilevars( const char *stype, const char *slimit,
                              int *type, unsigned int *limit,
                              const char *userslogname, 
                              char *logname, unsigned int maxlognamelen,
                              char *logbasedir, unsigned int maxlogdirlen )
{
  unsigned int len;
  int climit = 0;
  char scratch[20];
  unsigned long l = 0;

  *type = LOGFILETYPE_NOLIMIT;
  *limit = 0;
  *logname = 0;
  *logbasedir = 0;

  if (userslogname)
  {
    while (*userslogname && isspace(*userslogname))
      userslogname++;
    strncpy( logname, userslogname, maxlognamelen );
    logname[maxlognamelen-1]='\0';
    len = strlen( logname );
    while (len > 1 && isspace(logname[len-1]))
      logname[--len]='\0';
    if (strcmp( logname, "none" )==0)
    {
      *logname='\0';
      *type = LOGFILETYPE_NONE;
      return 0;
    }
  }

  /* generate a basedir if we're going to be needing it */
  if (!*logname || strcmp(GetFullPathForFilename(logname),logname)!=0)
  {
    /* get dir with trailing dir separator. returns NULL if buf is too small*/
    if (!GetWorkingDirectory( logbasedir, maxlogdirlen ))
      logbasedir[0] = '\0';
  }
  TRACE_OUT((0,"log file = '%s', basedir='%s'\n", logname, logbasedir));

  if (!slimit)
  {
    slimit = "";
    climit = 0;
  }
  else
  {
    while (*slimit && isspace(*slimit))
      slimit++;
    while (isdigit(*slimit))
      l=((l*10)+(*slimit++)-'0');
    while (*slimit && isspace(*slimit))
      slimit++;
    climit = tolower(*slimit++);
  }
  if (stype)
  {
    static struct { int type; const char *name; } logtypelist[] = {
                   //nolimit is default, so we don't need it here.
                  { LOGFILETYPE_NONE,    "none"   },
                  { LOGFILETYPE_ROTATE,  "rotate" },
                  { LOGFILETYPE_RESTART, "restart"},
                  { LOGFILETYPE_FIFO,    "fifo"   } };
    unsigned int i = 0;
    while (*stype && isspace(*stype))
      stype++;
    len = strlen(stype);
    while (len>1 && isspace(stype[len-1]))
      len--;
    while (i<len && i<(sizeof(scratch)-1))
      scratch[i++]=(char)tolower(*stype++);
    scratch[i]='\0';

    for (i=0;i<(sizeof(logtypelist)/sizeof(logtypelist[0]));i++)
    {
      if (strcmp(logtypelist[i].name, scratch ) == 0)
      {
        *type = logtypelist[i].type;
        break;
      }
    }
  }

  if (*type == LOGFILETYPE_ROTATE)
  {
    TRACE_OUT((0,"log type: LOGFILETYPE_ROTATE\n"));
    if (l == 0)
      l++;
    /* convert to days */
    if (climit == 'm')
      l *= 30;
    else if (climit == 'w')
      l *= 7;
    else if (climit == 'y' || (climit == 'a' && (*slimit=='n' || *slimit=='N')))
      l *= 365;
    if (l > INT_MAX)
      l = INT_MAX;
    *limit = (unsigned int)l;
  }
  else if (*logname == '\0' || *type == LOGFILETYPE_NONE)
  {
    TRACE_OUT((0,"log type: LOGFILETYPE_NONE\n"));
    *type = LOGFILETYPE_NONE;
    *limit = 0;
  }
  else if (*type == LOGFILETYPE_RESTART || *type == LOGFILETYPE_FIFO)
  {
    TRACE_OUT((0,"log type: LOGFILETYPE_RESTART/FIFO\n"));
    /* convert to Kb */
    if (climit == 'g') /* dickheads! */
    {
      *type = LOGFILETYPE_NOLIMIT;
      *limit = 0;
    }
    else
    {
      if (climit == 'b')
        l /= 1024;
      else if (climit == 'm')
        l *= 1024;
      if (l == 0)
        l++;
      if (l > INT_MAX)
        l = INT_MAX;
      *limit = (unsigned int)l;
    }
  }
  else //if (*type == LOGFILETYPE_NOLIMIT)
  {
    TRACE_OUT((0,"log type: LOGFILETYPE_NOLIMIT\n"));
    *limit = 0; /* limit is ignored */
  }
  TRACE_OUT((0,"logfile limit = %u\n", *limit));
  return 0;
}

void InitializeLogging( int noscreen, int crunchmeterstyle, int nopercbaton,
                        const char *logfilename,
                        const char *logfiletype, const char *logfilelimit,
                        long mailmsglen, const char *smtpsrvr,
                        unsigned int smtpport, const char *smtpfrom,
                        const char *smtpdest, const char *id )
{
  DeinitializeLogging();
  logstatics.spoolson = 1;
  logstatics.stdoutisatty = ConIsScreen();
  logstatics.crunchmeter = crunchmeterstyle;
  logstatics.percbaton = (crunchmeterstyle != 0 && nopercbaton == 0 && 
                          !ConIsGUI() && logstatics.stdoutisatty);
                         /* baton not for macos or win GUI */

  if ( noscreen == 0 )
  {
    logstatics.loggingTo |= LOGTO_SCREEN;
    logstatics.stableflag = 0;   //assume next log screen needs a '\n' first
  }

  fixup_logfilevars( logfiletype, logfilelimit,
                     &logstatics.logfileType, &logstatics.logfileLimit,
                     logfilename, 
                     logstatics.logfile, sizeof(logstatics.logfile),
                     logstatics.basedir, sizeof(logstatics.basedir));
  if (logstatics.logfileType != LOGFILETYPE_NONE)
  {
    logstatics.loggingTo |= LOGTO_FILE;
  }
  if (mailmsglen > 0)
  {
    logstatics.mailmessage = smtp_construct_message( mailmsglen, 
                                                     smtpsrvr, smtpport,
                                                     smtpfrom, smtpdest, id );
    if (logstatics.mailmessage) /* initialized ok */
    {
      logstatics.loggingTo |= LOGTO_MAIL;
    }
  }  
  return;
}

// ---------------------------------------------------------------------------

