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
return "@(#)$Id: logstuff.cpp,v 1.37.2.29 2000/09/24 23:31:08 andreasb Exp $"; }

#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "mail.h"      // MailMessage
#include "clitime.h"   // CliGetTimeString(NULL,1)
#include "pathwork.h"  // GetFullPathForFilename(), GetWorkingDirectory()
#include "problem.h"   // Problem object for logscreenpercent
#include "probman.h"   // GetProblemPointerFromIndex() for LogScreenPercent
#include "console.h"   // for ConOut() and ConIsScreen()
#include "util.h"      // TRACE
#include "logstuff.h"  // keep the prototypes in sync

//-------------------------------------------------------------------------

#if 0 /* logstuff.h defines */

#define LOGFILETYPE_NONE    0 //
#define LOGFILETYPE_NOLIMIT 1 //unlimited (or limit == -1)
#define LOGFILETYPE_RESTART 2 //then logLimit is in KByte
#define LOGFILETYPE_FIFO    3 //then logLimit is in KByte (minimum 100K)
#define LOGFILETYPE_ROTATE  4 //then logLimit is in days

#define LOGTO_NONE       0x00
#define LOGTO_SCREEN     0x01
#define LOGTO_FILE       0x02
#define LOGTO_MAIL       0x04

#define LOGTO_PERCENTBAR 0x40  // use together with _SCREEN and _RAWMODE
#define LOGTO_RAWMODE    0x80
#define LOGTO_ALLMODES   (LOGTO_RAWMODE | LOGTO_PERCENTBAR)

#define MAX_LOGENTRY_LEN 1024 //don't make this smaller than 1K!

#define ASSERT_WIDTH_80     //show where badly formatted lines are cropping up

#endif /* logstuff.h defines */

#define ASSUMED_SCREEN_WIDTH 80
// ========================================================================

static struct
{
  int initlevel;
  char loggingTo;            // LOGTO_xxx bitfields
  char spoolson;             // mail/file logging and time stamping is on/off.
  char percprint;            // percentprinting is enabled
  char percbaton;            // percent baton is enabled

  void *mailmessage;         //handle returned from smtp_construct_message()
  char basedir[256];         //filled if user's 'logfile' is not qualified
  char logfile[128+20];      //fname when LOGFILETYPE_RESTART or _FIFO
                             //lastused fname when LOGFILETYPE_ROTATE
  FILE *logstream;           //open logfile

  unsigned int logfilebaselen;//len of the log fname without ROTATE suffix
  int  logfileType;          //rotate, restart, fifo, none
  unsigned int logfileLimit; //days when rotating or kbyte when fifo/restart
  unsigned int logfilestarted; // 1 after the first logfile write

  char stdoutisatty;         //log screen can handle lines not ending in '\n'
  char stableflag;           //last log screen did end in '\n'
  int last_screen_flags;     //flags used in last screenlog (RAWMODE, PERCENTBAR)

} logstatics = {
  0,      // initlevel
  LOGTO_NONE,   // loggingTo
  0,      // spoolson
  0,      // percprint
  0,      // percbaton
  NULL,   // *mailmessage
  {0},    // basedir[]
  {0},    // logfile[]
  NULL,   // logstream
  0,      // logfilebaselen
  LOGFILETYPE_NONE, // logfileType
  0,      // logfileLimit
  0,      // logfilestarted
  0,      // stdoutisatty
  0,      // stableflag
  0       // last_screen_flags
};

// ========================================================================

static void InternalLogScreen( const char *msgbuffer, unsigned int msglen, int loggingMode )
{
  if ((logstatics.loggingTo & LOGTO_SCREEN) != 0)
  {
    if ( msglen && (msgbuffer[msglen-1] == '\n' || ConIsScreen() ) )
    {
      if (strlen( msgbuffer ) == msglen) //we don't do binary data
      {                                  //which shouldn't happen anyway.
        if (*msgbuffer == '\r' && (loggingMode & LOGTO_PERCENTBAR) == 0)
        {
          const char *p;
          char blankline[ASSUMED_SCREEN_WIDTH+1];
          memset(blankline, ' ' ,sizeof(blankline));
          blankline[sizeof(blankline)-1] = '\0';
          blankline[0] = '\r';
          p = strchr(msgbuffer, '\n');
          if (!p && (loggingMode & LOGTO_RAWMODE) == 0)
          {
            // Cursor position won't be exact: right edge of screen instead of
            // end of message. But saves some (sometimes expensive) redraws.
            p = strchr(msgbuffer, '\0');
          }
          if (p && (((unsigned int)(p-msgbuffer))<sizeof(blankline)-1))
          {
            memcpy(blankline, msgbuffer, (p-msgbuffer));
            msgbuffer = p;
          }  
          ConOut( blankline );
        }  
        if (*msgbuffer)
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

//this can ONLY be called from LogWithPointer.
static void InternalLogFile( const char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  #if (CLIENT_OS == OS_NETWARE || CLIENT_OS == OS_DOS || \
       CLIENT_OS == OS_OS2 || CLIENT_OS == OS_WIN16 || \
       CLIENT_OS == OS_WIN32 || CLIENT_OS == OS_WIN32S )
    #define ftruncate( fd, sz )  chsize( fd, sz )
  #elif (CLIENT_OS == OS_VMS || CLIENT_OS == OS_AMIGAOS)
    #define ftruncate( fd, sz ) //nada, not supported
    #define FTRUNCATE_NOT_SUPPORTED
  #endif

  int logfileType = logstatics.logfileType;
  unsigned int logfileLimit = logstatics.logfileLimit;

  if ( !msglen || msgbuffer[msglen-1] != '\n')
    return;
  if ( logfileType == LOGFILETYPE_NONE || logstatics.spoolson==0 ||
       (logstatics.loggingTo & LOGTO_FILE) == 0)
    return;

  if ( logfileType == LOGFILETYPE_RESTART)
  {
    if ( logstatics.logfile[0] == 0 )
      return;
    if ( logfileLimit < 1 ) /* no size specified */
      return;
    if (!logstatics.logstream)
    {
      logstatics.logstream = __fopenlog( logstatics.logfile, "a" );
    }
    if ( logstatics.logstream )
    {
      long filelen;
      fseek( logstatics.logstream, 0, SEEK_END );
      filelen = (long)ftell( logstatics.logstream );
      if (filelen != (long)(-1))
      {
        //filelen += msglen;
        if ( ((unsigned int)(filelen >> 10)) > logfileLimit )
        {
          int truncated = 1;
          fclose( logstatics.logstream );
          logstatics.logstream = __fopenlog( logstatics.logfile, "w" );
          if (logstatics.logstream && truncated)
          {
            fprintf( logstatics.logstream,
               "[%s] Log file exceeded %uKbyte limit. Restarted...\n\n",
                CliGetTimeString( NULL, 1 ),
               (unsigned int)( logstatics.logfileLimit ));
          }
        }
      }
      if (logstatics.logstream)
      {
        logstatics.logfilestarted = 1;
        fwrite( msgbuffer, sizeof( char ), msglen, logstatics.logstream );
        fclose( logstatics.logstream );
        logstatics.logstream = NULL;
      }
    }
  }
  else if ( logfileType == LOGFILETYPE_ROTATE )
  {
    static unsigned logfilebaselen = 0;
    static unsigned int last_year = 0, last_mon = 0, last_day = 0;
    time_t ttime = time(NULL);
    struct tm *currtmP = localtime( &ttime );
    int abortwrite = 0;

    if (!logstatics.logfilestarted)
      logfilebaselen = strlen( logstatics.logfile );

    if ( logfileLimit >= 28 && logfileLimit <= 31 ) /* monthly */
    {
      if (currtmP != NULL)
      {
        last_year = currtmP->tm_year+ 1900;
        last_mon  = currtmP->tm_mon + 1;
      }
      if (last_mon == 0)
        abortwrite = 1;
      else if (last_mon <= 12)
      {
        static const char *monnames[] = { "jan","feb","mar","apr","may","jun",
                                          "jul","aug","sep","oct","nov","dec" };
        sprintf( logstatics.logfile+logfilebaselen,
                 "%02d%s"EXTN_SEP"log", (int)(last_year%100),
                  monnames[last_mon-1] );
      }
    }
    else if (logfileLimit == 365 ) /* annually */
    {
      if (currtmP != NULL)
        last_year = currtmP->tm_year+ 1900;
      if (last_year == 0)
        abortwrite = 1;
      else
      {
        sprintf( logstatics.logfile+logfilebaselen,
                 "%04d"EXTN_SEP"log", (int)last_year );
      }
    }
    else if (logfileLimit == 7 ) /* weekly */
    {
      /* technically, week 1 is the first week with a monday in it.
         But we don't care about that here.                 - cyp
      */
      if (currtmP != NULL)
      {
        last_year = currtmP->tm_year + 1900;
        last_day  = currtmP->tm_yday + 1; /* note */
      }
      if (last_day == 0)
        abortwrite = 1;
      else
      {
        sprintf( logstatics.logfile+logfilebaselen,
                 "%02dw%02d"EXTN_SEP"log", (int)(last_year%100),
                  ((last_day+6)/7) );
      }
    }
    else /* anything else: daily = 1, fortnightly = 14 etc */
    {
      if (currtmP != NULL )
      {
        static unsigned long last_jdn = 0;
        unsigned int curr_year = currtmP->tm_year+ 1900;
        unsigned int curr_mon  = currtmP->tm_mon + 1;
        unsigned int curr_day  = currtmP->tm_mday;
        #define _jdn( y, m, d ) ( (long)((d) - 32076) + 1461L * \
          ((y) + 4800L + ((m) - 14) / 12) / 4 + 367 * \
          ((m) - 2 - ((m) - 14) / 12 * 12) / 12 - 3 * \
          (((y) + 4900L + ((m) - 14) / 12) / 100) / 4 + 1 )
        unsigned long curr_jdn = _jdn( curr_year, curr_mon, curr_day );
        #undef _jdn
        if (( curr_jdn - last_jdn ) > logfileLimit )
        {
          last_jdn  = curr_jdn;
          last_year = curr_year;
          last_mon  = curr_mon;
          last_day  = curr_day;
        }
      }
      if (last_day == 0)
        abortwrite = 1;
      else
      {
        sprintf( logstatics.logfile+logfilebaselen,
               "%02d%02d%02d"EXTN_SEP"log", (int)(last_year%100),
               (int)(last_mon), (int)(last_day) );
      }
    }
    if (!abortwrite)
    {
      static char lastfilename[sizeof(logstatics.logfile)] = {0};
      if (logstatics.logstream)
      {
        if ( strcmp( lastfilename, logstatics.logfile ) != 0 )
        {
          fclose( logstatics.logstream );
          logstatics.logstream = NULL;
        }
      }
      if (!logstatics.logstream)
      {
        strcpy( lastfilename, logstatics.logfile );
        logstatics.logstream = __fopenlog( logstatics.logfile, "a" );
      }
      if ( logstatics.logstream )
      {
        fwrite( msgbuffer, sizeof( char ), msglen, logstatics.logstream );
        fclose( logstatics.logstream );
        logstatics.logstream = NULL;
      }
      logstatics.logfilestarted = 1;
    }
  }
  #ifndef FTRUNCATE_NOT_SUPPORTED
  else if ( logfileType == LOGFILETYPE_FIFO )
  {
    unsigned long filelen = 0;
    if ( logstatics.logfile[0] == 0 )
      return;
    if ( logfileLimit < 100 )
      logfileLimit = 100;

    if (!logstatics.logstream)
    {
      logstatics.logstream = __fopenlog( logstatics.logfile, "a" );
    }
    if ( logstatics.logstream )
    {
      fwrite( msgbuffer, sizeof( char ), msglen, logstatics.logstream );
      if (((long)(filelen = ftell( logstatics.logstream ))) == ((long)(-1)))
        filelen = 0;
      fclose( logstatics.logstream );
      logstatics.logstream = NULL;
    }
    if ( filelen > (((unsigned long)(logfileLimit))<<10) )
    {    /* careful: file must be read/written without translation - cyp */
      unsigned int maxswapsize = 1024*4; //assumed dpage/sector size
      char *swapbuffer = (char *)malloc( maxswapsize );
      if (swapbuffer)
      {
        if (!logstatics.logstream)
          logstatics.logstream = __fopenlog( logstatics.logfile, "r+b" );
        if ( logstatics.logstream )
        {
          unsigned long next_top = filelen - /* keep last 90% */
                                 ((((unsigned long)(logfileLimit))<<10)*9)/10;
          if ( fseek( logstatics.logstream, next_top, SEEK_SET ) == 0 &&
            ( msglen = fread( swapbuffer, sizeof( char ), maxswapsize,
                      logstatics.logstream ) ) != 0 )
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
            while ( fseek( logstatics.logstream, next_top, SEEK_SET ) == 0 &&
               ( msglen = fread( swapbuffer, sizeof( char ), maxswapsize,
                      logstatics.logstream ) ) != 0 &&
                  fseek( logstatics.logstream, filelen, SEEK_SET ) == 0 &&
               ( msglen == fwrite( swapbuffer, sizeof( char ), msglen,
                       logstatics.logstream ) ) )
            {
              next_top += msglen;
              filelen += msglen;
            }
            ftruncate( fileno( logstatics.logstream ), filelen );
          }
          fclose( logstatics.logstream );
          logstatics.logstream = NULL;
        }
        free((void *)swapbuffer);
      }
    }
    logstatics.logfilestarted = 1;
  }
  #endif
  else /* if ( logfileType == LOGFILETYPE_NOLIMIT ) */
  {
    if ( logstatics.logfile[0] == 0 )
      return;
    if (!logstatics.logstream)
      logstatics.logstream = __fopenlog( logstatics.logfile, "a" );
    if (logstatics.logstream)
    {
      fwrite( msgbuffer, sizeof( char ), msglen, logstatics.logstream );
      #if defined(__unix__) /* don't close it if /dev/tty* */
      if (isatty(fileno(logstatics.logstream)))
        fflush( logstatics.logstream );
      else
      #endif
      {
        fclose( logstatics.logstream );
        logstatics.logstream = NULL;
      }
    }
    logstatics.logfilestarted = 1;
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
  #if ((!defined(MAX_LOGENTRY_LEN)) || (MAX_LOGENTRY_LEN < 1024))
    #ifdef MAX_LOGENTRY_LEN
    #undef MAX_LOGENTRY_LEN
    #endif
    #define MAX_LOGENTRY_LEN 1024
  #endif
  char msgbuffer[MAX_LOGENTRY_LEN];
  unsigned int msglen = 0, sel;
  char *buffptr, *obuffptr;
  const char *timestamp;
  int loggingMode = loggingTo & LOGTO_ALLMODES;

  msgbuffer[0]=0;
  loggingTo &= logstatics.loggingTo;

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
      loggingTo &= LOGTO_SCREEN;  //screen only obviously
  }

  // add timestamp/indent following lines if (!RAWMODE)
  if (loggingTo != LOGTO_NONE && logstatics.spoolson /* timestamps */ &&
      (loggingMode & LOGTO_RAWMODE) == 0 )
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
        timestamp = CliGetTimeString( NULL, sel );
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

  // log to file, mail
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

  // truncate lines and log to screen
  if (( loggingTo & LOGTO_SCREEN ) != 0)
  {
    #ifdef ASSERT_WIDTH_80  //"show" where badly formatted lines are cropping up
    //if (ConIsScreen())
    {
      // FIXME: !logstatics.stableflag and LogScreenRaw() can produce too long lines
      int scrwidth = ASSUMED_SCREEN_WIDTH; /* assume this for consistancy */
      buffptr = &msgbuffer[0];
      do 
      {
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
    }
    #endif

    buffptr = &msgbuffer[0];
    msglen = strlen( msgbuffer );
    if (logstatics.stableflag)      /* if cursor is in first column */
    {
      while (*buffptr == '\r')      /* remove all leading '\r' */
      {
        buffptr++;
        msglen--;
      }
      if ((loggingMode & LOGTO_RAWMODE) == 0 && *buffptr == '\n')
      {                             /* and the first '\n' if !RAWMODE */
        buffptr++;
        msglen--;
        
        while (*buffptr == '\r')    /* remove all leading '\r' */
        {
          buffptr++;
          msglen--;
        }
      }
    }
    else if (*buffptr == '\n')
    {
      //logstatics.stableflag = 1;
    }
    else if ((loggingMode != logstatics.last_screen_flags) || 
             ((loggingMode & LOGTO_RAWMODE) == 0 && *buffptr != '\r'))
      // cursor is not in first column and message has no leading '\n'
      // modes have changed or concatenation in !RAWMODE not permitted 
      //   -> add a newline before the message
    {
      msglen++;
      memmove( msgbuffer+1, msgbuffer, msglen );
      msgbuffer[0] = '\n';
      //logstatics.stableflag = 1;
    }

    if (msglen > 0)
    {
      InternalLogScreen( buffptr, msglen, loggingMode);
      logstatics.stableflag = ( buffptr[(msglen-1)] == '\n' );
      logstatics.last_screen_flags = loggingMode;
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

const char *LogGetCurrentLogFilename( void )
{
  if ((logstatics.loggingTo & LOGTO_FILE) == 0 ||
    logstatics.logfileType == LOGFILETYPE_NONE )
    return NULL;
  if ( !logstatics.logfilestarted )
    return "";
  return logstatics.logfile;
}

// ---------------------------------------------------------------------------

//#define NO_PERCENTOMATIC_BATON

void LogScreenPercent( unsigned int load_problem_count )
{
  static unsigned int displevel = 0, lastperc = 0, batonpos = 0;
  unsigned int percent, restartperc, endperc, equals, prob_i;
  int istty, multiperc;
  char ch; char batonchar; char buffer[88];
  char *bufptr; unsigned char pbuf[30]; /* 'a'-'z' */

  if (!logstatics.percprint || ( logstatics.loggingTo & LOGTO_SCREEN ) == 0 )
    return;

  istty  = ConIsScreen();
  endperc = restartperc = 0;

  for (prob_i = 0; prob_i < load_problem_count; prob_i++)
  {
    Problem *selprob = GetProblemPointerFromIndex(prob_i);
    percent = 0;

    if (selprob && selprob->IsInitialized())
    {
      unsigned int permille = selprob->CalcPermille();
      if (permille == 0)
        permille = selprob->startpermille;
      if (permille < 995) /* only round up if < 99.5 */
        permille += 5;
      percent = permille/10;
      if (load_problem_count == 1 && percent != 100)
      {   /* don't do 'R' if multiple-problems */
        restartperc = (selprob->startpermille)/10;
        restartperc = (!restartperc || percent == 100) ? 0 :
            ( restartperc - ((restartperc > 90) ? (restartperc & 1) :
            (1 - (restartperc & 1))) );
      }
      if (percent > endperc)
          endperc = percent;
      if (percent && ((percent>90)?((percent&1)!=0):((percent&1)==0)))
        percent--;  /* make sure that it is visible */
    }
    if (load_problem_count <= 26) /* a-z */
      pbuf[prob_i] = (unsigned char)(percent);
  }

  if ((logstatics.last_screen_flags & LOGTO_PERCENTBAR) == 0 || istty)
    lastperc = 0;
  multiperc = (load_problem_count > 1 && load_problem_count <= 26 /*a-z*/
                 && lastperc == 0 && istty && endperc < 100);

  batonchar = 0;                 
  bufptr = &buffer[0];
  if (lastperc == 0 && endperc > 0 && istty )
  {
    #ifndef NO_PERCENTOMATIC_BATON
    if (endperc < 100 && logstatics.percbaton && !ConIsGUI()) 
    {/* not for macos or win GUI - window repaints are _expensive_ */
      const char batonchars[4] = {'|','/','-','\\'};
      batonchar = batonchars[(++batonpos) % sizeof(batonchars)];
    }  
    #endif
    *bufptr++ = '\r';
  }  
  for (percent = lastperc+1; percent <= endperc; percent++)
  {
    if ( percent >= 100 )
    { strcpy( bufptr, "100" ); bufptr+=sizeof("100"); /*endperc=0;*/ break;}
    else if ( ( percent % 10 ) == 0 )
    { sprintf( bufptr, "%d%%", (int)(percent) ); bufptr+=3; }
    else if ( restartperc == percent)
    { *bufptr++='R'; }
    else if (((percent&1)?(percent<90):(percent>90)))
    {
    #if (CLIENT_OS == OS_OS2)
      if (percent+1 >= endperc)
         ch = 'Û';
      else
         ch = '°';
    #else
      ch = '.';
    #endif
      if (multiperc)
      {
        equals = 0;
        for ( prob_i=0; prob_i<load_problem_count; prob_i++ )
        {
          if ( pbuf[prob_i] == (unsigned char)(percent) )
          {
            ch = (char)('a'+prob_i);
            if ( (++equals)>displevel )
              break;
          }
        }
      }
      *bufptr++ = ch;
    }
  }
  displevel++;
  if (displevel >= load_problem_count)
    displevel=0;
  lastperc = endperc;

  if (batonchar)
    *bufptr++ = batonchar;
  *bufptr = '\0';
  if ( (buffer[0]==0) || (buffer[0]=='\r' && buffer[1]==0) )
    ;
  else
  {
    int doit = 1;
    if ((logstatics.last_screen_flags & LOGTO_PERCENTBAR) != 0)
    {
      static char lastbuffer[sizeof(buffer)] = {0};
      if ((doit = strcmp( lastbuffer, buffer )) != 0)
        strcpy( lastbuffer, buffer );
    }
    if (doit)
    {
      LogWithPointer( LOGTO_SCREEN|LOGTO_RAWMODE|LOGTO_PERCENTBAR, buffer, NULL );
    }
  }
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
                              const char *userslogname, char *logname,
                              unsigned int maxlognamelen,
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

void InitializeLogging( int noscreen, int nopercent, const char *logfilename,
                        const char *logfiletype, const char *logfilelimit,
                        long mailmsglen, const char *smtpsrvr,
                        unsigned int smtpport, const char *smtpfrom,
                        const char *smtpdest, const char *id )
{
  DeinitializeLogging();
  logstatics.percprint = (nopercent == 0);
  logstatics.percbaton = (nopercent == 0 /*&& nobaton == 0 */);
  logstatics.spoolson = 1;

  if ( noscreen == 0 )
  {
    logstatics.loggingTo |= LOGTO_SCREEN;
    logstatics.stableflag = 0;   //assume next log screen needs a '\n' first
  }

  fixup_logfilevars( logfiletype, logfilelimit,
                     &logstatics.logfileType, &logstatics.logfileLimit,
                     logfilename, logstatics.logfile,
                     (sizeof(logstatics.logfile)-10),
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
