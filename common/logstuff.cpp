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
return "@(#)$Id: logstuff.cpp,v 1.37.2.35 2000/11/02 22:31:08 oliver Exp $"; }

#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "client.h"    // CONTEST_COUNT
#include "mail.h"      // MailMessage
#include "clitime.h"   // CliGetTimeString(NULL,1)
#include "pathwork.h"  // GetFullPathForFilename(), GetWorkingDirectory()
#include "problem.h"   // Problem object for logscreenpercent
#include "probman.h"   // GetProblemPointerFromIndex() for LogScreenPercent
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
  char logfile[128+20];      //fname when LOGFILETYPE_RESTART or _FIFO
                             //lastused fname when LOGFILETYPE_ROTATE
  FILE *logstream;           //open logfile

  unsigned int logfilebaselen;//len of the log fname without ROTATE suffix
  int  logfileType;          //rotate, restart, fifo, none
  unsigned int logfileLimit; //days when rotating or kbyte when fifo/restart
  unsigned int logfilestarted; // 1 after the first logfile write

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
  0,      // logfilebaselen
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
      if (logstatics.stableflag) /* previous print ended with '\n' */
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
      logstatics.stableflag = ( buffptr[(msglen-1)] == '\n' );
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
  unsigned int percent, restartperc, endperc, prob_i, cont_i;
  unsigned int selprob_i = logstatics.perc_callcount % load_problem_count;
  char buffer[128]; unsigned char pbuf[52]; /* 'a'-'z','A'-'Z' */
  int use_alt_fmt; unsigned int prob_count[CONTEST_COUNT];

  if (!logstatics.crunchmeter || ( logstatics.loggingTo & LOGTO_SCREEN ) == 0 )
    return;

  for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++)
    prob_count[cont_i] = ProblemCountLoaded(cont_i); /* -1=all contests */

  use_alt_fmt = 0;
  if (logstatics.crunchmeter == 1)
    use_alt_fmt = +1; 
  else if (logstatics.crunchmeter < 0 && logstatics.stdoutisatty &&
     (prob_count[OGR] > 0 || load_problem_count >= sizeof(pbuf)))
    use_alt_fmt = -1;      

  buffer[0] = '\0';
  endperc = restartperc = 0;
  for (prob_i = 0; prob_i < load_problem_count; prob_i++)
  {
    Problem *selprob = GetProblemPointerFromIndex(prob_i);
    pbuf[prob_i] = 0;
    if (selprob)
    {
      unsigned int permille = 0, startpermille = 0;  int girc = -1;
      cont_i = 0;

      if (use_alt_fmt && (!buffer[0] || prob_i == selprob_i))
      {
        char blkdone[32], blksig[32]; const char *contname;
        girc = selprob->GetProblemInfo(&cont_i, &contname, 0, 0, 0, 0, 0, 
                         &permille, &startpermille, 0,
                         blksig, sizeof(blksig), 0, 0, 0, 0, 0,0,0, 0,0,0,
                         0, 0, 0, 0, blkdone, sizeof(blkdone) );
        if (permille == 1000 &&  use_alt_fmt < 0) /* auto */
          use_alt_fmt = 0;
        else if (girc != -1)
        {
          sprintf(buffer, "#%u: %s:%s [%s]", 
                  prob_i+1, contname, blksig, blkdone );
          //there isn't enough space for percent so don't even think about it
        }
      }
      else
      {
        girc = selprob->GetProblemInfo(&cont_i, 0, 0, 0, 0, 0, 0,
                         &permille, &startpermille, 0,
                         0, 0, 0, 0, 0, 0, 0,0,0, 0,0,0, 0, 0, 0, 0, 0, 0 );
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
          if (endperc == 100 && use_alt_fmt < 0) /* auto */
            use_alt_fmt = 0;
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

  if (buffer[0] && use_alt_fmt)
  {
    #if (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_POWERPC)
    // temporary fix - updating the progress every 5 seconds is not a good idea
    // since it causes a number of context-switches to the 68K, which in turn
    // slows the 68K client down quite dramatically if both clients are running
    // in parallel.  So, we only do it once per minute, minus the time display.
    static int cnt = 11;
    if (cnt++ > 10)
    {
      LogScreenRaw( "\r%s", buffer, NULL );
      cnt = 0;
      logstatics.stableflag = 0; //(endperc == 0);  //cursor is not at column 0
      logstatics.lastwasperc = 1; //(endperc != 0); //percbar requires reset
    }
    #else
    LogScreen( "\r%s", buffer, NULL );
    logstatics.stableflag = 0; //(endperc == 0);  //cursor is not at column 0
    logstatics.lastwasperc = 1; //(endperc != 0); //percbar requires reset
    /* simple, eh? :) */
    #endif
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

