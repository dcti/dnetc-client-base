/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
 *
 * Please get in touch with me before implementing support for 
 * the extended file logging types (rotate/fifo/restart types).
 * ----------------------------------------------------------------------
*/
const char *logstuff_cpp(void) {
return "@(#)$Id: logstuff.cpp,v 1.34 1999/04/20 02:03:28 cyp Exp $"; }

#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "mail.h"      // MailMessage
#include "clitime.h"   // CliGetTimeString(NULL,1)
#include "pathwork.h"  // GetFullPathForFilename( x )
#include "problem.h"   // needed for logscreenpercent
#include "cmpidefs.h"  // strcmpi()
#include "console.h"   // for ConOut() and ConIsScreen()
#include "triggers.h"  // don't print percbar if pause/exit/restart triggered
#include "logstuff.h"  // keep the prototypes in sync

//-------------------------------------------------------------------------

#if 0 /* logstuff.h defines */

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

#endif /* logstuff.h defines */

#if (CLIENT_OS == OS_NETWARE || CLIENT_OS == OS_DOS || \
     CLIENT_OS == OS_OS2 || CLIENT_OS == OS_WIN16 || \
     CLIENT_OS == OS_WIN32 || CLIENT_OS == OS_WIN32S )
  #define ftruncate( fd, sz )  chsize( fd, sz )
#elif (CLIENT_OS == OS_VMS || CLIENT_OS == OS_RISCOS || \
     CLIENT_OS == OS_AMIGAOS || CLIENT_OS == OS_MACOS)
  #define ftruncate( fd, sz ) //nada, not supported
  #define FTRUNCATE_NOT_SUPPORTED
#endif  

#ifdef DONT_USE_PATHWORK
  #define GetFullPathForFilename( x ) ( x )
#endif  

#if ((!defined(MAX_LOGENTRY_LEN)) || (MAX_LOGENTRY_LEN < 1024))
  #ifdef MAX_LOGENTRY_LEN
  #undef MAX_LOGENTRY_LEN
  #endif
  #define MAX_LOGENTRY_LEN 1024
#endif   

// ========================================================================

static struct 
{
  int initlevel;
  char loggingTo;            // LOGTO_xxx bitfields 
  char spoolson;             // mail/file logging and time stamping is on/off.
  char percprint;            // percentprinting is enabled
  
  MailMessage *mailmessage;  //note: pointer, not class struct.
  char logfile[128+20];      //fname when LOGFILETYPE_RESTART or _FIFO
                             //lastused fname when LOGFILETYPE_ROTATE
  unsigned int logfilebaselen;//len of the log fname without ROTATE suffix
  int  logfileType;          //rotate, restart, fifo, none
  unsigned int logfileLimit; //days when rotating or kbyte when fifo/restart
  unsigned int logfilestarted; // 1 after the first logfile write

  char stdoutisatty;         //log screen can handle lines not ending in '\n'
  char stableflag;           //last log screen did end in '\n'
  char lastwasperc;          //last log screen was a percentbar
  
} logstatics = { 
  0,      // initlevel
  LOGTO_NONE,   // loggingTo
  0,      // spoolson
  0,      // percprint
  NULL,     // *mailmessage
  {0},      // logfile[]
  0,      // logfilebaselen
  LOGFILETYPE_NONE, // logfileType
  0,      // logfileLimit
  0,      // logfilestarted
  0,      // stdoutisatty
  0,      // stableflag
  0 };      // lastwasperc

// ========================================================================

static void InternalLogScreen( const char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  if ((logstatics.loggingTo & LOGTO_SCREEN) != 0)
  {
    if ( msglen && (msgbuffer[msglen-1] == '\n' || ConIsScreen() ) )
    {
      if (strlen( msgbuffer ) == msglen) //we don't do binary data
        ConOut( msgbuffer );             //which shouldn't happen anyway.
    }
    else
      ConOut( "" ); //flush.
  }
  return;
}

// ------------------------------------------------------------------------

//this can ONLY be called from LogWithPointer. msgbuffer is recycled!
static void InternalLogFile( char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  int logfileType = logstatics.logfileType;
  unsigned int logfileLimit = logstatics.logfileLimit;
  FILE *logstream = NULL;
  
  if ( logstatics.logfile[0] == 0 )
    return;
  if ( !msglen || msgbuffer[msglen-1] != '\n') 
    return;
  if ( logfileType == LOGFILETYPE_NONE || logstatics.spoolson==0 ||
       (logstatics.loggingTo & LOGTO_FILE) == 0)
    return;
    
  #ifdef FTRUNCATE_NOT_SUPPORTED
  if ( ( logfileType & LOGFILETYPE_FIFO ) != 0 ) 
    logfileType = LOGFILETYPE_NOLIMIT;
  #endif    

  if ( logfileType == LOGFILETYPE_NOLIMIT )
  {
    logstream = fopen( GetFullPathForFilename( logstatics.logfile ), "a" );
    if (logstream)
    {
      fwrite( msgbuffer, sizeof( char ), msglen, logstream );
      fclose( logstream );
    }
    logstatics.logfilestarted = 1;
  }
  else if ( logfileType == LOGFILETYPE_RESTART ) 
  {
    long filelen = (long)(-1);
    //if ( logfileLimit < 100 )
    //  logfileLimit = 100;
    logstream = fopen( GetFullPathForFilename( logstatics.logfile ), "a" );
    if ( logstream )
    {
      fwrite( msgbuffer, sizeof( char ), msglen, logstream );
      filelen = (long)ftell( logstream );
      fclose( logstream );
    }
    if ( filelen != (long)(-1) && ((unsigned int)(filelen >> 10)) > logfileLimit )
    {
      logstream= fopen( GetFullPathForFilename( logstatics.logfile ), "w" ); 
      if ( logstream )
      {
        fprintf( logstream, "[%s] Log file exceeded %uKbyte limit. "
           "Restarted...\n\n", CliGetTimeString( NULL, 1 ), 
           (unsigned int)( logstatics.logfileLimit ));
        fwrite( msgbuffer, sizeof( char ), msglen, logstream );
        fclose( logstream );
      }
    }
    logstatics.logfilestarted = 1;
  }
  else if ( logfileType == LOGFILETYPE_ROTATE )
  {
    static logfilebaselen = 0;
    static unsigned int last_year = 0, last_mon = 0, last_day = 0;
    static unsigned long last_jdn = 0;
    unsigned int curr_year, curr_mon, curr_day;
    unsigned long curr_jdn;
    struct tm *currtmP;
    time_t ttime;

    if (!logstatics.logfilestarted)
      logfilebaselen = strlen( logstatics.logfile );

    ttime = time(NULL);
    currtmP = localtime( &ttime );
    if (currtmP != NULL ) 
    {
      curr_year = currtmP->tm_year+ 1900;
      curr_mon  = currtmP->tm_mon + 1;
      curr_day  = currtmP->tm_mday;
      #define _jdn( y, m, d ) ( (long)((d) - 32076) + 1461L * \
        ((y) + 4800L + ((m) - 14) / 12) / 4 + 367 * \
        ((m) - 2 - ((m) - 14) / 12 * 12) / 12 - 3 * \
        (((y) + 4900L + ((m) - 14) / 12) / 100) / 4 + 1 )
      curr_jdn = _jdn( curr_year, curr_mon, curr_day );
      #undef _jdn
      if (( curr_jdn - last_jdn ) > logfileLimit )
      {
        last_jdn  = curr_jdn;
        last_year = curr_year;
        last_mon  = curr_mon;
        last_day  = curr_day;
      }
    }
    sprintf( logstatics.logfile+logfilebaselen, 
             "%02d%02d%02d.log", (int)(last_year%100), 
             (int)(last_mon), (int)(last_day) );
    logstream = fopen( GetFullPathForFilename( logstatics.logfile ), "a" );
    if ( logstream )
    {
      fwrite( msgbuffer, sizeof( char ), msglen, logstream );
      fclose( logstream );
    }
    logstatics.logfilestarted = 1;
  }
  else if ( logfileType == LOGFILETYPE_FIFO ) 
  {
    unsigned long filelen = 0;
    if ( logfileLimit < 100 )
      logfileLimit = 100;
    logstream = fopen( GetFullPathForFilename( logstatics.logfile ), "a" );
    if ( logstream )
    {
      fwrite( msgbuffer, sizeof( char ), msglen, logstream );
      if (((long)(filelen = ftell( logstream ))) == ((long)(-1)))
        filelen = 0;
      fclose( logstream );
    }
    if ( filelen > (((unsigned long)(logfileLimit))<<10) )
    {    /* careful: file must be read/written without translation - cyp */
      logstream = fopen( GetFullPathForFilename( logstatics.logfile ), "r+b" );
      if ( logstream )
      {
        unsigned long next_top = filelen - /* keep last 90% */
                               ((((unsigned long)(logfileLimit))<<10)*9)/10;
        if ( fseek( logstream, next_top, SEEK_SET ) == 0 &&
          ( msglen = fread( msgbuffer, sizeof( char ), MAX_LOGENTRY_LEN-1, 
                    logstream ) ) != 0 )
        {
          msgbuffer[msglen]=0;
          char *p = strchr( msgbuffer, '\n' );  //translate manually
          char *q = strchr( msgbuffer, '\r' );  //to find next line start
          if ( q != NULL && q > p ) 
            p = q;
          if ( p != NULL )
          {
            while (*p=='\r' || *p=='\n') 
              p++;
            next_top += ( p - (&msgbuffer[0]) );
          }
          filelen = 0;
          
          while ( fseek( logstream, next_top, SEEK_SET ) == 0 &&
             ( msglen = fread( msgbuffer, sizeof( char ), MAX_LOGENTRY_LEN, 
                    logstream ) ) != 0 &&
                fseek( logstream, filelen, SEEK_SET ) == 0 &&
             ( msglen == fwrite( msgbuffer, sizeof( char ), msglen, 
                     logstream ) ) )
          {
            next_top += msglen;
            filelen += msglen;
          }
          ftruncate( fileno( logstream ), filelen );
        }
        fclose( logstream );
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
    logstatics.mailmessage->append( msgbuffer ); 
  return;
}

// ------------------------------------------------------------------------

// On NT/Alpha (and maybe some other platforms) the va_list type is a struct,
// not an int or a pointer-type.  Hence, NULL is not a valid va_list.  Pass
// a (va_list *) instead to avoid this problem
void LogWithPointer( int loggingTo, const char *format, va_list *arglist ) 
{
  static int recursion_check = 0;
  if ((++recursion_check) > 2) /* log->mail->network */
    loggingTo &= ~LOGTO_MAIL;

  char msgbuffer[MAX_LOGENTRY_LEN];
  unsigned int msglen = 0;
  char *buffptr, *obuffptr;
  const char *timestamp;
  int sel, old_loggingTo = loggingTo;
  
  msgbuffer[0]=0;
  loggingTo &= (logstatics.loggingTo|LOGTO_RAWMODE);


  if ( !format || !*format )
    loggingTo = LOGTO_NONE;
  
  if ( loggingTo != LOGTO_NONE && *format == '\r' )   //can only be screen
  {                                                 //(or nothing)
    if (( loggingTo & LOGTO_SCREEN ) != 0 )
      loggingTo = (LOGTO_SCREEN|LOGTO_RAWMODE);  //force into raw mode
    else
      loggingTo = LOGTO_NONE;
  }

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
    do{
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

  #ifdef ASSERT_WIDTH_80  //"show" where badly formatted lines are cropping up
  if (loggingTo != LOGTO_NONE)
  {
    buffptr = &msgbuffer[0];
    do{
      while (*buffptr == '\r' || *buffptr == '\n' )
         buffptr++;
      obuffptr = buffptr;
      while (*buffptr && *buffptr != '\r' && *buffptr != '\n' )
        buffptr++;
      if ((buffptr-obuffptr) > 79)
      {
        obuffptr[75] = ' '; obuffptr[76] = obuffptr[77] = obuffptr[78] = '.';
        memmove( obuffptr+79, buffptr, strlen(buffptr)+1 );
        buffptr = obuffptr+79;
      }    
    } while (*buffptr);
    msglen = strlen( msgbuffer );
  }      
  #endif

  if (( loggingTo & LOGTO_SCREEN ) != 0)
  {
    buffptr = &msgbuffer[0];
    if ((loggingTo & LOGTO_RAWMODE)==0)
    {
      if (*buffptr=='\n' && logstatics.stableflag) 
      {
        buffptr++;
        msglen--;
      }
      else if (*buffptr!='\r' && *buffptr!='\n' && !logstatics.stableflag) 
      {
        msglen++;
        memmove( msgbuffer+1, msgbuffer, msglen );
        msgbuffer[0] = '\n';
  // CRAMER - should this even be here? (I think not)
        logstatics.stableflag = 1;
      }
    }  
    if (msglen)
    {
      logstatics.lastwasperc = 0; //perc bar looks for this
      logstatics.stableflag = ( buffptr[(msglen-1)] == '\n' );
      InternalLogScreen( buffptr, msglen, 0 );
    }
  }
  
  if (logstatics.spoolson && ( loggingTo & LOGTO_FILE ) != 0)
    InternalLogFile( msgbuffer, msglen, 0 );

  if (logstatics.spoolson && ( loggingTo & LOGTO_MAIL ) != 0)
    InternalLogMail( msgbuffer, msglen, 0 );
      
  --recursion_check;
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
      logstatics.mailmessage->checktosend(forceflush);
  }
  return;
}

// ------------------------------------------------------------------------

void LogScreen( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN, format, &argptr );
  va_end(argptr);
  return;
}    

void LogScreenRaw( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_RAWMODE|LOGTO_SCREEN, format, &argptr );
  va_end(argptr);
  return;
}  

void Log( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN|LOGTO_FILE|LOGTO_MAIL, format, &argptr );
  va_end(argptr);
  return;
}  

void LogRaw( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_RAWMODE|LOGTO_SCREEN|LOGTO_FILE|LOGTO_MAIL, format, &argptr );
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

#include "probman.h"

void LogScreenPercent( unsigned int load_problem_count )
{
  static unsigned int lastperc = 0, displevel = 0;
  unsigned int percent, restartperc, endperc, equals, prob_i;
  int isatty, multiperc;
  char ch; char buffer[88];
  char *bufptr = &buffer[0];
  unsigned char pbuf[30]; /* 'a'-'z' */

  if (CheckExitRequestTrigger() || CheckPauseRequestTrigger() || 
    !logstatics.percprint || ( logstatics.loggingTo & LOGTO_SCREEN ) == 0 )
    return;

  isatty  = ConIsScreen();
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
    
  if (!logstatics.lastwasperc || isatty)
    lastperc = 0;
  multiperc = (load_problem_count > 1 && load_problem_count <= 26 /*a-z*/ 
                 && lastperc == 0 && isatty && endperc < 100);
  if (lastperc == 0 && endperc > 0 && isatty )
    *bufptr++ = '\r';

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
      ch = '.';
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

  if ( (buffer[0]==0) || (buffer[0]=='\r' && buffer[1]==0) )
    ;
  else
  {
    *bufptr = 0;
    LogWithPointer( LOGTO_SCREEN|LOGTO_RAWMODE, buffer, NULL );
    logstatics.stableflag = 0; //(endperc == 0);  //cursor is not at column 0 
    logstatics.lastwasperc = 1; //(endperc != 0); //percbar requires reset
  }
  return;
}

// ------------------------------------------------------------------------

void DeinitializeLogging(void)
{
  if (logstatics.mailmessage) 
  {
    logstatics.mailmessage->Deinitialize(); //forces a send
    delete logstatics.mailmessage;
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
                              unsigned int maxlognamelen )
{
  unsigned int len;
  int climit = 0;
  char scratch[20];
  long l = 0;

  *type = LOGFILETYPE_NONE;
  *limit = 0;
  *logname = 0;
  
  if (!userslogname)
    return 0;
  while (*userslogname && isspace(*userslogname))
    userslogname++;
  strncpy( logname, userslogname, maxlognamelen );
  logname[maxlognamelen-1]='\0';
  len = strlen( logname );
  while (len > 1 && isspace(logname[len-1]))
    logname[--len]='\0';
  if (!*logname || strcmp( logname, "none" )==0)
  {
    *logname='\0';
    return 0;
  }

  *type = LOGFILETYPE_NOLIMIT;

  if (slimit)
  {
    while (*slimit && isspace(*slimit))
      slimit++;
    while (isdigit(*slimit))
      l=((l*10)+(*slimit++)-'0');
    while (*slimit && isspace(*slimit))
      slimit++;
    climit = tolower(*slimit);
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

  if (*type == LOGFILETYPE_NONE)
  {
    *limit = 0; /* limit is ignored */
  }
  else if (*type == LOGFILETYPE_NOLIMIT)
  {
    *limit = 0; /* limit is ignored */
  }
  else if (*type == LOGFILETYPE_ROTATE)
  {
    /* convert to days */
    if (climit == 'm')
      l *= 30;
    else if (climit == 'w')
      l *= 7;
    else if (climit == 'y')
      l *= 365;
    if (l > INT_MAX)
      l = INT_MAX;
    *limit = (unsigned int)l;
  }
  else //(*type == LOGFILETYPE_RESTART || *type == LOGFILETYPE_FIFO)
  {
    /* convert to Kb */
    if (climit == 'g') /* dickheads! */
    {
      *type = LOGFILETYPE_NOLIMIT;
      *limit = 0;
    }
    else 
    {
      if (climit == 'm')
        l *= 1024;
      if (l > INT_MAX)
        l = INT_MAX;
      *limit = (unsigned int)l;
    }
  }  
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
  logstatics.spoolson = 1;

  if ( noscreen == 0 )
  {
    logstatics.loggingTo |= LOGTO_SCREEN;
    logstatics.stableflag = 0;   //assume next log screen needs a '\n' first
  }

  fixup_logfilevars( logfiletype, logfilelimit,
                     &logstatics.logfileType, &logstatics.logfileLimit,
                     logfilename, logstatics.logfile, 
                     (sizeof(logstatics.logfile)-10));
  if (logstatics.logfileType != LOGFILETYPE_NONE)
    logstatics.loggingTo |= LOGTO_FILE;

  if (mailmsglen > 0)
    logstatics.mailmessage = new MailMessage();
  if (logstatics.mailmessage)
  {
    logstatics.loggingTo |= LOGTO_MAIL;
    logstatics.mailmessage->Initialize( mailmsglen, smtpsrvr, smtpport,
                                        smtpfrom, smtpdest, id );
  }
  return;
}  

// ---------------------------------------------------------------------------

