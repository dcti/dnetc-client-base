// Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: logstuff.cpp,v $
// Revision 1.2  1998/08/03 21:13:38  cyruspatel
// Fixed many bugs (especially percentbar related ones). New log file
// types work correctly now. Added some functionality, eg a function for
// obtaining the name of the last used log file (used for win32 gui graphing).
//
// Revision 1.1  1998/08/02 16:00:42  cyruspatel
// Created. Check the FIXMEs! Please get in touch with me before implementing
// support for the extended file logging types (rotate/fifo/restart types).
//
//

//-------------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *logstuff_cpp(void) {
return "@(#)$Id: logstuff.cpp,v 1.2 1998/08/03 21:13:38 cyruspatel Exp $"; }
#endif

//-------------------------------------------------------------------------

#include "cputypes.h"
#include "client.h"    // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "mail.h"      // MailMessage
#include "clitime.h"   // CliGetTimeString(NULL,1)
#include "pathwork.h"  // GetFullPathForFilename( x )
#include "problem.h"   // needed for logscreenpercent
#include "cmpidefs.h"  // strcmpi()
#include "logstuff.h"  // keep the prototypes in sync

//-------------------------------------------------------------------------

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

//#define ASSERT_DATE_STAMP   //make any non-stamped message a stamped one
//#define ASSERT_WIDTH_80     //show where badly formatted lines are cropping up

#if (CLIENT_OS == OS_NETWARE || CLIENT_OS == OS_DOS || \
     CLIENT_OS == OS_OS2 || CLIENT_OS == OS_WIN16 || \
     CLIENT_OS == OS_WIN32 || CLIENT_OS == OS_WIN32S )
  #define ftruncate( fd, sz )  chsize( fd, sz )
#elif (CLIENT_OS == OS_VMS || CLIENT_OS == OS_RISCOS || CLIENT_OS == OS_RISCOS)
  #define ftruncate( fd, sz ) //nada, not supported
  #define FTRUNCATE_NOT_SUPPORTED
#endif  

#ifdef DONT_USE_PATHWORK
  #define GetFullPathForFilename( x ) ( x )
#endif  

// ========================================================================

static struct 
{
  char loggingTo;            // LOGTO_xxx bitfields 
  
  MailMessage *mailmessage;  //note: pointer, not class struct.
  char logfile[128+20];      //fname when LOGFILETYPE_RESTART or _FIFO
                             //lastused fname when LOGFILETYPE_ROTATE
  unsigned int logfilebaselen;//len of the log fname without ROTATE suffix
  int  logfileType;          //rotate, restart, fifo, none
  unsigned int logfileLimit; //days when rotating or kbyte when fifo/restart
  unsigned int logfilestarted; // 1 after the first logfile write
  
  char stdoutisatty;         //log screen can handle lines not ending in '\n'
  char stableflag;           //last log screen didn't end in '\n'
  char lastwasperc;          //last log screen was a percentbar
  char mailpending;          //a best guess only
  
} logstatics = { LOGTO_NONE, NULL, {0}, 0, LOGFILETYPE_NONE, 0,0,0,0,0,0 };

// ========================================================================


#if (!defined(NEEDVIRTUALMETHODS))  // gui clients will override this function
void InternalLogScreen( const char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  #if (CLIENT_OS == OS_OS2)
  //DosSetPriority(PRTYS_THREAD, PRTYC_REGULAR, 0, 0);  //FIXME
  // Give prioirty boost so that text appears faster
  #endif

  if (msglen && (msgbuffer[msglen-1] == '\n' || logstatics.stdoutisatty ))
    fwrite( msgbuffer, sizeof(char), msglen, stdout);
  fflush(stdout);

  #if (CLIENT_OS == OS_OS2)
  //SetNiceness();              //FIXME
  #endif
  return;
}
#endif

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
  if ( logfileType == LOGFILETYPE_NONE )
    return;
    
  if ( logfileLimit == (unsigned int)(-1) ) 
    logfileType = LOGFILETYPE_NOLIMIT;
  #ifdef FTRUNCATE_NOT_SUPPORTED
  if ( ( logfileType & LOGFILETYPE_FIFO ) != 0 ) 
    logfileType = LOGFILETYPE_NOLIMIT;
  #endif    

  if (( logfileType & LOGFILETYPE_NOLIMIT ) != 0 ) 
    {
    logstream = fopen( GetFullPathForFilename( logstatics.logfile ), "a" );
    if (logstream)
      {
      fwrite( msgbuffer, sizeof( char ), msglen, logstream );
      fclose( logstream );
      }
    logstatics.logfilestarted = 1;
    }
  else if (( logfileType & LOGFILETYPE_RESTART ) != 0 ) 
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
  else if (( logfileType & LOGFILETYPE_ROTATE ) != 0)
    {
    static unsigned int last_year = 0, last_mon = 0, last_day = 0;
    static unsigned long last_jdn = 0;
    unsigned int curr_year, curr_mon, curr_day;
    unsigned long curr_jdn;
    struct tm *currtmP;
    time_t ttime;

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
      if (( curr_jdn - last_jdn ) > logfileLimit );
        {
        last_jdn  = curr_jdn;
        last_year = curr_year;
        last_mon  = curr_mon;
        last_day  = curr_day;
        }
      }
    sprintf( logstatics.logfile+logstatics.logfilebaselen, 
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
  else if ( ( logfileType & LOGFILETYPE_FIFO ) != 0 ) 
    {
    unsigned long filelen = 0;
    //if ( logfileLimit < 100 )
    //  logfileLimit = 100;
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

static void InternalLogMail( char *msgbuffer, unsigned int msglen, int /*flags*/ )
{
  if ( msglen && logstatics.mailmessage && logstatics.mailmessage->messagelen )
    {
    logstatics.mailmessage->addtomessage( msgbuffer ); //takes a char *
    logstatics.mailpending = 1;
    }
  return;
}

// ------------------------------------------------------------------------

void LogWithPointer( int loggingTo, const char *format, va_list argptr ) 
{
  char msgbuffer[MAX_LOGENTRY_LEN];
  unsigned int head, msglen = 0;
  
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

  if ( loggingTo != LOGTO_NONE && (loggingTo & LOGTO_RAWMODE) == 0) 
    {
    head = 0;
    msgbuffer[0]=0;
    while (format[head]=='\r' || format[head]=='\n')
      { msgbuffer[head] = format[head]; msgbuffer[++head]=0; }
    if ( format[head]!=0 && format[head]!='[' && format[head]!=' ')  
      {
      format+=head;
      sprintf( (msgbuffer+strlen(msgbuffer)), "[%s] ", "Jan 01 00:00:00 GMT" ); //CliGetTimeString(NULL,1) );
      }
    }

  if ( loggingTo != LOGTO_NONE )
    {
    if ( argptr == NULL )
      strcat( msgbuffer, format );
    else 
      vsprintf( msgbuffer, format, argptr);
    msglen = strlen( msgbuffer );
    if ( msglen == 0 )
      loggingTo = LOGTO_NONE;
    else if (msgbuffer[msglen-1] != '\n')
      loggingTo &= LOGTO_SCREEN|LOGTO_RAWMODE;  //screen only obviously
    }

  if (( loggingTo & LOGTO_SCREEN ) != 0)
    {
    head = 0;
    if ((loggingTo & LOGTO_RAWMODE)==0)
      {
      if (msgbuffer[0]=='\n' && logstatics.stableflag) 
        {
        head++;
        msglen--;
        }
      else if (msgbuffer[0]!='\n' && !logstatics.stableflag) 
        {
        InternalLogScreen( "\n", 1, 0 );
        logstatics.stableflag = 1;
        }
      }  
    if (msglen)
      {
      logstatics.lastwasperc = 0; //perc bar looks for this
      logstatics.stableflag = ( msgbuffer[head+(msglen-1)] == '\n' );
      InternalLogScreen( msgbuffer+head, msglen, 0 );
      }
    }
  
  if (( loggingTo & LOGTO_FILE ) != 0)
    InternalLogFile( msgbuffer, msglen, 0 );

  if (( loggingTo & LOGTO_MAIL ) != 0)
    InternalLogMail( msgbuffer, msglen, 0 );
      
  return;
}

// ------------------------------------------------------------------------

void LogFlush( int forceflush )
{
  if (( logstatics.loggingTo & LOGTO_SCREEN ) != 0)
    {
    LogWithPointer( LOGTO_SCREEN, "\n", NULL );  //LF if needed then fflush()
    }
  if (( logstatics.loggingTo & LOGTO_MAIL ) != 0)
    {
    if ( logstatics.mailpending && logstatics.mailmessage && logstatics.mailmessage->messagelen )
      {
      logstatics.mailpending = 0;
      //int quietly = logstatics.mailmessage->quietmode;
      //logstatics.mailmessage->quietmode = (( loggingTo & LOGTO_SCREEN ) != 0);
      logstatics.mailmessage->checktosend(forceflush);
      //logstatics.mailmessage->quietmode = quietly;
      }
    }
  return;
}

// ------------------------------------------------------------------------

void LogScreen( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN, format, argptr );
  va_end(argptr);
  return;
}    

#if 0
void LogScreenf( const char *format, ... )//Legacy function. Same as LogScreen
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN, format, argptr );
  va_end(argptr);
  return;
}  
#endif

void LogScreenRaw( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_RAWMODE|LOGTO_SCREEN, format, argptr );
  va_end(argptr);
  return;
}  

void Log( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_SCREEN|LOGTO_FILE|LOGTO_MAIL, format, argptr );
  va_end(argptr);
  return;
}  

void LogRaw( const char *format, ... )
{
  va_list argptr;
  va_start(argptr, format);
  LogWithPointer( LOGTO_RAWMODE|LOGTO_SCREEN|LOGTO_FILE|LOGTO_MAIL, format, argptr );
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

extern Problem problem[2*MAXCPUS];

static void GetPercDataForThread( unsigned int selthread, unsigned int numthreads, 
   unsigned int *percent, unsigned int *lastperc, unsigned int *restartperc )
{
  unsigned int percent_i, lastperc_i, startperc_i;

  if ( !((problem[selthread]).started) &&
    ( (problem[selthread+numthreads]).started ) ) 
    selthread+=numthreads;
  startperc_i = (problem[selthread]).startpercent / 1000;
  lastperc_i  = (problem[selthread]).percent;
  percent_i   = 0;

  if ( (problem[selthread]).finished )
    percent_i = 100;
  else if ( (problem[selthread]).started )
    percent_i = (problem[selthread]).CalcPercent();

  if ( percent_i == 0 )
    {
    percent_i = startperc_i;
    lastperc_i = 0;
    }
  (problem[selthread]).percent = percent_i;

  if (restartperc) *restartperc = ((lastperc_i == 0)?(startperc_i):(0));
  if (percent)   *percent   = percent_i;
  if (lastperc)  *lastperc  = lastperc_i;
  return;
}  

void LogScreenPercent( unsigned int load_problem_count )
{
  static unsigned int lastperc = 0, displevel = 0;
  unsigned int method, percent, restartperc, specperc, equals;
  unsigned int selthread, numthreads = (load_problem_count+1)>>1;
  char buffer[88];
  char *bufptr = &buffer[0];

  if (( logstatics.loggingTo & LOGTO_SCREEN ) != 0 )
    {
    method = 2; //LogScreenPercentSingle() type bar for multiple threads
    if ( load_problem_count <= 2 )
      method = 0;  //old LogScreenPercentSingle()
    #if (defined(PERCBAR_ON_ONE_LINE))
    else if (logstatics.stdoutisatty && (numthreads*sizeof("A:00% "))<79 ) 
      method = 3; //"\rA:10% B:20% C:30% D:40% E:50% ... M:99%" on *one* line
    #elif (defined(FORCE_OLD_LOGSCREENPERCENTMULTI))
    else
      method = 1; //old LogScreenPercentMulti()
    #endif
    
    if (!logstatics.lastwasperc)
      {
      lastperc = 0; 
      for ( selthread = 0; selthread < load_problem_count; selthread++)
        problem[selthread].percent = 0;
      }

    if ( method == 1 ) //old LogScreenPercentMulti()
      {
      for (selthread = 0; selthread < numthreads; selthread++)    
        {
        GetPercDataForThread( selthread, numthreads, &percent, &lastperc, &restartperc );
        if (restartperc) restartperc = (restartperc & 0xFD)|1;
        for ( ++lastperc; lastperc <= percent ; lastperc++ )
          {
          if ( lastperc == 100 )
            { sprintf( bufptr, "%c:100%% ", selthread+'A' ); 
              bufptr+=sizeof("A:100% "); break; }
          else if ( ( lastperc % 10 ) == 0 ) 
            { sprintf( bufptr, "%c:%02d%% ", selthread+'A', lastperc ); 
              bufptr+=sizeof("A:90% "); }
          else if ( ( lastperc & 1 ) == 0 ) 
            { *bufptr++ = ((lastperc==restartperc)?('R'):('.')); }
          }
        }
      }
    else if ( method == 2 || method == 0 ) //single bar for all threads
      {
      specperc=0;
      for (selthread = 0; selthread < numthreads; selthread++)    
        {
        GetPercDataForThread(selthread,numthreads, &percent, NULL, &restartperc);
        if (percent > specperc)
          specperc = percent;
        if (percent && ((percent>90)?((percent&1)!=0):((percent&1)==0)))
          percent--;  //make sure that it is visible
        if (numthreads > 1)
          problem[selthread].percent = percent; 
        }
      if (numthreads < 2)
        {
        restartperc = (!restartperc || percent == 100) ? 0 :
         ( restartperc - ((restartperc > 90) ? (restartperc & 1) : 
                      (1 - (restartperc & 1))) );
        }
      else
        {
        restartperc = 0; //we don't bother with the restart flag
        if ( logstatics.stdoutisatty )
          {
          *bufptr++ = '\r';
          lastperc = 0;  //always paint the whole bar
          }
        }
      percent = lastperc;
      lastperc = specperc;

      for ( ++percent; percent <= specperc; percent++ )
        {
        if ( percent >= 100 )
          { strcpy( bufptr, "100" ); bufptr+=3; break; }
        else if ( ( percent % 10 ) == 0 )
          { sprintf( bufptr, "%d%%", (int)(percent) ); bufptr+=3; }
        else if ( restartperc == percent) 
          { *bufptr++='R'; }
        else if (((percent&1)?(percent<90):(percent>90)))
          {
          equals = 0;
          if (numthreads > 1 && logstatics.stdoutisatty )
            {
            for (selthread=0; selthread<numthreads; selthread++ )
              {
              if ( (problem[selthread].percent) == percent )
                {
                *bufptr = (char)('a'+selthread);
                if ( (++equals)>displevel )
                  break;
                }
              }
            }
          if (!equals)
            *bufptr = '.'; 
          bufptr++;
          }
        }
      displevel++;
      if (displevel >= numthreads)
        displevel=0;
      }
    #if 0
    else if ( method == 0 ) //old LogScreenPercentSingle()
      {
      GetPercDataForThread( 0, 1, &percent, NULL, &restartperc);
      if ( percent > lastperc )
        {
        restartperc = (!restartperc || percent == 100) ? 0 :
         ( percent - ((percent > 90) ? (percent & 1) : (1 - (percent & 1))) );
        for ( specperc = lastperc+1; (specperc <= percent) ; specperc++ )
          {
          if ( specperc == 100 )
            { strcat( bufptr, "100" ); bufptr+=3; }
          else if ( specperc  == restartperc )
            { *bufptr++ = 'R'; }
          else if ( ( specperc  % 10 ) == 0 )
            { sprintf( bufptr, "%d%%", (int)(specperc) ); bufptr+=3; }
          else if ((specperc&1)?(specperc<90):(specperc>90))
            { *bufptr++='.'; }
          }
        lastperc = percent;
        }
      }
    #endif
    else if ( method == 3 ) //PERCBAR_ON_ONE_LINE. Requires stdoutisatty
      {
      *bufptr++='\r';
      #if (CLIENT_OS != OS_WIN32) //looks lousy with a proportional font
      sprintf( bufptr, " %s  ", CliGetTimeString(NULL,0) );
      if ((strlen(bufptr)+(sizeof("A:00% ")*numthreads)) < 79)
        bufptr += strlen(bufptr);
      #endif
      for ( selthread = 0; selthread < numthreads; selthread++)
        {
        GetPercDataForThread( selthread, numthreads, &percent, NULL, NULL );
        sprintf( bufptr, ((percent<100)?("%c:%02d%% "):("%c:%d ")),
                selthread+'A', percent );
        bufptr+=sizeof("A:00% ");
        }
      }

    if ( bufptr > (&buffer[0]))
      {
      *bufptr = 0;
      LogWithPointer( LOGTO_SCREEN|LOGTO_RAWMODE, buffer, NULL );
      logstatics.lastwasperc = 1; //reset to 1
      }
    }
  return;
}

// ------------------------------------------------------------------------

#if 0

void Client::LogScreenPercentSingle( u32, u32, bool )
{ LogScreenPercent( numcputemp<<1 ); } //should be load_problem_count;

//cpu is only so that the funct doesn't run for *every* Client::Run inner loop
void Client::LogScreenPercentMulti(u32 cpu, u32, u32, bool)
{ if ( cpu == 0 ) LogScreenPercentSingle( 0, 0, 0 ); }

#endif

// ------------------------------------------------------------------------

void Client::DeinitializeLogging(void)
{
  LogFlush( 1 );
  if (logstatics.mailmessage)
    {
    delete logstatics.mailmessage;
    logstatics.mailpending = 0;
    logstatics.mailmessage = NULL;
    logstatics.loggingTo &= ~LOGTO_MAIL;    
    }
  logstatics.logfileType = LOGFILETYPE_NONE;
  logstatics.loggingTo &= ~LOGTO_FILE;    
  return;
}

// ---------------------------------------------------------------------------

void Client::InitializeLogging(void)
{
  DeinitializeLogging();
  logstatics.loggingTo = LOGTO_NONE;
  logstatics.lastwasperc = 0;

  if ( !quietmode )
    {
    logstatics.loggingTo |= LOGTO_SCREEN;
    logstatics.stableflag = 0;   //assume next log screen needs a '\n' first
    logstatics.stdoutisatty = 1; 
    #if (!defined(NEEDVIRTUALMETHODS))
    if (!isatty(fileno(stdout)))
      logstatics.stdoutisatty = 0;
    #endif
    }

  unsigned int mailmsglen = messagelen;
  if ( ((int)(mailmsglen)) < 0) mailmsglen = 0;
  if ( mailmsglen > MAXMAILSIZE) mailmsglen = MAXMAILSIZE;

  if (!logstatics.mailmessage && mailmsglen && !offlinemode)
    logstatics.mailmessage = new MailMessage();
  if (logstatics.mailmessage)
    {
    logstatics.mailpending = 0;
    logstatics.loggingTo |= LOGTO_MAIL;
    strcpy(logstatics.mailmessage->destid,smtpdest);
    strcpy(logstatics.mailmessage->fromid,smtpfrom);
    strcpy(logstatics.mailmessage->smtp,smtpsrvr);
    strcpy(logstatics.mailmessage->rc5id,id);
    logstatics.mailmessage->port = (int)smtpport;
    if (logstatics.mailmessage->port < 0) logstatics.mailmessage->port=25;
    if (logstatics.mailmessage->port > 65535L) logstatics.mailmessage->port=25;
    logstatics.mailmessage->quietmode = ((logstatics.loggingTo & LOGTO_SCREEN )!=0);
    logstatics.mailmessage->messagelen = mailmsglen;
    }

  if ( logname[0] && strcmpi( logname, "none" )!= 0)
    {
    logstatics.logfileType = LOGFILETYPE_NONE;
    if ( strlen( logname ) >= (sizeof( logstatics.logfile )-1) )
      LogScreen( "Log filename is too long. Logging to file remains disabled.\n");
    else
      {
      strcpy( logstatics.logfile, logname );
      logstatics.logfilebaselen = strlen( logstatics.logfile );
      logstatics.logfilestarted = 0;

      logstatics.loggingTo |= LOGTO_FILE;
      logstatics.logfileType = LOGFILETYPE_NOLIMIT;
      logstatics.logfileLimit = 0; // unused if LOGFILETYPE_NOLIMIT;

      /* ****************************************************************
      FIXME: Please check with me before implementing support for the 
             new logfiletypes - cyp
      ******************************************************************* */
      }
    }
  return;
}

// ------------------------------------------------------------------------

void CliScreenClear( void )  // SLIGHTLY out of place.... :) - cyp
{
  if (logstatics.stdoutisatty)
    {
    #if (CLIENT_OS == OS_WIN32)
      HANDLE hStdout;
      CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
      DWORD nLength;
      COORD topleft = {0,0};
      DWORD temp;

      hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
      if (hStdout == INVALID_HANDLE_VALUE) return;
      if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo)) return;
      nLength = csbiInfo.dwSize.X * csbiInfo.dwSize.Y;
      SetConsoleCursorPosition(hStdout, topleft);
      FillConsoleOutputCharacter(hStdout, (TCHAR) ' ', nLength, topleft, &temp);
      FillConsoleOutputAttribute(hStdout, csbiInfo.wAttributes, nLength, topleft, &temp);
      SetConsoleCursorPosition(hStdout, topleft);
    #elif (CLIENT_OS == OS_OS2)
      BYTE space[] = " ";
      VioScrollUp(0, 0, -1, -1, -1, space, 0);
      VioSetCurPos(0, 0, 0);      // move cursor to upper left
    #elif (CLIENT_OS == OS_DOS)
      dosCliClearScreen(); //in platform/dos/clidos.cpp
    #elif (CLIENT_OS == OS_NETWARE)
      clrscr();
    #elif (CLIENT_OS == OS_RISCOS)
      riscos_clear_screen();
    #else
      printf("\x1B" "[2J" "\x1B" "[H" "\r       \r" );
      //ANSI cls  '\r space \r' is in case ansi is not supported
    #endif
    }
  return;
}
