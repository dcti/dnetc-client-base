// Created by Tim Charron (tcharron@interlog.com) 97.9.17
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: mail.cpp,v $
// Revision 1.25  1998/08/24 23:50:03  cyp
// added mailmessage.clear() so logstuff can clear the spool if necessary.
//
// Revision 1.24  1998/08/21 00:05:47  cyruspatel
// Added a sendpendingflag so that smtp_deinitialize() (or the MailMessage
// destructor) will attempt a send() before the spool is destroyed/cleared.
//
// Revision 1.23  1998/08/20 19:25:01  cyruspatel
// Reverted to spooling by static buffer until infinite autobuffer 
// growth can be restricted.
//
// Revision 1.22  1998/08/15 21:32:11  jlawson
// updated mail to use autobuffer
//
// Revision 1.21  1998/08/15 18:09:45  cyruspatel
// (a) completely restructured to remove the logical limit on the size of a
// message. (b) mem bleeds cauterized - net object is now destroyed from only
// one place. (c) completed fifo handling of spool buffer. (d) mail is now
// also discarded on smtp error. (d) error messages are now in english :) -
// they hint at possible hotspots. (f) many sanity checks added, eg address
// handling is now RFC822 aware. (g) cleaned up a bit.
//
// Revision 1.20  1998/08/10 20:29:39  cyruspatel
// Call to gethostname() is now a call to Network::GetHostName(). Updated
// send routine to reflect new NetworkInitialize()/NetworkDeinitialize()
// requirements. Removed all references to NO!NETWORK.
//
// Revision 1.19  1998/08/02 16:18:06  cyruspatel
// Completed support for logging.
//
// Revision 1.18  1998/08/02 03:16:54  silby
// Log,LogScreen, and LogScreenf are in logging.cpp, and are global functions 
// Lurk handling has been added into the Lurk class, which resides in lurk.
//
// Revision 1.17  1998/07/26 12:46:07  cyruspatel
// Network constructor extended to take 'autofindkeyserver' as an argument.
//
// Revision 1.16  1998/07/13 23:54:23  cyruspatel
// Cleaned up NO!NETWORK handling.
//
// Revision 1.15  1998/07/13 03:30:05  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// "declaration/type or an expression" ambiguities. 
//
// Revision 1.14  1998/07/08 05:19:32  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.13  1998/07/07 21:55:43  cyruspatel
// client.h has been split into client.h and baseincs.h
//
// Revision 1.12  1998/07/06 09:21:24  jlawson
// added lint tags around cvs id's to suppress unused variable warnings.
//
// Revision 1.11  1998/06/15 12:04:01  kbracey
// Lots of consts.
//
// Revision 1.10  1998/06/14 08:26:51  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.9  1998/06/14 08:12:56  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.8  1998/06/11 09:43:27  jlawson
// mail, network, and problem will no longer print any error messages if
// the NEEDVIRTUALMETHODS item is defined.  Printing messages to stdout in
// the Win32 GUI compile was causing the i/o queus to become backed up.
//
// Revision 1.7  1998/06/08 15:47:09  kbracey
// Added lots of "const"s and "static"s to reduce compiler warnings, and
// hopefully improve output code, too.
//
// Revision 1.6  1998/06/04 00:19:15  timc
// Changed subject from RC5-64 to RC5DES
//
// Revision 1.5  1998/06/03 08:16:54  bovine
// preliminary changes for win32s windows 3.1 client
//
// Revision 1.4  1998/05/29 08:01:16  bovine
// copyright update, indents
//
// Revision 1.3  1998/05/25 07:16:47  bovine
// fixed warnings on g++/solaris
//
// Revision 1.2  1998/05/25 05:58:34  bovine
// fixed warnings for Borland C++
//
// Revision 1.1  1998/05/24 14:25:53  daa
// Import 5/23/98 client tree
//
// Revision 1.0  1997/09/17 09:17:07  timc
// Created
//-------------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *mail_cpp(void) {
return "@(#)$Id: mail.cpp,v 1.25 1998/08/24 23:50:03 cyp Exp $"; }
#endif

#include "network.h"
#include "version.h"
#include "baseincs.h"
#include "mail.h"
#include "logstuff.h"
#include "clitime.h"
#define Time() (CliGetTimeString(NULL,1))

//-------------------------------------------------------------------------

//#define SHOWMAIL    // define showmail to see mail transcript on stdout

//-------------------------------------------------------------------------

static int get_smtp_result( Network * net )
{
  char in_data[10];
  int index = 0;
  in_data[3] = 0;

  while (index>=0) //ie, while(1)
    {
    if ( net->Get( 1, &in_data[index], 2*NETTIMEOUT ) != 1)
      return(-1);
    #ifdef SHOWMAIL
    printf( "%c", in_data[index] );
    #endif
    if (in_data[index] == '\n')
      {
      if ( in_data[3] != '-') //if not an ESMPT multi-line reply
        break;
      index = 0;
      }
    else if (index < 5)
      index++;
    }
  if (isdigit(in_data[0]))
    return ( atoi( in_data ) );
  return( -1 );
}

//-------------------------------------------------------------------------

static int put_smtp_line(const char * line, unsigned int nchars, Network *net)
{
  #ifdef SHOWMAIL
  fwrite( line, nchars, 1, stdout );
  #endif 
  if ( net->Put( nchars, line ) )
    return -1;
  return (0);
}

//-------------------------------------------------------------------------

static int smtp_close_net( Network *net )
{
  if (net) NetClose(net);
  //if (net) { net->Close(); delete net; }
  //NetworkDeinitialize();
  return 0;
}  

//-------------------------------------------------------------------------
    
static Network *smtp_open_net( const char *smtphost, unsigned int smtpport )
{
  if ( smtpport == 0 || smtpport > 0xFFFE)
    smtpport = 25; //standard SMTP port
  if ( !smtphost || !*smtphost )
    smtphost = "127.0.0.1";   

  return NetOpen( smtphost, (s32)(smtpport) );
    
  #if 0
  Network *net;
  
  if (NetworkInitialize() < 0 ) 
    return NULL;

  if ((net = new Network( smtphost, smtphost, (s16)smtpport, 0 ))!=NULL)
    {
    if (!net->Open())
      net->MakeBlocking(); // no reason to be non-blocked
    else
      {
      delete net;
      net = NULL;
      }
    }

  if (!net)
    {
    #ifdef SHOWMAIL
    printf("SHOWMAIL: net->Open() failed\n");
    #endif
    NetworkDeinitialize();
    }

  return(net);
  #endif
}  

//-------------------------------------------------------------------------

//returns -1 if totally illegal address, +1 if addr is incomplete (truncated)
//this only handles single addresses. multi-address lines require tokenizing.
static int rfc822Address( char *buffer, const char *addr, 
                                       const char *host, const char **next )
{
  char pchar;
  int errcode = 0;
  int started = 0; 
  char *ptr = buffer;
  
  if (!addr)
    addr = "";

  while (*addr)
    {    
    if (*addr == '<')
      {
      ptr = buffer;
      if (started)
        errcode = -1;
      else
        {
        errcode = 0;
        addr++;
        while (*addr!='>')
          {
          if (!*addr)
            {
            ptr = buffer;
            errcode = -1;
            break;
            }    
          if (*addr==',' || !isprint(*addr) || isspace(*addr))
            {
            while (!isprint(*addr) || isspace(*addr))
            addr++;
            if (started && *addr!='>')
              {
              ptr = buffer;
              errcode = (*addr)?(-1):(+1);
              break;
              }
            }
          else //officially [A-Za-z0-9_.=\-]
            {
            *ptr++ = *addr++;
            started = 1;
            }
          }
        }
      break;
      }
    #if 0 //if strings in () or "" or '' ARE ALLOWED to be addresses
    if (*addr == '(' || *addr == '\"' || *addr == '\'')
      {
      pchar = ((*addr=='(')?(')'):(*addr));
      addr++;

      char *ptr2 = ptr;
      while (*addr && *addr!=pchar && (isspace(*addr) || !isprint(*addr)))
        addr++;
      while (*addr && *addr!=pchar)
        { 
        if (!started) 
          *ptr2++=*addr;
        *addr++;
        }
      if (*addr!=pchar)
        {
        errcode = +1;
        break;  
        }
      while (*addr && (*addr==',' || isspace(*addr) || !isprint(*addr)))
        addr++;
      if (started)
        break;    

      *ptr2=0;
      if (ptr2 > ptr)
        {
        do{    
          ptr2--;
          } while (ptr2>=ptr) && 
                 (isspace(*ptr2) || !isprint(*ptr2) || *ptr2==','))
        }
      if (ptr2 >= ptr)
        {
        *ptr2 = 0;
        ptr2 = ptr;
        while (*ptr2)
          {
          if (isspace(*ptr2) || !isprint(*ptr2) || *ptr2==',')
          break;
          }
        if (*ptr2==0 && strchr(ptr,'@')!=NULL)
          {
          ptr = ptr2;
          break;
          } 
        }
      *ptr = 0;
      }
    #else  //if strings in (), "" or '' ARE NOT ALLOWED to be addresses
    if (*addr == '(' || *addr == '\"' || *addr == '\'')
      {
      pchar = ((*addr=='(')?(')'):(*addr));
      addr++;
      while (*addr && *addr!=pchar)
        addr++;
      if (*addr!=pchar)
        {
        errcode=+1;
        break; 
        }
      addr++;
      while (*addr && (*addr==',' || isspace(*addr) || !isprint(*addr)))
        addr++;
      if (started)
        break;
      }
    #endif   
    else if (*addr==',' || isspace(*addr) || !isprint(*addr))
      {
      while (*addr && (*addr==',' || isspace(*addr) || !isprint(*addr)))
        addr++;
      if (started)
        break;    
      }  
    else
      {
      started = 1;
      *ptr++ = *addr++;
      }
    }
  *ptr = 0;
  if (next)
    *next = addr;
    
  if (!*buffer)
    {
    //strcpy( buffer, "postmaster" ); //leaving at <> will usually allow the
    if (!errcode) errcode = -1;       //msg through with a copy to postmaster
    }
  else
    { 
    ptr = strrchr( buffer, '@' );
    if (ptr && !ptr[1])
      {
      *ptr = 0;
      ptr = NULL;
      }
    if (!ptr && host && *host)
      {
      strcat( buffer, "@" );
      if ( isdigit( *host ) )
        strcat( buffer, "[" );
      strcat( buffer, host );
      if ( isdigit( *host ) )
        strcat( buffer, "]" );
      }
    }
  return errcode;
}  

//---------------------------------------------------------------------

//returns 0 if success, <0 if smtp error, >0 if network error (should defer)
static int smtp_open_message_envelope(Network *net, 
    const char *fromid, const char *destid, const char *smtphost )
{
  char out_data[300];
  const char *writefailmsg="[%s] Mail::Timeout waiting for SMTP server.\n";
  const char *errmsg = NULL;
  unsigned int pos;
    
  if ( get_smtp_result(net) != 220 ) //wait for server to welcome us
    errmsg = writefailmsg;
    
  if (!errmsg)
    {
    strcpy( out_data, "HELO " );
    pos = strlen( out_data );
    if (net->GetHostName( out_data+pos, 256)!=0)
      {
      out_data[pos]=0;
      strcat( out_data, "127.0.0.1" );//wicked! try it anyway
      }
    strcat( out_data, "\r\n" );  
    if ( put_smtp_line( out_data, strlen(out_data), net ))  
      errmsg = writefailmsg;
    else if ( get_smtp_result(net) != 250 )
      errmsg = "[%s] Mail::SMTP server refused our connection.\n";
    }
       
  if (!errmsg)  
    {
    strcpy( out_data, "MAIL From:<" );
    rfc822Address( out_data+11, fromid, smtphost, &fromid );
    strcat( out_data, ">\r\n" );
    if ( put_smtp_line( out_data, strlen(out_data), net) )  
      errmsg = writefailmsg;
    else if (get_smtp_result(net) != 250)
      errmsg = "[%s] Mail::SMTP server rejected sender name.\n";
    }

  if (!errmsg)
    {    
    unsigned int addrtries = 0, addrok = 0;
    while (!errmsg && *destid)
      {
      strcpy( out_data, "RCPT To:<" );
      pos = strlen( out_data );
      while (*destid && (*destid==',' || !isprint(*destid) ||isspace(*destid)))
        destid++;
      if ( *destid == '\"' || *destid == '(' || *destid == '\'')
        {
        out_data[pos] = ((*destid=='(')?(')'):(*destid));
        destid++;
        while (*destid && *destid!=out_data[pos])
          destid++;
        if (*destid)  
          destid++;  
        continue;
        }
      if ( !*destid )
        break;
      if ( rfc822Address( out_data+pos, destid, smtphost, &destid ) )
        break;
      strcat( out_data, ">\r\n" ); 
      addrtries++;     
      if ( put_smtp_line( out_data, strlen(out_data), net ))  
        errmsg = writefailmsg;
      else if (get_smtp_result(net) == 250)
        addrok++;
      }
    if (!errmsg)
      {
      if (addrtries==0)       
        errmsg = "[%s] Mail::Invalid or missing recipient address(es).\n";
      else if (addrok<addrtries) //this is not a fatal error, so continue.
        LogScreen( "[%s] Mail::One or more recipient addresses are invalid.\n",
         Time());
      }
    }

  if (!errmsg)
    {
    strcpy(out_data, "DATA\r\n");
    if ( put_smtp_line( out_data, strlen (out_data) , net))  
      errmsg = writefailmsg;
    if (get_smtp_result(net) != 354)
      errmsg = "[%s] Mail::SMTP server refused to accept message.\n";
    }
  
  if (errmsg)
    {
    LogScreen( errmsg, Time());
    return ((errmsg==writefailmsg)?(-1):(+1)); //retry hint
    }
  return(0);
}

//-------------------------------------------------------------------------

static char *rfc822Date(char *timestring)  //min 32 chars
{
  time_t timenow;
  struct tm * tmP;

  static const char *monnames[12] = {"Jan","Feb","Mar","Apr","May","Jun",
                                     "Jul","Aug","Sep","Oct","Nov","Dec" };
  static const char *wdaynames[7] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
  struct tm loctime, utctime;
  int haveutctime, haveloctime, tzdiff, abstzdiff;

  timestring[0]=0;

  timenow = time(NULL);
  tmP = localtime( (const time_t *) &timenow);
  if ((haveloctime = (tmP != NULL))!=0)
    memcpy( &loctime, tmP, sizeof( struct tm ));
  tmP = gmtime( (const time_t *) &timenow);
  if ((haveutctime = (tmP != NULL))!=0)
    memcpy( &utctime, tmP, sizeof( struct tm ));
  if (!haveutctime && !haveloctime)
    return timestring; // ""
  if (haveloctime && !haveutctime)
    memcpy( &utctime, &loctime, sizeof( struct tm ));
  else if (haveutctime && !haveloctime)
    memcpy( &loctime, &utctime, sizeof( struct tm ));

  tzdiff =  ((loctime.tm_min  - utctime.tm_min) )
           +((loctime.tm_hour - utctime.tm_hour)*60 );
  /* last two are when the time is on a year boundary */
  if      (loctime.tm_yday == utctime.tm_yday)     { ;/* no change */ }
  else if (loctime.tm_yday == utctime.tm_yday + 1) { tzdiff += 1440; }
  else if (loctime.tm_yday == utctime.tm_yday - 1) { tzdiff -= 1440; }
  else if (loctime.tm_yday <  utctime.tm_yday)     { tzdiff += 1440; }
  else                                             { tzdiff -= 1440; }

  abstzdiff = ((tzdiff<0)?(-tzdiff):(tzdiff));
  if (utctime.tm_wday<0 || utctime.tm_wday>6)
    {
    //for those (eg chinese, korean) locales that use w_day field as tm*
    #define dow(y,m,d) \
        ( ( ( 3*(y) - (7*((y)+((m)+9)/12))/4 + (23*(m))/9 + (d) + 2    \
        + (((y)-((m)<3))/100+1) * 3 / 4 - 15 ) % 7 ) )
    utctime.tm_wday=dow(loctime.tm_year+1900,loctime.tm_mon,loctime.tm_mday);
    #undef dow
    }
                      //5    3   4    3   3    3    2  1 1  2   2
  sprintf( timestring, "%s, %02d %s %02d %02d:%02d:%02d %c%02d%02d" ,
       wdaynames[loctime.tm_wday], loctime.tm_mday, monnames[loctime.tm_mon],
       loctime.tm_year, loctime.tm_hour, loctime.tm_min,
       loctime.tm_sec, ((tzdiff<0)?('-'):('+')), abstzdiff/60, abstzdiff%60);

  return(timestring);
}

// -----------------------------------------------------------------------

static int smtp_send_message_header( Network * net,  
                                char *fromid, char *destid, char *statsid )
{
  //fromid, destid and desthost would have been validated during
  //the EHLO/MAIL/RCPT exchange, so only need to check for statsid

  char buffer[512];
  int errcode = 0;
  char *p;

  if (errcode == 0) //send the senders address
    {
    sprintf( buffer, "From: %s", ((fromid && *fromid)?(fromid):("<>")) );
    if ( put_smtp_line( buffer, strlen(buffer), net ) ) 
      errcode = -1;
    }

  if (errcode == 0) //send the recipients address
    {
    sprintf( buffer, "\r\nTo: %s", ((destid && *destid)?(destid):("<>")));
    if ( put_smtp_line( buffer, strlen(buffer), net ) ) 
      errcode = -1;
    }

  if (errcode == 0)
    {
    p = (!statsid)?(NULL):(strchr(statsid,'@'));
    if ( p && strcmp( p, "@distributed.net" )!=0 )
      {
      sprintf( buffer,"\r\nErrors-to: %s", statsid );
      if ( put_smtp_line( buffer, strlen(buffer), net ) ) 
        errcode = -1;
      else
        { 
        sprintf( buffer,"\r\nReply-to: %s", statsid );
        if ( put_smtp_line( buffer, strlen(buffer), net ) ) 
          errcode = -1;
        }
      }
    }
    
  if (errcode == 0) //send the date 
    {
    sprintf( buffer, "\r\nDate: %s" 
        "\r\nX-Mailer: distributed.net RC5DES "CLIENT_VERSIONSTRING" client",
        rfc822Date( buffer + 256 ) ); 
    if ( put_smtp_line( buffer, strlen( buffer ), net ) ) 
      errcode = -1;
    }
  
  if (errcode == 0) //make sure mail forward agents don't screw with this
    { /*                                             
    strcpy( buffer, "\r\nMIME-Version: 1.0"
        "\r\nContent-Type: text/plain; charset=\"us-ascii\"" );
    if ( put_smtp_line( buffer, strlen( buffer ), net ) ) 
      errcode = -1; */
    }    
       
  if (errcode == 0) //send the subject
    {
    strcpy( buffer, "\r\nSubject: RC5DES stats (" );
    if ((net->GetHostName( buffer+25, 256 ))!=0) buffer[25] = 0;
    if ((!isdigit(buffer[25])) && ((p=strchr(buffer+25,'.'))!=NULL)) *p = 0;
    if ((buffer[25]) && ( statsid && *statsid )) strcat( buffer+25, ":" ); 
    if ( statsid && *statsid ) strcat( buffer+25, statsid );
    if ( buffer[25] ) strcat( buffer, ")" ); else buffer[23]=0;
    if ( put_smtp_line( buffer, strlen(buffer), net ) ) 
      errcode = -1;
    }

       
  if (errcode == 0) //finish off
    {
    if ( put_smtp_line( "\r\n\r\n", 4, net ) ) 
      errcode = -1;
    }
  return errcode;
}

//-------------------------------------------------------------------------

//returns 0 if success, <0 if smtp error, >0 if network error (should defer)
#if defined(MAILSPOOL_IS_AUTOBUFFER)
static int smtp_send_message_text(Network * net, const AutoBuffer &txt)
{
  AutoBuffer txtbuf(txt);       // make a working copy since we modify ours
  AutoBuffer netbuf;
  int errcode = 0;

  while (!errcode && txtbuf.RemoveLine(netbuf))
  {
    if (*netbuf.GetHead() == '.' && netbuf.GetLength() == 1)
    {
      // '.' on a new line?  convert to two dots
      netbuf.Reserve(4);
      strcat(netbuf.GetTail(), ".\r\n");
      netbuf.MarkUsed(3);
    }
    else
    {
      netbuf.Reserve(3);
      strcat(netbuf.GetTail(), "\r\n");
      netbuf.MarkUsed(2);
    }

    if ( put_smtp_line( netbuf, netbuf.GetLength(), net ) )
      errcode = -1;
  }
  return (errcode); // <=0
}
#else

#if defined(MAILSPOOL_IS_MEMFILE)
static int smtp_send_message_text(Network * net, const MEMFILE *mfile)
#else  // (defined(MAILSPOOL_IS_STATICBUFFER))
static int smtp_send_message_text(Network * net, const char *txt)
#endif
{
  char netbuf[512]; 
  unsigned int index=0;
  int eotext = 0, errcode = 0;
  char thischar, prevchar = 0;
  unsigned long txtlen, txtpos;
  
  #if defined(MAILSPOOL_IS_MEMFILE)
  txtlen = (unsigned long)(filelength( mfile ));
  mfrewind( mfile );
  #else 
  txtlen = (unsigned long)(strlen( txt ));
  #endif
  txtpos = 0;
  
  while (!errcode && !eotext)
    {
    #if defined(MAILSPOOL_IS_MEMFILE)
    if ( mfread( &thischar, 1, sizeof(char), mfile ) != 1 )
      break;
    #else
    if ((thischar = *txt)==0)
      break;
    ++txt;
    #endif

    eotext = ((++txtpos) == txtlen );
    
    if ((thischar == '.') && (prevchar == '\n'))  // '.' on a new line?
      {
      netbuf[index++]='.'; //convert to two dots (ie nextchar won't be a CR)
      netbuf[index++]='.';
      }
    else if (thischar == '\r')
      {
      if (txt[1]=='\r' && txt[2]=='\n') //ignore softbreaks "\r\r\n"
        {
        txt+=2;
        if (prevchar!=' ' && prevchar!='\t' && prevchar!='-')
          netbuf[index++]=' ';
        }
      else if (txt[1] != '\n')
        {
        netbuf[index++]='\r';
        netbuf[index++]='\n';
        }
      }
    else if (thischar == '\n')
      {
      if (prevchar != '\r') // all \n's should be preceeded by \r's...
        netbuf[index++]='\r';
      netbuf[index++]='\n';
      }
    else 
      netbuf[index++] = thischar;
    prevchar = (char)((index)?(netbuf[index-1]):(0));

    if ( eotext || (index >= (sizeof(netbuf)-10))) //little safety margin
      {
      if ( put_smtp_line( netbuf, index, net ) ) 
        errcode = -1;
      index = 0;
      }
    }
  return (errcode); // <=0
} 

#endif //defined(MAILSPOOL_IS_AUTOBUFFER) or other

//-------------------------------------------------------------------------

static int smtp_send_message_footer( Network *net )
{
  if ( put_smtp_line( "\r\n.\r\n", 5, net ) ) 
    return -1;
  if ( get_smtp_result(net) != 250 )
    {
    LogScreen("[%s] Mail::Message was not accepted by server.\n",Time());
    return +1;
    }
  return 0;
}

//-------------------------------------------------------------------------

static int smtp_close_message_envelope(Network * net)
{ return (put_smtp_line( "QUIT\r\n", 6 , net)?(-1):(0)); }

//-------------------------------------------------------------------------

unsigned long smtp_countspooled( struct mailmessage *msg )
{
  if (((long)(msg->sendthreshold)) <= 0 ) //sanity check
    msg->sendthreshold = 0;
  if (msg->sendthreshold == 0)
    return 0;

  #if (defined(MAILSPOOL_IS_AUTOBUFFER))
    {
    if (msg->spoolbuff) 
      return (unsigned long)( msg->spoolbuff->GetLength() );
    }
  #elif (defined(MAILSPOOL_IS_MEMFILE))
    {
    if (msg->spoolbuff) 
      return (unsigned long)( mfilelength( mfileno( msg->spoolbuff ) ) );
    }
  #else
    {
    if (msg->sendthreshold)
      return (unsigned long)( strlen( msg->spoolbuff ) );
    }
  #endif
  return 0;
}

//-------------------------------------------------------------------------

int smtp_deinitialize_message( struct mailmessage *msg ); //fwd resolution

//returns 0 if success, <0 if send error, >0 no network (should defer)
int smtp_send_message( struct mailmessage *msg )
{
  int errcode = 0;
  Network *net;

  if (msg->sendpendingflag == 0) //no changes since we last tried to send
    return 0;
  msg->sendpendingflag = 0; //protect against recursive calls

  #ifdef SHOWMAIL
  LogScreen("SHOWMAIL: beginning send(): spool length: %d bytes\n", 
                               (int)(smtp_countspooled(msg)) );
  #endif

  if (smtp_countspooled( msg ) == 0)
    return 0;

  if ((net = smtp_open_net( msg->smtphost, msg->smtpport )) == NULL)
    return +1; //retry hint

  //---------------
  if (errcode == 0)
    errcode = smtp_open_message_envelope(net,msg->fromid,msg->destid,NULL);
  if (errcode == 0)
    errcode = smtp_send_message_header(net, msg->fromid,msg->destid,msg->rc5id);
  if (errcode == 0)
    errcode = smtp_send_message_text( net, msg->spoolbuff );
  if (errcode == 0) 
    errcode = smtp_send_message_footer( net );
  if (errcode >= 0)                             
    smtp_close_message_envelope( net );  // always send QUIT unless net error
  //---------------

  if (errcode > 0) //smtp error (error message has already been printed)
    LogScreen("[%s] Mail::Message has been discarded.\n", Time() );
  else if ( errcode < 0 ) //net error (only send_envelope() said something)
    LogScreen("[%s] Mail::Network error. Send cancelled.\n", Time());
  else //if (errcode == 0)  //no error - yippee
    LogScreen("[%s] Mail::Message has been sent.\n", Time() );

  //---------------

  if ( errcode >= 0 ) // always clear message unless net error
    smtp_deinitialize_message( msg );

  smtp_close_net(net);
  return(errcode);
}

//-------------------------------------------------------------------------

int smtp_append_message( struct mailmessage *msg, const char *txt )
{
  unsigned long txtlen, msglen;

  if (((long)(msg->sendthreshold)) <= 0 ) //sanity check
    msg->sendthreshold = 0;
  if (msg->sendthreshold == 0)
    return 0;
  if (msg->sendthreshold < 1024)  //max size before we force send
    msg->sendthreshold = 1024;

  txtlen = strlen( txt );
  msg->sendpendingflag = 1;    

  #ifdef SHOWMAIL
  LogScreen("SHOWMAIL: appending %d bytes to mail spool.\n"
            "          max spool len: %d, old spool len: %d\n", 
	    (int)txtlen,
    #if (defined(MAILSPOOL_IS_AUTOBUFFER) || defined(MAILSPOOL_IS_MEMFILE))
       (int)((msg->sendthreshold/10)*11),
    #else
       (int)(MAILBUFFSIZE),
    #endif       
       (int)(smtp_countspooled(msg)) );
  #endif
  
  #if (defined(MAILSPOOL_IS_AUTOBUFFER))
    {
    msg->maxspoolsize = ((msg->sendthreshold/10)*11);

    if (txtlen > 0)
      {
      if (msg->spoolbuff == NULL)
        msg->spoolbuff = new Autobuffer();
      if (msg->spoolbuff != NULL)
        {
        msglen = (unsigned long)(msg->spoolbuff->GetLength());
	      
        if (( msglen + txtlen ) >= ( msg->maxspoolsize )) 
          {
          msg->spoolbuff->Reserve(((s32)(txtlen))); //
          strncpy(msg->spoolbuff->GetTail(), txt, txtlen);
          smtp_send_message( msg );
          msglen = 0;
          if ( msg->spoolbuff ) //net error, message still there
            {
            msg->spoolbuff->Clear();  //clear out the old message
            msg->spoolbuff->Reserve(((s32)(txtlen))+2048); 
            strncpy(msg->spoolbuff->GetTail(), txt, txtlen);
            msg->spoolbuff->MarkUsed(txtlen);
            msglen = txtlen;
            }
          }
        else
          {
          msg->spoolbuff->Reserve(((s32)(txtlen))+2048); //
          strncpy(msg->spoolbuff->GetTail(), txt, txtlen);
          msg->spoolbuff->MarkUsed(txtlen);
          msglen = (unsigned long)(msg->spoolbuff->GetLength());
          }
        }
      }
    }
  #elif defined(MAILSPOOL_IS_MEMFILE)
    {
    msg->maxspoolsize = ((msg->sendthreshold/10)*11);

    msglen = 0;
    if (txtlen > 0 ) 
      {  
      if (msg->spoolbuff == NULL)
        msg->spoolbuff = mfopen( "mail spool", "w+b" );
      if (msg->spoolbuff != NULL)
        {
        msglen = mfilelength( mfileno( msg->spoolbuff ) );

        if (( msglen + txtlen ) >= ( msg->maxspoolsize )) 
          {
          smtp_send_message( msg );
          if ( msg->spoolbuff == NULL )    //message got sent
            msg->spoolbuff = mfopen( "mail spool", "w+b" );
          else
            mftruncate( msg->spoolbuff, 0 );
          msglen = 0;  
          if ( msg->spoolbuff != NULL ) //message still there or recreated
            {
            if ( mfwrite( txt, txtlen, sizeof(char), msg->spoolbuff )!=txtlen )
              mftruncate( msg->spoolbuff, 0 );
            else
              msglen = txtlen;
            }
          }
        else
          {  
          mfseek( msg->spoolbuff, 0, SEEK_END );
          if ( mfwrite( txt, txtlen, sizeof(char), msg->spoolbuff ) != txtlen )
            {
            msg->maxspoolsize = msglen;
            msg->sendthreshold = ((msglen/10)*9);
            if (msg->sendthreshold == 0)
              msglen = 0;
            mftruncate( msg->spoolbuff, msglen );  
            if (msglen == 0)
              return -1;
            smtp_send_message( msg ); //try and send
            mftruncate( msg->spoolbuff, 0 );  //truncate if send failed
            if (mfwrite( txt, txtlen, sizeof(char), msg->spoolbuff )!=txtlen)
              {                                //if no mem to add txt,
              msg->sendthreshold = 0;          //stop the add process.
              mftruncate( msg->spoolbuff, 0 ); 
              return -1;
              }
            }
          }
        }    
      }
    }
  #else //if (defined(MAILSPOOL_IS_STATICBUFFER))
    {
    msg->maxspoolsize = MAILBUFFSIZE;
    if (msg->sendthreshold > ((msg->maxspoolsize/10)*9))
      msg->sendthreshold = ((msg->maxspoolsize/10)*9);

    unsigned long maxlen = msg->maxspoolsize;
    msglen = (unsigned long)(strlen( msg->spoolbuff ));
    
    if (txtlen > 0 ) 
      {  
      if (( msglen + txtlen + 1) >= maxlen ) 
        {
        if ( (txtlen + 1) >= maxlen ) //shouldn't happen
          {
          char *p = (char *)strchr( (txt + ((txtlen + 1)- maxlen )),'\n');
          if ( p != NULL )
            txt = (const char *)(p);  
          else //try to find some kind of sensible place to start
            {
            while (*txt && *txt!=' ' && *txt!='\t' && *txt!='\r')
              txt++;
            }
          while (*txt == '\n' || *txt == '\r')
            txt++;
          txtlen = strlen( txt );
          msg->spoolbuff[0]=0;
          msglen = 0;
          }
        maxlen -= (txtlen + 1); 
        if (msglen >= maxlen)
          {
          char *p = NULL;
          p = strchr( (msg->spoolbuff + (msglen-maxlen)), '\n' );
          if ( p != NULL )
            {
            while ( *p == '\n' || *p == '\r' )
            p++;
            }
          if (!p || !*p) 
            {
            msg->spoolbuff[0]=0;
            msglen = 0;
            }
          else
            {
            msglen = strlen( p );
            memmove( msg->spoolbuff, p, msglen );
            }
          }
        }  
      strcat(msg->spoolbuff,txt);
      if ('\r' == txt[txtlen-1]) 
        strcat(msg->spoolbuff,"\n");
      }
    }
  #endif
  
  #ifdef SHOWMAIL
  LogScreen("          new spool length: %d bytes\n", 
                               (int)(smtp_countspooled(msg)) );
  #endif

  if (smtp_countspooled( msg ) > msg->sendthreshold ) //crossed the threshold?
    return smtp_send_message( msg );  
  return 0;
}

//-------------------------------------------------------------------------

int smtp_clear_message( struct mailmessage *msg )
{
  #if (defined(MAILSPOOL_IS_AUTOBUFFER))
    {
    if (msg->spoolbuff)
      {
      delete (msg->spoolbuff);
      msg->spoolbuff = NULL;
      }
    }
  #elif defined(MAILSPOOL_IS_MEMFILE)
    {
    if (msg->spoolbuff)
      {
      mfclose(msg->spoolbuff);
      msg->spoolbuff = NULL;
      }
    }
  #else // (defined(MAILSPOOL_IS_STATICBUFFER))
    {
    msg->spoolbuff[0]=0;
    }
  #endif
  return 0;
}

int smtp_deinitialize_message( struct mailmessage *msg )
{
  smtp_send_message( msg );
  smtp_clear_message( msg );
  return 0;
}

//-------------------------------------------------------------------------

int smtp_initialize_message( struct mailmessage *msg, unsigned long sendthresh,
               const char *smtphost, unsigned int smtpport, const char *fromid,
                                        const char *destid, const char *rc5id )
{
  if (!msg) return -1;
  memset( (void *)(msg), 0, sizeof( struct mailmessage ));
  if (smtphost)   strncpy( msg->smtphost, smtphost, sizeof(msg->smtphost)-1);
  if (smtpport)   msg->smtpport = smtpport;
  if (fromid)     strncpy( msg->fromid, fromid, sizeof(msg->fromid)-1);
  if (destid)     strncpy( msg->destid, destid, sizeof(msg->destid)-1);
  if (rc5id)      strncpy( msg->rc5id, rc5id, sizeof(msg->rc5id)-1);
  if (sendthresh) msg->sendthreshold = sendthresh;
  
  return 0;
}    

