// Created by Tim Charron (tcharron@interlog.com) 97.9.17
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: mail.cpp,v $
// Revision 1.22  1998/08/15 21:32:11  jlawson
// updated mail to use autobuffer
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
// new inifile option: 'autofindkeyserver', ie if keyproxy= points to a
// xx.v27.distributed.net then that will be interpreted by Network::Resolve()
// to mean 'find a keyserver that covers the timezone I am in'. Network
// constructor extended to take this as an argument.
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
// Revision 1.0  1997/09/17 09:17:07  timc
// Created
//-------------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *mail_cpp(void) {
return "@(#)$Id: mail.cpp,v 1.22 1998/08/15 21:32:11 jlawson Exp $"; }
#endif

#include "network.h"
#include "autobuff.h"
#include "baseincs.h"
#include "sleepdef.h"
#include "mail.h"
#include "logstuff.h"
#include "triggers.h"
#include "clitime.h"

//-------------------------------------------------------------------------

//#define SHOWMAIL    // define showmail to see mail transcript on stdout

// Define to value to over-allocate the mail spool buffer by.  Over-
// allocating reduces wasted effort involved in frequently enlarging
// the buffer for small increments.
#define MAILBUFFER_INCREMENT 2048

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
  if (net) { net->Close(); delete net; }
  NetworkDeinitialize();
  return 0;
}

//-------------------------------------------------------------------------

static Network *smtp_open_net( const char *smtphost, s16 smtpport )
{
  Network *net;
  int errcode = 0;

  if (NetworkInitialize() < 0 )
    return NULL;

  if ( smtpport <= 0 )
    smtpport = 25; //standard SMTP port

  if ((net = new Network( smtphost, smtphost, (s16)smtpport, 0 ))!=NULL)
  {
    int retry = 0;
    do
    {
      if (!net->Open())
      {
        net->MakeBlocking(); //message is sent byte-by-byte,
        break;               //so no reason to be non-blocked
      }
      if ((++retry) == 3)
      {
        errcode = -1;
        break;
      }
      if (!CheckExitRequestTrigger())
      {
        LogScreen("[%s] Mail::Unable to establish SMTP connection (%d).\n",
                  (CliGetTimeString(NULL,1)), (int) retry );
        sleep( 3 );
      }
      if (CheckExitRequestTrigger())
        errcode = -1;
    } while (!errcode);
    if (errcode)
    {
      delete net;
      net = NULL;
    }
  }
  if (!net)
    smtp_close_net(NULL);
  return(net);
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
      while (*addr && (*addr == ',' || isspace(*addr) || !isprint(*addr)))
        addr++;
      if (started)
        break;

      *ptr2=0;
      if (ptr2 > ptr)
      {
        do {
          ptr2--;
        } while (ptr2>=ptr) && (isspace(*ptr2) ||
            !isprint(*ptr2) || *ptr2 == ','))
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
    strcpy( buffer, "postmaster" );

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
  return errcode;
}

//---------------------------------------------------------------------

//returns 0 if success, <0 if smtp error, >0 if network error (should defer)
static int smtp_open_message_envelope(Network *net, const char *fromid,
                                                     const char *destid)
{
  char out_data[300];
  char hostname[256];
  const char *writefailmsg="[%s] Mail::Timeout waiting for SMTP server.\n";
  const char *errmsg = NULL;
  unsigned int pos;

  if ( get_smtp_result(net) != 220 ) //wait for server to welcome us
    errmsg = writefailmsg;

  if (!errmsg)
  {
    strcpy( out_data, "HELO " );
    if (net->GetHostName( hostname, 256)==0)
      strcat( out_data, hostname );
    else
    {
      hostname[0]=0;
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
    rfc822Address( out_data+11, fromid, hostname, &fromid );
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
      if ( rfc822Address( out_data+pos, destid, hostname, &destid ) )
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
            (CliGetTimeString(NULL,1)));
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
    LogScreen( errmsg, (CliGetTimeString(NULL,1)));
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

static int smtp_send_message_header( Network * net, char *desthost,
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

  if (errcode == 0) //send the date
  {
    sprintf( buffer, "\r\nDate: %s", rfc822Date(buffer+256) );
    if ( put_smtp_line( buffer, strlen(buffer), net ) )
      errcode = -1;
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

//-------------------------------------------------------------------------

static int smtp_send_message_footer( Network *net )
{
  if ( put_smtp_line( "\r\n.\r\n", 5, net ) )
    return -1;
  if ( get_smtp_result(net) != 250 )
  {
    LogScreen("[%s] Mail::Message was not accepted by server.\n",
              (CliGetTimeString(NULL,1)));
    return +1;
  }
  return 0;
}

//-------------------------------------------------------------------------

static int smtp_close_message_envelope(Network * net)
{ return (put_smtp_line( "QUIT\r\n", 6 , net)?(-1):(0)); }

//-------------------------------------------------------------------------

//returns 0 if success, <0 if send error, >0 no network (should defer)
static int smtp_send_message( MailMessage *msg )
{
  int errcode = 0;
  Network *net;

  if (msg->sendthreshold == 0 )
    return 0; //nomail enabled. nothing to do
  if (msg->spoolbuff.GetLength() == 0 )
    return 0; //nothing to do
  if ((net = smtp_open_net( msg->smtphost, msg->smtpport )) == NULL)
    return +1; //retry hint

  //---------------
  if (errcode == 0)
    errcode = smtp_open_message_envelope(net, msg->fromid, msg->destid);
  if (errcode == 0)
    errcode = smtp_send_message_header(net, msg->smtphost, msg->fromid,
                                                 msg->destid, msg->rc5id);
  if (errcode == 0)
    errcode = smtp_send_message_text( net, msg->spoolbuff );
  if (errcode == 0)
    errcode = smtp_send_message_footer( net );
  if (errcode >= 0)
    smtp_close_message_envelope( net );  // always send QUIT unless net error
  //---------------

  if (errcode > 0) //smtp error (error message has already been printed)
  {
    LogScreen("[%s] Mail::Message has been discarded.\n",
              (CliGetTimeString(NULL,1)) );
    msg->spoolbuff.Clear(); //smtp error. not recoverable. so clear up
  }
  else if ( errcode < 0 ) //net error (only send_envelope() said something)
  {
    LogScreen("[%s] Mail::Network error. Send cancelled.\n",
              (CliGetTimeString(NULL,1)));
    //we do not clear the message buffer. Could try again later.
  }
  else //if (errcode == 0)  //no error - yippee
  {
    LogScreen("[%s] Mail::Message has been sent.\n",
              (CliGetTimeString(NULL,1)) );
    msg->spoolbuff.Clear(); //successfully sent, so clear up
  }

  smtp_close_net(net);
  return(errcode);
}

//-------------------------------------------------------------------------

static int smtp_append_message( MailMessage *msg, const char *txt )
{
  // mail isn't initialized or is disabled
  if (msg->sendthreshold == 0)
    return 0;

  // enforce minimum threshold size
  if (msg->sendthreshold < 1024)
    msg->sendthreshold = 1024;

  // add the text to our buffer
  int txtlen = strlen(txt);
  msg->spoolbuff.Reserve(txtlen + MAILBUFFER_INCREMENT);
  strncpy(msg->spoolbuff.GetTail(), txt, txtlen);
  msg->spoolbuff.MarkUsed(txtlen);

  // crossed the threshold?  force a send.
  if ( msg->spoolbuff.GetLength() > msg->sendthreshold )
    return smtp_send_message( msg );
  return 0;
}

// =========================================================================
// Class wrapper functions
// =========================================================================

//returns 0 if success, <0 if send error, >0 no network (should defer)
int MailMessage::send(void)
{ return smtp_send_message(this); }

//returns same as send
int MailMessage::append(const char *txt)
{ return smtp_append_message(this,txt); }

//-------------------------------------------------------------------------

