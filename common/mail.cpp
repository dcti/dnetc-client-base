// Created by Tim Charron (tcharron@interlog.com) 97.9.17
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: mail.cpp,v $
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
// Revision 1.0  1997/09/17 09:17:07  timc
// Created
//-------------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *mail_cpp(void) {
return "@(#)$Id: mail.cpp,v 1.21 1998/08/15 18:09:45 cyruspatel Exp $"; }
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
  if (net) { net->Close(); delete net; }
  NetworkDeinitialize();
  return 0;
}  

//-------------------------------------------------------------------------
    
static Network *smtp_open_net( const char *smtphost, unsigned int smtpport )
{
  Network *net;
    
  if (NetworkInitialize() < 0 ) 
    return NULL;
    
  if ( smtpport == 0 || smtpport > 0xFFFE)
    smtpport = 25; //standard SMTP port
  if ( !smtphost || !*smtphost )
    smtphost = "127.0.0.1";   
    
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
    NetworkDeinitialize();
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
static int smtp_send_message_text(Network * net, const char *txt)
{
  char netbuf[512]; 
  unsigned int index=0;
  int errcode = 0;
  char thischar, prevchar = 0;
  
  while (!errcode && (thischar = *txt)!=0)
    {
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
    prevchar = ((index)?(netbuf[index-1]):(0));

    if ( (*(++txt))==0 || (index >= (sizeof(netbuf)-10)))//little safety margin
      {
      if ( put_smtp_line( netbuf, index, net ) ) 
        errcode = -1;
      index = 0;
      }
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
    LogScreen("[%s] Mail::Message was not accepted by server.\n",Time());
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
    return 0; //mail disabled. nothing to do
  if (msg->spoolbuff[0] == 0 ) 
    return 0; //nothing to do
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
    {   
    LogScreen("[%s] Mail::Message has been discarded.\n", Time() );
    msg->spoolbuff[0] = 0; //smtp error. not recoverable. so clear up
    }
  else if ( errcode < 0 ) //net error (only send_envelope() said something)
    { 
    LogScreen("[%s] Mail::Network error. Send cancelled.\n", Time());
    //we do not clear the message buffer. Could try again later.
    }
  else //if (errcode == 0)  //no error - yippee
    {   
    LogScreen("[%s] Mail::Message has been sent.\n", Time() );
    msg->spoolbuff[0] = 0; //successfully sent, so clear up
    }

  smtp_close_net(net);
  return(errcode);
}

//-------------------------------------------------------------------------

static int smtp_append_message( MailMessage *msg, const char *txt )
{
  size_t txtlen, msglen;
  
  if (msg->sendthreshold == 0)    //mail isn't initialized or is disabled 
    return 0;                          
  if (msg->sendthreshold < 1024)  //max size before we force send
    msg->sendthreshold = 1024;
  if (msg->sendthreshold > ((msg->spoolbuffmaxsize/10)*9))
    msg->sendthreshold = ((msg->spoolbuffmaxsize/10)*9);  

  txtlen = strlen( txt );
  msglen = strlen( msg->spoolbuff );
  if (txtlen > 0 ) 
    {  
    if (( msglen + txtlen + 1) >= msg->spoolbuffmaxsize) 
      {
      size_t maxlen = msg->spoolbuffmaxsize; 
      if ( (txtlen + 1) >= maxlen ) //shouldn't happen
        {
        char *p = (char *)strchr( (txt + ((txtlen + 1)-maxlen)),'\n');
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
          strcpy( msg->spoolbuff, p );  
          //memmove( msg->spoolbuff, p, msglen );
          }
        }
      }  
    strcat(msg->spoolbuff,txt);
    if ('\r' == txt[txtlen-1]) 
      strcat(msg->spoolbuff,"\n");
    }
  if ( msglen > msg->sendthreshold )   //crossed the threshold?
    return smtp_send_message( msg );  
  return 0;
}
  
// =========================================================================
// Class wrapper functions
// =========================================================================
// ---------------the rest are in the class definition ------------

//returns 0 if success, <0 if send error, >0 no network (should defer)
int MailMessage::send(void) 
{ return smtp_send_message(this); }

//returns same as send
int MailMessage::append(const char *txt)    
{ return smtp_append_message(this,txt); }

//-------------------------------------------------------------------------
