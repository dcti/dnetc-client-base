/*
 * Created by Tim Charron (tcharron@interlog.com) 97.9.17
 * Complete rewrite by Cyrus Patel (cyp@fb14.uni-mainz.de) 1998/08/15
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *mail_cpp(void) {
return "@(#)$Id: mail.cpp,v 1.32.2.12 2000/10/24 21:36:35 cyp Exp $"; }

//#define SHOWMAIL    // define showmail to see mail transcript on stdout

#include "baseincs.h"
#include "version.h"  /* for mail header generation */
#include "logstuff.h" /* LogScreen() */
#include "triggers.h" /* CheckExitRequestTrigger() */
#include "netconn.h"  /* netconn_xxx() */
#include "util.h"     /* TRACE_OUT */
#include "mail.h"     /* keep prototypes in sync */

#if (CLIENT_OS == OS_OS390)
  #define __hton_mem(__ebcdic_data,__len) __etoa_l(__ebcdic_data,__len)
  #define __ntoh_mem(__ascii_data,__len)  __atoe_l(__ascii_data,__len)
#else
  #define __hton_mem(__ebcdic_data,__len) /* nothing */
  #define __ntoh_mem(__ascii_data,__len)  /* nothing */
#endif

/* --------------------------------------------------------------------- */

#if (defined(MAILSPOOL_IS_AUTOBUFFER))
  #include "autobuff.h"
#elif defined(MAILSPOOL_IS_MEMFILE)
  #include "memfile.h"
#elif (defined(MAILSPOOL_IS_STATICBUFFER))
  #include <limits.h>
  #define MAILBUFFSIZE 150000L
  #if (UINT_MAX < MAILBUFFSIZE)
    #undef MAILBUFFSIZE
    #define MAILBUFFSIZE 32000
  #endif
#else
  #define MAILSPOOL_IS_MALLOCBUFFER
#endif  

//=========================================================================
// SMTP primitives (all static)
//=========================================================================

static int get_smtp_result( void * net )
{
  char in_data[10];
  int idx = 0;
  in_data[3] = 0;

  for (;;)
  {
    if (netconn_read( net, &in_data[idx], 1 )!=1)
      return(-1);
    __ntoh_mem( &in_data[idx], 1 );
    #ifdef SHOWMAIL
    LogRaw( "%c", in_data[idx] );
    #endif
    if (in_data[idx] == '\n')
    {
      if ( in_data[3] != '-') //if not an ESMPT multi-line reply
        break;
      idx = 0;
    }
    else if (idx < 5)
      idx++;
  }
  if (isdigit(in_data[0]))
    return ( atoi( in_data ) );
  return( -1 );
}

//-------------------------------------------------------------------------

static int put_smtp_line(void * net, char * line, unsigned int nchars)
{
  int rc;
  #ifdef SHOWMAIL
  for (unsigned int i=0;i<nchars;i++)
    LogRaw( "%c", line[i] );
  #endif
  __hton_mem( line, nchars );
  rc = netconn_write(net, line, nchars);
  __ntoh_mem( line, nchars );
  if (rc != ((int)nchars) )
    return -1;
  return (0);
}

//-------------------------------------------------------------------------

static int smtp_close_net( void *net )
{
  netconn_close(net);
  return 0;
}

//-------------------------------------------------------------------------

static void *smtp_open_net( const char *smtphost, unsigned int smtpport )
{
  void *net;
  if ( smtpport == 0 || smtpport > 0xFFFE)
    smtpport = 25; //standard SMTP port
  if ( !smtphost || !*smtphost )
    smtphost = "127.0.0.1";

  #ifdef SHOWMAIL
  LogRaw("Mail: NetOpen(\"%s\",%d)\n",smtphost,smtpport);
  #endif
  net = netconn_open( smtphost, smtpport, 1, -1, 0, 0, 0, 0 );
  #ifdef SHOWMAIL
  LogRaw("Mail: NetOpen(\"%s\",%d) = %p\n",smtphost,smtpport,net);
  #endif
  return net;
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
static int smtp_open_message_envelope(void *net,
    const char *fromid, const char *destid, const char *smtphost )
{
  char out_data[300];
  const char *writefailmsg="Timeout waiting for SMTP server.";
  const char *errmsg = NULL;
  unsigned int pos; int rc;

  if ( (rc = get_smtp_result(net)) != 220 ) //wait for server to welcome us
  {
    errmsg = writefailmsg;
    if (rc <= 0)
      errmsg = "";
  }

  if (!errmsg)
  {
    strcpy( out_data, "HELO " );
    pos = strlen( out_data );
    if (netconn_getname( net, &out_data[pos], 256 )!=0)
    {
      u32 addr = netconn_getaddr( net ); 
      if (!addr)
        strcpy( &out_data[pos], "127.0.0.1" );//wicked! try it anyway
      else
      { 
        char *p = (char *)(&addr);
        sprintf( &out_data[pos], "%d.%d.%d.%d",
               (p[0]&255), (p[1]&255), (p[2]&255), (p[3]&255) );
      }
    }
    strcat( out_data, "\r\n" );
    if ( put_smtp_line( net, out_data, strlen(out_data) ))
      errmsg = writefailmsg;
    else if ( (rc = get_smtp_result(net)) != 250 )
    {
      errmsg = "SMTP server refused our connection.";
      if (rc <= 0)
        errmsg = "";
    }
  }

  if (!errmsg)
  {
    strcpy( out_data, "MAIL From:<" );
    rfc822Address( out_data+11, fromid, smtphost, &fromid );
    strcat( out_data, ">\r\n" );
    if ( put_smtp_line( net, out_data, strlen(out_data)) )
      errmsg = writefailmsg;
    else if ((rc = get_smtp_result(net)) != 250)
    {
      errmsg = "SMTP server rejected sender name.";
      if (rc <= 0)
        errmsg = "";
    }
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
      if ( put_smtp_line( net, out_data, strlen(out_data) ))
        errmsg = writefailmsg;
      else if (get_smtp_result(net) == 250)
        addrok++;
    }
    if (!errmsg)
    {
      if (addrtries==0)
        errmsg = "Invalid or missing recipient address(es).";
      else if (addrok<addrtries) //this is not a fatal error, so continue.
        LogScreen( "One or more recipient addresses are invalid.");
    }
  }

  if (!errmsg)
  {
    strcpy(out_data, "DATA\r\n");
    if ( put_smtp_line( net, out_data, strlen (out_data) ))
      errmsg = writefailmsg;
    if ((rc = get_smtp_result(net)) != 354)
    {
      errmsg = "SMTP server refused to accept message.";
      if (rc <= 0)
        errmsg = "";
    } 
  }

  if (errmsg)
  {
    if (*errmsg)
      LogScreen( "Mail::%s\n", errmsg );
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
  if      (loctime.tm_yday == utctime.tm_yday)   { ;/* no change */ }
  else if (loctime.tm_yday == utctime.tm_yday + 1) { tzdiff += 1440; }
  else if (loctime.tm_yday == utctime.tm_yday - 1) { tzdiff -= 1440; }
  else if (loctime.tm_yday <  utctime.tm_yday)   { tzdiff += 1440; }
  else                                           { tzdiff -= 1440; }

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
                      //5    2   4    5   3    3    2  1 1  2   2 = 31
  sprintf( timestring, "%s, %02d %s %04d %02d:%02d:%02d %c%02d%02d" ,
       wdaynames[loctime.tm_wday], loctime.tm_mday, monnames[loctime.tm_mon],
       loctime.tm_year+1900, loctime.tm_hour, loctime.tm_min,
       loctime.tm_sec, ((tzdiff<0)?('-'):('+')), abstzdiff/60, abstzdiff%60);

  return(timestring);
}

// -----------------------------------------------------------------------

static int smtp_send_message_header( void * net,
                                char *fromid, char *destid, char *statsid )
{
  //fromid, destid and desthost would have been validated during
  //the EHLO/MAIL/RCPT exchange, so only need to check for statsid

  char buffer[512];
  int len, errcode = 0;
  char *p;

  if (statsid)
  {
    if (!*statsid)
      statsid = (char *)0;     
    else
    {
      p = strchr( statsid, '@' );
      if (p)
      {
        p = strstr(p,"distributed.net");
        if (p)
        {
          if ( ((*(p-1 ))=='@'  || (*(p-1 ))=='.') && 
               ((*(p+15))=='\0' || (*(p+15))=='.') )
          {     
            statsid = (char *)0;
          }
        }
      }
    }
  }

  if (destid)
  {
    if (!*destid)
      destid = (char *)0;
  }
  if (!destid)
    destid = statsid;

    if (fromid)
    {
      if (!*fromid)
        fromid = (char *)0;
    }
    if (!fromid)
      fromid = destid;

  if (errcode == 0) //send the senders address
  {
    sprintf( buffer, "From: %s", ((fromid)?(fromid):("<>")) );
    if ( put_smtp_line( net, buffer, strlen(buffer) ) )
      errcode = -1;
  }

  if (errcode == 0) //send the recipients address
  {
    sprintf( buffer, "\r\nTo: %s", ((destid)?(destid):("<>")));
    if ( put_smtp_line( net, buffer, strlen(buffer) ) )
      errcode = -1;
  }

  if (errcode == 0)
  {
    if (statsid)
    {
      sprintf( buffer,"\r\nErrors-to: %s", statsid );
      if ( put_smtp_line( net, buffer, strlen(buffer) ) )
        errcode = -1;
      else
      {
        sprintf( buffer,"\r\nReply-to: %s", statsid );
        if ( put_smtp_line( net, buffer, strlen(buffer)) )
          errcode = -1;
      }
    }
  }

  if (errcode == 0) //send the date
  {
    sprintf( buffer, "\r\nDate: %s"
        "\r\nX-Mailer: distributed.net v"CLIENT_VERSIONSTRING
           " client for "CLIENT_OS_NAME, rfc822Date( buffer + 256 ) );
    if ( put_smtp_line( net, buffer, strlen( buffer ) ) )
      errcode = -1;
  }

  #if 0
  if (errcode == 0) //make sure mail forward agents don't screw with this
  { 
    strcpy( buffer, "\r\nMIME-Version: 1.0"
        "\r\nContent-Type: text/plain; charset=\"us-ascii\"" );
    if ( put_smtp_line( net, buffer, strlen( buffer ) ) )
      errcode = -1; 
  }
  #endif

  if (errcode == 0) //send the subject
  {
    len = strlen( strcpy( buffer, "\r\nSubject: distributed.net client log (" ) );
    if ((netconn_getname( net, buffer+len, sizeof(buffer)>>1 ))!=0) 
      buffer[len] = '\0';
    else
    {
      p = &buffer[len];
      while (*p=='.' || isdigit(*p))
        p++;
      if (*p) /* hostname is not an IP address */
      {       /* so drop all but the first part of the hostname */
        p = strchr(&buffer[len],'.');
        if (p) *p = '\0';
      }
      if ((buffer[len]) && ( statsid && *statsid )) 
        strcat( buffer+len, ":" );
    }
    if ( statsid && *statsid ) 
      strcat( buffer, statsid );
    if ( !buffer[len] ) 
      buffer[len-1] = '\0';
    else
      strcat( buffer, ")" ); 
    if ( put_smtp_line( net, buffer, strlen(buffer) ) )
      errcode = -1;
  }

  if (errcode == 0) //finish off
  {
    if ( put_smtp_line( net, "\r\n\r\n", 4 ) )
      errcode = -1;
  }
  return errcode;
}

//-------------------------------------------------------------------------

//returns 0 if success, <0 neterror+sentsome, >0 neterror+sentnone (can retry)
static int smtp_send_message_text(void * net, 
                                  #if defined(MAILSPOOL_IS_MEMFILE)
                                  const MEMFILE *mfile
                                  #else /* STATICBUFFER or MALLOCBUFFER */
                                  const char *txt 
                                  #endif
                                  )
{
  char netbuf[512];
  unsigned int idx=0;
  int errcode = 0, eotext = 0, sentsome = 0;
  char thischar, prevchar = 0;
  unsigned long txtlen, txtpos;

  #if defined(MAILSPOOL_IS_MEMFILE)
  if (!mfile) return 0;
  txtlen = (unsigned long)( mfilelength( mfileno( mfile ) ) );
  mfrewind( mfile );
  #else /* MALLOCBUFFER or STATICBUFFER */
  if (!txt) return 0;
  txtlen = (unsigned long)(strlen( txt ));
  #endif

  txtpos = 0;
  while (!errcode && !eotext)
  {
    #if defined(MAILSPOOL_IS_MEMFILE)
    if ( mfread( &thischar, 1, sizeof(char), mfile ) != 1 )
      thischar = 0;
    #else
    thischar = *txt;
    if (thischar) txt++;
    #endif
 
    eotext = (thischar == 0);
    if (!eotext)
      eotext = ((++txtpos) == txtlen);

    if ((thischar == '.') && (txtpos == 0 || (prevchar == '\n')))
    {                                              /* '.' on a new line? */
      if (!eotext) /* if last char discard it */
      {
        netbuf[idx++]='.'; //convert to two dots (ie nextchar won't be a CR)
        netbuf[idx++]='.';
      }
    }
    else if (thischar == '\r')
    {
      char sbcheck[2] = {0,0};
      unsigned int nn;
      for (nn = 0; !eotext && nn < 2; nn++)
      {
        #if defined(MAILSPOOL_IS_MEMFILE)
        if ( mfread( &thischar, 1, sizeof(char), mfile ) != 1 )
          thischar = 0;
        #else
        thischar = *txt;
        if (thischar) txt++;
        #endif
        eotext = (thischar == 0);
        if (!eotext)
          eotext = ((++txtpos) == txtlen);
        if (!eotext)
        {
          sbcheck[nn] = thischar;
          if (nn == 0 && thischar != '\r')
            break;
        }
      }
      if (!eotext)
      {
        if (sbcheck[0] == '\r')
        {
          if (sbcheck[1] == '\n') //ignore softbreaks "\r\r\n" 
          {                         
            if (prevchar!=' ' && prevchar!='\t' && prevchar!='-')
              netbuf[idx++]=' ';
          }
          else
          {
            netbuf[idx++]='\r';
            netbuf[idx++]='\n'; 
            netbuf[idx++]='\r';
            netbuf[idx++]='\n'; 
            netbuf[idx++] = sbcheck[1];
          }
        }
        else 
        {
          netbuf[idx++]='\r';
          netbuf[idx++]='\n'; 
          if (sbcheck[0] != '\n') /* not CRLF pair */
            netbuf[idx++] = sbcheck[0];
        }
      }
    }
    else if (thischar == '\n')
    {
      if (prevchar != '\r') // all \n's should be preceeded by \r's...
        netbuf[idx++]='\r';
      netbuf[idx++]='\n';
    }
    else if (thischar)
    {
      netbuf[idx++] = thischar;
    }
    prevchar = (char)((idx)?(netbuf[idx-1]):(0));

    if ( eotext || (idx >= (sizeof(netbuf)-10))) //little safety margin
    {
      if ( put_smtp_line( net, netbuf, idx ) )
        errcode = -1;
      else 
        sentsome = 1;
      idx = 0;
    }
  }
  if (errcode != 0)
  {
    errcode = +1; /* we can retry */
    if (sentsome)
      errcode = -1;
  }
  return (errcode);
}

//-------------------------------------------------------------------------

static int smtp_send_message_footer( void *net )
{
  int rc;
  if ( put_smtp_line( net, "\r\n.\r\n", 5 ) )
    return -1;
  if ( (rc = get_smtp_result(net)) != 250 )
  {
    if (rc <= 0)
      return +1;
    LogScreen("Mail::Message was not accepted by server.\n");
  }
  return -1;
}

//-------------------------------------------------------------------------

static int smtp_close_message_envelope(void * net)
{ 
  return (put_smtp_line( net, "QUIT\r\n", 6)?(-1):(0)); 
}

//=========================================================================
// MAILMESSAGE
//=========================================================================

#define MAILMESSAGE_SIGNATURE 0x4d4d5347ul  /* ('MMSG') */

struct mailmessage
{
  unsigned long signature; /* 'MMSG' (0x4D4D5347) */
  unsigned long sendthreshold; 
  unsigned long maxspoolsize;
  int send_reentrancy_guard;   /* \  Log()->Mail()->Net()->Log()->Mail->... */
  int append_reentrancy_guard; /* /  condition guard */
  int sendpendingflag; //0 after a send() (successful or not), 1 after append()
    
  #if (defined(MAILSPOOL_IS_AUTOBUFFER))
    AutoBuffer *spoolbuff;
  #elif defined(MAILSPOOL_IS_MEMFILE)
    MEMFILE *spoolbuff;
  #elif (defined(MAILSPOOL_IS_STATICBUFFER))
    char spoolbuff[MAILBUFFSIZE];
  #elif (defined(MAILSPOOL_IS_MALLOCBUFFER))
    char *spoolbuff;
  #endif

  char fromid[256];
  char destid[256];
  char rc5id[256];
  char smtphost[256];
  unsigned int smtpport;
};    

static struct mailmessage *__smpt_convert_handle_to_mailmessage( void *msghandle )
{
  if (msghandle)
  {
    if (((struct mailmessage *)msghandle)->signature == MAILMESSAGE_SIGNATURE)
      return (struct mailmessage *)msghandle;
  }
  return (struct mailmessage *)0;
}

//-------------------------------------------------------------------------

unsigned long smtp_countspooled( void *msghandle )
{
  struct mailmessage *msg = __smpt_convert_handle_to_mailmessage( msghandle );
  if (!msg)
    return 0;
  if (((long)(msg->sendthreshold)) <= 0 ) //sanity check
    msg->sendthreshold = 0;
  if (msg->sendthreshold == 0)
    return 0;

  #if (defined(MAILSPOOL_IS_MEMFILE))
  {
    if (msg->spoolbuff)
      return (unsigned long)( mfilelength( mfileno( msg->spoolbuff ) ) );
  }
  #elif (defined(MAILSPOOL_IS_MALLOCBUFFER))
  {
    if (msg->spoolbuff)
      return (unsigned long)( strlen( msg->spoolbuff ) );
  }
  #elif (defined(MAILSPOOL_IS_STATICBUFFER))
  {
    if (msg->sendthreshold)
      return (unsigned long)( strlen( msg->spoolbuff ) );
  }
  #endif
  return 0;
}

//-------------------------------------------------------------------------

int smtp_clear_message( void *msghandle )
{
  struct mailmessage *msg = __smpt_convert_handle_to_mailmessage( msghandle );
  if (!msg) 
    return -1;
  #if defined(MAILSPOOL_IS_MEMFILE)
  {
    if (msg->spoolbuff)
    {
      mfclose(msg->spoolbuff);
      msg->spoolbuff = NULL;
    }
  }
  #elif (defined(MAILSPOOL_IS_MALLOCBUFFER))
  {
    if (msg->spoolbuff)
      free(((void *)(msg->spoolbuff)));
    msg->spoolbuff = NULL;
    msg->maxspoolsize = 0;
  }
  #elif (defined(MAILSPOOL_IS_STATICBUFFER))
  {
    msg->spoolbuff[0]=0;
  }
  #endif
  return 0;
}

//-------------------------------------------------------------------------

//returns 0 if success, <0 if send error, >0 no network (should defer)
int smtp_send_message( void *msghandle )
{
  int errcode, deferred;
  struct mailmessage *msg = __smpt_convert_handle_to_mailmessage( msghandle );
  if (!msg)
    return -1;

  errcode = deferred = 0;
  if ((++msg->send_reentrancy_guard) == 1)
  {
    if (msg->sendpendingflag != 0) //have changes since we last tried to send
    {
      msg->sendpendingflag = 0;

      #ifdef SHOWMAIL
      LogRaw("SHOWMAIL: beginning send(): spool length: %d bytes\n",
                                  (int)(smtp_countspooled(msg)) );
      #endif

      if (smtp_countspooled( msghandle ) != 0)
      {
        const char *resmsg = (const char *)0;
        void *net = smtp_open_net( msg->smtphost, msg->smtpport );

        if (!net)
        {
          errcode = +1; //retry hint
          deferred = 1;
        }
        else
        {
          errcode = smtp_open_message_envelope(net,msg->fromid,msg->destid,NULL);
          if (errcode > 0) /* net error (retry later) */
          {
            resmsg = "Failed to send envelope.";
            deferred = 1;
          }
          else /* <0=no smtp ack, 0=success */
          {
            if (errcode == 0) /* success */
            {
              errcode = smtp_send_message_header(net, msg->fromid,msg->destid,msg->rc5id);
              if (errcode == 0) /* success */
              {
                int sendres = smtp_send_message_text( net, msg->spoolbuff );
                //0=success, <0 neterror+sentsome, >0 neterror+sentnone (can retry)
                if (sendres < 0)
                {
                  errcode = +1; /* net error */
                  smtp_clear_message( msghandle );
                  resmsg = "Mail message discarded.";
                }
                else if (sendres > 0)
                {
                  errcode = +1; /* net error */
                  deferred = 1;
                }
                else /* sendres == 0 */
                {
                  smtp_clear_message( msg );
                  resmsg = "Message has been sent.";
                  errcode = 0; /* no error */
                  deferred = 0; /* don't defer */
                }
                if (errcode <= 0) /* only if no net error */
                  errcode = smtp_send_message_footer( net ); /* close the message */
              }
            }
            if (errcode <= 0)  /* only if no net error */
              smtp_close_message_envelope( net ); // QUIT unless net error
          } 
          smtp_close_net(net);
        }
        if (!resmsg && deferred)
          resmsg = "";
        if (resmsg)
        {
           LogScreen("Mail::%s%s\n", resmsg,
           ((deferred && !CheckExitRequestTriggerNoIO())?(" Send deferred."):("")));
        }  
      }  /* if (smtp_countspooled( msghandle ) != 0) */
    }  /* if (msg->sendpendingflag != 0) */
  } /* if ((++msg->send_reentrancy_guard) == 1) */
  msg->send_reentrancy_guard--;
  
  return(errcode);
}

//-------------------------------------------------------------------------

int smtp_send_if_needed( void *msghandle )
{
  struct mailmessage *msg = __smpt_convert_handle_to_mailmessage( msghandle );
  if (!msg)
    return -1;
  if (smtp_countspooled( msghandle ) >= (( msg->sendthreshold / 10) * 9))
    return smtp_send_message( msghandle );
  return 0;  
}

//-------------------------------------------------------------------------

int smtp_append_message( void *msghandle, const char *txt )
{
  int retcode = 0;
  struct mailmessage *msg = __smpt_convert_handle_to_mailmessage( msghandle );
  if (!txt || !msg)
    return 0;

  if (((long)(msg->sendthreshold)) <= 0 ) //sanity check
    msg->sendthreshold = 0;
  if (msg->sendthreshold == 0)
    return 0;
  if (msg->sendthreshold < 1024)  //max size before we force send
    msg->sendthreshold = 1024;

  TRACE_OUT((+1,"smtp_append_message()\n"));

  if ((++msg->append_reentrancy_guard)==1)  
  {
    #if defined(MAILSPOOL_IS_MEMFILE)
    {
#error Untested
      unsigned long txtlen = (unsigned long)strlen( txt );
      msg->maxspoolsize = ((msg->sendthreshold/10)*11);
      if (txtlen > 0 )
      {
        if (msg->spoolbuff == NULL)
          msg->spoolbuff = mfopen( "mail spool", "w+b" );
        if (msg->spoolbuff != NULL)
        {
          register char *p;
          unsigned long maxsize = msg->maxspoolsize;
          unsigned long msglen = mfilelength( mfileno( msg->spoolbuff ) );
          if ((txtlen + 2) > maxsize)
          {
            txt += ((txtlen+2)-maxsize);
            while (*txt && (*txt != '\n' && *txt != '\r'))
              txt++;
            while (*txt == '\n' || *txt == '\r')
              txt++;
            txtlen = strlen(txt);
            mftruncate( mfileno(msg->spoolbuff), 0 );
            msglen = 0;
          }
          else if ((msglen + txtlen +2) > maxsize)
          {
            unsigned long woffset = 0, roffset = maxsize-(txtlen+2);
            char cpybuff[128];
            unsigned int readsize;
            mfseek( msg->spoolbuff, roffset, SEEK_SET );
            while ((readsize = mfread( cpybuff, sizeof(cpybuff), sizeof(char), 
                   msg->spoolbuff)) != 0)
            {
              unsigned int pos;
              for (pos = 0; pos < readsize; pos++)
              {
                roffset++;
                if (cpybuf[pos] == '\r' || cpybuf[pos] == '\n')
                {
                  pos = 0; /* '\r' || '\n' */
                  while (readsize) /* dummy */
                  {
                    mfseek( msg->spoolbuff, roffset, SEEK_SET );
                    if ((readsize = mfread( cpybuff, sizeof(cpybuff), 
                                   sizeof(char), msg->spoolbuff)) != 0)
                    {
                      p = cpybuff;
                      roffset += readsize;
                      if (pos == 0)  /* still marking '\r' || '\n' */
                      {
                        while (readsize > 0 && (*p == '\r' || *p == '\n'))
                        {
                          p++;
                          readsize--;
                        }
                        pos = readsize; /* for next pass */
                      }
                      if (readsize)
                      {
                        mfseek( msg->spoolbuff, woffset, SEEK_SET );
                        mfwrite( p, readsize, sizeof(char), msg->spoolbuff);
                        woffset += readsize;
                      }
                    }
                  }
                  break;
                }
              }
            }
            mftruncate( mfileno(msg->spoolbuff), woffset );
            msglen = woffset;
          }
          if (txtlen && (msglen + txtlen +2) <= maxsize)
          {
            mfseek( msg->spoolbuff, 0, SEEK_END );
            if (mfwrite( txt, txtlen, sizeof(char), msg->spoolbuff) == txtlen )
              msg->sendpendingflag = 1; /* buffer has changed */
            else /* "write" failed - zap the buffer */
              mftruncate( mfileno(msg->spoolbuff), 0 );
          }
        }
      }
    }
    #else /* MAILSPOOL_IS_MALLOCBUFFER or MAILSPOOL_IS_STATICBUFFER */
    {
      unsigned long txtlen = (unsigned long)strlen( txt );
      if (txtlen > 0)
      {
        register char *p;
        unsigned long maxsize;
        
        #if defined(MAILSPOOL_IS_STATICBUFFER)
        msg->maxspoolsize = sizeof(msg->spoolbuff); /*MAILBUFFSIZE;*/
        if (msg->sendthreshold > ((msg->maxspoolsize/10)*9))
          msg->sendthreshold = ((msg->maxspoolsize/10)*9);
        #else /* defined(MAILSPOOL_IS_MALLOCBUFFER) */
        if (msg->spoolbuff == NULL) /* msg->maxspoolsize is zero */
        {
          msg->maxspoolsize = 0;
          maxsize = msg->sendthreshold;
          if (maxsize > (UINT_MAX - (4096*2)))
            maxsize = (UINT_MAX - (4096*2));
          if (maxsize < 1024)
            maxsize = 1024;
          msg->sendthreshold = maxsize;
          maxsize = msg->sendthreshold + 4096; /* a little padding */
          do
          {
            unsigned int mallocsize = ((unsigned int)maxsize) - 16;
            p = (char *)malloc( mallocsize );
            if (p)
            {
              msg->maxspoolsize = mallocsize;
              p[0] = 0;
              msg->spoolbuff = p;
              msg->sendthreshold = msg->maxspoolsize - 4096;
              break;
            }
            maxsize -= 4096;
          } while (maxsize >= (4096*2));
        }  
        #endif

        if (msg->maxspoolsize) /* msg->spoolbuff is not NULL */
        {
          unsigned long msglen = (unsigned long)strlen(msg->spoolbuff);
          maxsize = (unsigned long)msg->maxspoolsize;
        
          if ((txtlen + 2) > maxsize) 
          {
            txt += ((txtlen+2)-maxsize);
            while (*txt && (*txt != '\n' && *txt != '\r'))
              txt++;
            while (*txt == '\n' || *txt == '\r')
              txt++;
            txtlen = strlen(txt);
            msg->spoolbuff[0] = '\0';
            msglen = 0;
          }
          else if ((msglen + txtlen + 2) > maxsize)
          {
            p = &msg->spoolbuff[maxsize-(txtlen+2)];
            while (*p && *p != '\n' && *p != '\r')
              p++;
            while (*p == '\n' || *p == '\r')
              p++;
            msglen = strlen(p);
            if (msglen)
              memmove( msg->spoolbuff, p, msglen );
            msg->spoolbuff[msglen] = '\0';
          }
  
          if (txtlen && (msglen + txtlen + 2) <= maxsize)
          {
            msg->sendpendingflag = 1; /* buffer has changed */
            strcat( msg->spoolbuff, txt );
          }
        }
      }
    }
    #endif
    #ifdef SHOWMAIL
    LogRaw("new spool length: %d bytes\n",  (int)(smtp_countspooled(msg)) );
    #endif

    if (smtp_countspooled( msg ) > msg->sendthreshold ) //crossed the threshold?
      retcode = smtp_send_message( msg );

  } /* if ((++msg->append_reentrancy_guard) == 1) */
  msg->append_reentrancy_guard--;

  TRACE_OUT((-1,"smtp_append_message() =>%d\n",retcode));
  return retcode;
}

//-------------------------------------------------------------------------

static void cleanup_field( int atype, char *addr, const char *default_addr )
{
  const char sig[]="distributed.net";
  char *p = addr;
  unsigned int len;
  
  while (*addr && isspace(*addr))
    addr++;
  if (addr != p)
    strcpy( p, addr );
  addr = p;
  if (atype == 1 || atype == 2  /* single address or hostname */)
  {
    while (*p && !isspace(*p))
      p++;
    *p = 0;
  }

  len = strlen( addr );
  while (len > 0 && isspace(addr[len-1]))
    addr[--len]=0;

  /* reject if lc "^distributed.net$" or "*[\@|\.]distributed.net$" */ 
  if (len>=(sizeof(sig)-1))
  {
    char scratch[sizeof(sig)+1];
    strncpy(scratch,&(addr[(len-(sizeof(sig)-1))]),sizeof(scratch));
    scratch[sizeof(scratch)-1] = '\0';
    p = scratch;
    while (*p) 
    {
      *p = (char)tolower(*p);
      p++;
    }  
    if ( strcmp( sig, scratch ) == 0 )
    {  
      if (len==(sizeof(sig)-1))
        len = 0;
      else if (addr[(len-(sizeof(sig)-1))-1]=='.' || addr[(len-(sizeof(sig)-1))-1]=='@')
        len = 0;
    }    
  }

  if (len == 0)
    strcpy( addr, default_addr );
  return;
}

//-------------------------------------------------------------------------

void *smtp_construct_message(unsigned long sendthresh,
               const char *smtphost, unsigned int smtpport, const char *fromid,
                                        const char *destid, const char *rc5id )
{
  void *msghandle = (void *)0;
  TRACE_OUT((+1,"smtp_contruct_message()\n"));
  if (sendthresh > 0)
  {
    msghandle = malloc(sizeof(struct mailmessage));
    if (msghandle)
    {
      struct mailmessage *msg = (struct mailmessage *)msghandle;
      memset( msghandle, 0, sizeof( struct mailmessage ));

      msg->signature = MAILMESSAGE_SIGNATURE;
      msg->sendthreshold = sendthresh;
      msg->smtpport = smtpport;

      TRACE_OUT((0,"msg->sendthreshold=%lu\n", msg->sendthreshold ));
      if (smtphost)   
      {
        strncpy( msg->smtphost, smtphost, sizeof(msg->smtphost));
        msg->smtphost[sizeof(msg->smtphost)-1] = '\0';   
      }
      cleanup_field( 2, msg->smtphost, "" /* localhost */ );
      TRACE_OUT((0,"msg->smtphost:port='%s':%d\n", msg->smtphost, msg->smtpport ));    
      if (rc5id)      
      {
        strncpy( msg->rc5id, rc5id, sizeof(msg->rc5id));
        msg->rc5id[sizeof(msg->rc5id)-1] = '\0';   
      }
      cleanup_field( 1, msg->rc5id, "" );
      TRACE_OUT((0,"msg->rc5id='%s'\n", msg->rc5id ));    
      if (fromid) /* else inherit from rc5id */
      {
        strncpy( msg->fromid, fromid, sizeof(msg->fromid));
        msg->fromid[sizeof(msg->fromid)-1] = '\0';
      }
      cleanup_field( 1, msg->fromid, msg->rc5id ); //if fromid or destid are zero
      TRACE_OUT((0,"msg->fromid='%s'\n", msg->fromid ));    
      if (destid) /* else inherit from rc5id */
      {
        strncpy( msg->destid, destid, sizeof(msg->destid));
        msg->destid[sizeof(msg->destid)-1] = '\0';
      }
      cleanup_field( 0, msg->destid, msg->rc5id ); //len, they default to the rc5id
      TRACE_OUT((0,"msg->destid='%s'\n", msg->destid ));    

      if (strstr(msg->destid,"@distributed.net"))
      {                                            /* no mail */
        free(msghandle);
        msghandle = (void *)0;
      }
    } /* if msghandle */
  } /* if sendthresh > 0 */
  TRACE_OUT((-1,"smtp_contruct_message() => %p\n",msghandle));
  return msghandle;
}

//-------------------------------------------------------------------------
    
int smtp_destruct_message(void *msghandle)
{
  struct mailmessage *msg = __smpt_convert_handle_to_mailmessage( msghandle );
  if (!msg)
    return -1;
  TRACE_OUT((+1,"smtp_destruct_message()\n"));
  smtp_send_message( msghandle );
  smtp_clear_message( msghandle );
  memset((void *)msghandle, 0, sizeof(struct mailmessage));
  free(msghandle);
  TRACE_OUT((-1,"smtp_destruct_message() => 0\n"));
  return 0;
}  

