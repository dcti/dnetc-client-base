// Created by Tim Charron (tcharron@interlog.com) 97.9.17
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: mail.cpp,v $
// Revision 1.20  1998/08/10 20:29:39  cyruspatel
// Call to gethostname() is now a call to Network::GetHostName(). Updated
// send routine to reflect new NetworkInitialize()/NetworkDeinitialize()
// requirements. Removed all references to NO!NETWORK.
//
// Revision 1.19  1998/08/02 16:18:06  cyruspatel
// Completed support for logging.
//
// Revision 1.18  1998/08/02 03:16:54  silby
// Major reorganization:  Log,LogScreen, and LogScreenf are now in logging.cpp, and are global functions - client.h #includes logging.h, which is all you need to use those functions.  Lurk handling has been added into the Lurk class, which resides in lurk.cpp, and is auto-included by client.h if lurk is defined as well. baseincs.h has had lurk-specific win32 includes moved to lurk.cpp, cliconfig.cpp has been modified to reflect the changes to log/logscreen/logscreenf, and mail.cpp uses logscreen now, instead of printf. client.cpp has had variable names changed as well, etc.
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
// ambiguities. ("declaration/type or an expression")
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
return "@(#)$Id: mail.cpp,v 1.20 1998/08/10 20:29:39 cyruspatel Exp $"; }
#endif

#include "network.h"
#include "baseincs.h"
#include "sleepdef.h"
#include "mail.h"
#include "logstuff.h"
#include "triggers.h"
#include "clitime.h"
#define Time() (CliGetTimeString(NULL,1))

//-------------------------------------------------------------------------

#undef SHOWMAIL               // define showmail to see mail transcript on stdout
#define FIFO_ON_BUFF_OVERFLOW // if defined, lines are thrown out as needed.
                              // if !defined, then all old text is discarded.

//-------------------------------------------------------------------------

char *rfc822Date(void); // at the end of this file

MailMessage::MailMessage(void)
{
   messagetext[0]=0;
   messagelen=10000;
   strcpy(fromid,"RC5notify");
   smtp[0]=0;
   destid[0]=0;
   strcpy(my_hostname, "");
   strcpy(rc5id,"");
   port=25;
   timeoffset=0;
}

//-------------------------------------------------------------------------

MailMessage::~MailMessage()
{
  timeoffset=0; // nothing to do. - suppress compiler warning.
}

//-------------------------------------------------------------------------

void MailMessage::checktosend( u32 forcesend)
{
  int retry;
   
  if (strlen(messagetext) == 0) 
    {
    this->inittext(0);
    }
  if (messagelen != 0 && strlen(messagetext) != 0)
    {
    if ((strlen(messagetext) >= messagelen) || forcesend) 
      {
      retry=0;
      //sendmessage returns 0 if success, <0 if send error, >0 no network (defer)
      do{
        if (this->sendmessage() >= 0) 
          break;
        ++retry;
        #ifndef FIFO_ON_BUFF_OVERFLOW
        if (retry > 3)
          {
          LogScreen("[%s] Mail::sendmessage - - Giving up. Contents discared.\n", Time() );
          strcpy(messagetext,"");
          break;
          }
        #endif
        LogScreen("[%s] Mail::sendmessage %d - Unable to send mail message.\n", Time(), (int) retry);
        } while (retry <= 3);
      }
    }
  return;
}

//-------------------------------------------------------------------------

void MailMessage::addtomessage(char *txt ) 
{
  if (messagelen != 0) 
    {
    if ((strlen(messagetext)+strlen(txt)+1) >= MAILBUFFSIZE) 
      { // drop old mail due to overflow?
      #ifdef FIFO_ON_BUFF_OVERFLOW
      int len;
      if ( ((len=strlen(txt))+1) >= MAILBUFFSIZE ) // should never happen
        return;
      else
        {
        char *p=strchr(messagetext+len,'\n');
        if (p != NULL)
          strcpy( messagetext, p+1 ); // move the whole lot up
        else
          messagetext[0]=0; //no '\n' terminated lines in buffer anyway.
        }
      #else
      strcpy(messagetext,"");
      #endif
      }
    if (strlen(messagetext) == 0) 
      {
      this->inittext(0);
      }
    strcat(messagetext,txt);
    if ('\r' == txt[strlen(txt)-1]) 
      {
      strcat(messagetext,"\n");
      }
    if (strlen(messagetext) >= (messagelen+5000)) 
      { 
      checktosend(1);
      }
    } 
  return;
}

//-------------------------------------------------------------------------

int MailMessage::inittext(int out)
{
  if (messagelen != 0) 
    {
    if (messagelen<500) 
       messagelen=500;
     if (messagelen>MAXMAILSIZE) 
       messagelen=MAXMAILSIZE;
    }
  if (out == 1) 
    {
    LogScreen("Mail server:port is %s:%d\n", smtp, (int) port);
    LogScreen("Mail id is %s\n", fromid);
    LogScreen("Destination is %s\n", destid);
    LogScreen("Message length set to %d\n", (int) messagelen);
    LogScreen("RC5id set to %s\n", rc5id);
    }
  return(0);
}

//-------------------------------------------------------------------------

//returns 0 if success, <0 if send error, >0 no network (should defer)

int MailMessage::sendmessage()
{
  // Get the SMTP server name, destination mailid from the INI file...
  s32 retry;
  int retcode;
  Network *net;

  retcode = 0;
  net = NULL;
  
  if (messagelen == 0) 
    return 0;
  
  if (NetworkInitialize() < 0)
    return +1;
  
  net = new Network( (const char *) smtp , (const char *) smtp, (s16) port, 0 );
  if (!net) 
    {
    retcode = +1;
    }
    
  if (retcode == 0)
    {
    net->quietmode = quietmode;
    retry=0;
    do 
      {
      if (!net->Open())
        break;
      if ((++retry) == 3)
        {
        retcode = -1;
        break;
        }
      if (!CheckExitRequestTrigger())
        {
        LogScreen("[%s] Network::MailMessage %d - Unable to open connection to smtp server\n", Time(), (int) retry );
        sleep( 3 );
        }
      if (CheckExitRequestTrigger())
        retcode = -1;
      } while (!retcode);
    }

  if (retcode == 0)
    {
    net->MakeBlocking(); // This message is sent byte-by-byte -- there's no reason to be non-blocked

    if (prepare_smtp_message(net) == -1)
      {
      LogScreen("[%s] Error in prepare_smtp_message\n", Time());
      retcode = -1;
      }
    }
  
  if (retcode == 0)
    {
    if (0 == send_smtp_edit_data(net))
      {
      finish_smtp_message(net);
      LogScreen("[%s] Mail message has been sent.\n", Time());
      }
    else
      retcode = -1;
    }

  if (net)
    {
    net->Close();
    delete net;
    }
    
  NetworkDeinitialize();

  if (!retcode)               //clear message if no errors
    strcpy(messagetext,"");

  return(retcode);
}

//-------------------------------------------------------------------------

// 'destination' is the address the message is to be sent to
// 'message' is a pointer to a null-terminated 'string' containing the
// entire text of the message.

int MailMessage::prepare_smtp_message(Network *net)
{
  char out_data[255];
  char str[1024];
  char destidcopy[255];
  char *ptr;
  int len, startLen;

  if ( get_smtp_line(net) != 220 )
    {
    smtp_error (net,"SMTP server error");
    return(-1);
    }

  net->GetHostName(my_hostname, 255);

  sprintf( out_data, "HELO %s\r\n", my_hostname );
  if (0 != put_smtp_line( out_data, strlen (out_data), net ))  
    {
    smtp_error(net,"SMTP server error");
    return -1;
    }

  if ( get_smtp_line(net) != 250 )
    {
    smtp_error (net,"SMTP server error");
    return -1;
    }

  sprintf (out_data, "MAIL From:<%s>\r\n", fromid);
  if (0 != put_smtp_line( out_data, strlen (out_data) , net))  
    {
    smtp_error(net,"SMTP server error");
    return -1;
    }

  if (get_smtp_line(net) != 250)
    {
    smtp_error (net,"The mail server doesn't like the sender name,\nhave you set your mail address correctly?\n");
    return -1;
    }

  // do a series of RCPT lines for each name in address line
  strcpy( (char *)(&destidcopy[0]), (char const *)(&destid[0]) );
  for (ptr = destidcopy; *ptr; ptr += len + 1)
    {
    // if there's only one token left, then len will = startLen,
    // and we'll iterate once only
    startLen = strlen (ptr);
    if ((len = strcspn (ptr, " ,\n\t\r")) != startLen)
      {
      ptr[len] = '\0';                  // replace delim with NULL char
      while (strchr (" ,\n\t\r", ptr[len+1]))   // eat white space
        ptr[len++] = '\0';
      }

    sprintf (out_data, "RCPT To: <%s>\r\n", ptr);
    if (0 != put_smtp_line( out_data, strlen (out_data) , net ))  
      {
      smtp_error(net,"SMTP server error");
      return -1;
      } 

    if (get_smtp_line(net) != 250)
      {
      sprintf (str, "The mail server doesn't like the name %s.\nHave you set the 'To' field correctly\n",ptr);
      smtp_error (net,str);
      return -1;
      }

    if (len == startLen)        // last token, we're done
      break;
    }

  sprintf (out_data, "DATA\r\n");
  if (0 != put_smtp_line( out_data, strlen (out_data) , net))  
    {
    smtp_error(net,"SMTP server error");
    return -1;
    }

  if (get_smtp_line(net) != 354)
    {
    smtp_error(net,"Mail server error accepting message data");
    return -1;
    }

  return(0);
}

//-------------------------------------------------------------------------

int MailMessage::send_smtp_edit_data (Network * net)
{
  transform_and_send_edit_data(net);

  if (get_smtp_line(net) != 250)
    {
    smtp_error (net,"Message not accepted by server");
    return -1;
    }
  #ifdef FIFO_ON_BUFF_OVERFLOW   // send OK. The buffer is finally emptied
  strcpy(messagetext,"");
  #endif
  return(0);
}

//-------------------------------------------------------------------------

int MailMessage::get_smtp_line( Network * net )
{
  char in_data[10];
  int index = 0;
  in_data[3] = 0;

  while (1)
    {
    if ( net->Get( 1, &in_data[index], 2*NETTIMEOUT ) != 1)
      {
      LogScreen("[%s] Mailmessage: recv error\n", Time());
      return(-1);
      }
    #ifdef SHOWMAIL
    LogScreen("%s%c", ((index==0)?("GET: "):("")), in_data[index] );
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
  return( atoi( in_data ) );
}

//-------------------------------------------------------------------------

int MailMessage::put_smtp_line( const char * line, unsigned int nchars , Network *net)
{

  #ifdef SHOWMAIL
  LogScreen("PUT: %s",line);
  #endif
  //  delay(1); //Some servers can't talk too fast.
  if ( net->Put( nchars, line ) )
    {
    LogScreen("[%s] Mailmessage:Send error\n", Time());
    return -1;
    }
  return (0);
}

//-------------------------------------------------------------------------

int MailMessage::finish_smtp_message(Network * net)
{
  return put_smtp_line( "QUIT\r\n", 6 , net);
}

//-------------------------------------------------------------------------

int send_message_header( Network * net, char *desthost, char *fromhost,
                 char *destid, char *fromid, char *statsid, char *date )
{
   //fromid, destid and desthost would have been validated
   //during the EHLO/MAIL/RCPT exchange, so only need to check for
   //statsid and fromhost

   if ( net->Put( 6, "From: " ) ) return(-1);
   if ( net->Put( strlen(fromid), fromid ) ) return(-1);
   if (!strchr(fromid, '@') )
     {
     if ( net->Put( 1, "@" ) ) return(-1);
     if ( net->Put( strlen(desthost), desthost ) ) return(-1);
     }
   if ( net->Put( 6, "\r\nTo: " ) ) return(-1);
   if ( net->Put( strlen(destid), destid ) ) return(-1);
   if (!strchr(fromid, '@') )
     {
     if ( net->Put( 1, "@" ) ) return(-1);
     if ( net->Put( strlen(desthost), desthost ) ) return(-1);
     }
   if (date && *date)
     {
     if ( net->Put( 8, "\r\nDate: " ) ) return(-1);
     if ( net->Put( strlen(date), date ) ) return(-1);
     }
   if ( net->Put( 23, "\r\nSubject: RC5DES stats" ) ) return(-1);
   if (( fromhost && *fromhost) || ( statsid && *statsid ))
     {
     if ( net->Put( 2, " (" )) return (-1);
     if ( fromhost && *fromhost)
       {
       char *p = strchr( fromhost, '.' );
       unsigned int len = strlen(fromhost);
       if ( p && p!=fromhost ) len = (p-fromhost)-1;
       if ( len && net->Put( len, fromhost ) ) return(-1);
       if ( statsid && *statsid && len && net->Put( 1, ":" )) return (-1);
       }
     if ( statsid && *statsid )
       {
       if ( net->Put( strlen(statsid), statsid ) ) return(-1);
       }
     if ( net->Put( 1, ")" )) return (-1);
     }
   if ( net->Put( 4, "\r\n\r\n" ) ) return(-1);

   return 0;
}

//-------------------------------------------------------------------------

int MailMessage::transform_and_send_edit_data(Network * net)
{
  char *index;
  char *header_end;
  char previous_char = 'x';
  char this_char;
  unsigned int send_len;
  bool done = 0;

  send_len = strlen(messagetext);
  index = messagetext;

  if (strlen( my_hostname )==0)
    net->GetHostName(my_hostname, 255);

  if ( send_message_header( net, smtp,   my_hostname,
                          destid, fromid, rc5id, rfc822Date() ) )
    return -1;
  header_end = messagetext;

  while (!done)
    {
    // room for extra char for double dot on end case
    while ((unsigned int) (index - messagetext) < send_len)
      {
      this_char = *index;
      #if defined(SHOWMAIL)
      LogScreen("%c",this_char);
      #endif
      //delay(1); //Some servers can't talk too fast.
      switch (this_char)
      {
      case '.':
         if (previous_char == '\n')
                      /* send _two_ dots... */
            if ( net->Put( 1, index ) ) {return(-1);}
            if ( net->Put( 1, index ) ) {return(-1);}
            break;
      case '\r':
                     // watch for soft-breaks in the header, and ignore them
         if (index < header_end && (strncmp (index, "\r\r\n", 3) == 0))
            index += 2;
         else
            if (previous_char != '\r')
               if ( net->Put( 1, index ) ) {return(-1);}
                     // soft line-break (see EM_FMTLINES), skip extra CR */
         break;
      case '\n': // all \n's should be preceeded by \r's...
         if (previous_char != '\r') {
            if ( net->Put( 2, "\r\n" ) ) {return(-1);}
         } else {
            if ( net->Put( 1, index ) ) {return(-1);}
         }
         break;
      default:
         if ( net->Put( 1, index ) ) {return(-1);}
      }
      previous_char = *index;
      index++;
    }
    if( (unsigned int) (index - messagetext) == send_len) done = 1;
  }

  // this handles the case where the user doesn't end the last
  // line with a <return>

  if (messagetext[send_len-1] != '\n')
  {
     if ( net->Put( 5, "\r\n.\r\n" ) ) {return(-1);}
  } else {
     if ( net->Put( 3, ".\r\n" ) ) {return(-1);}
  }

  return (0);
}

//-------------------------------------------------------------------------

void MailMessage::smtp_error (Network *net, const char * message)
{
  LogScreen("[%s] %s\n", Time(),message);
  if (net)
    {
    put_smtp_line("QUIT\r\n", 6,net);
    net->Close();
    }
  return;
}

//-------------------------------------------------------------------------

char *rfc822Date(void)
{
  time_t timenow;
  struct tm * tmP;
  static char timestring[32];

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

