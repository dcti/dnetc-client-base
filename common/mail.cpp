// Created by Tim Charron (tcharron@interlog.com) 97.9.17
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: mail.cpp,v $
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
//



#undef SHOWMAIL               // define showmail to see mail transcript on stdout
#define GEN_HEADER_AT_SEND    // if defined, then the message header is
                              // generated and sent at message send time, and
                              // is not pre-made as part of the message itself
#define FIFO_ON_BUFF_OVERFLOW // if defined, lines are thrown out as needed.
                              // if !defined, then all old text is discarded.

#if defined(FIFO_ON_BUFF_OVERFLOW) && !defined(GEN_HEADER_AT_SEND)
#define GEN_HEADER_AT_SEND    // FIFO buffer requires generating headers at
#endif                        // send time rather than at msg creation time.

static const char *id="@(#)$Id: mail.cpp,v 1.11 1998/06/15 12:04:01 kbracey Exp $";

#include "network.h"
#include "client.h"
#include "mail.h"


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

MailMessage::~MailMessage()
{
  // nothing to do.
}

void MailMessage::checktosend( u32 forcesend)
{
#ifndef NONETWORK
   s32 retry;
   if (strlen(messagetext) == 0) {
      this->inittext(0);
   }
   if (messagelen != 0) {
      if ((strlen(messagetext) >= messagelen) || forcesend) {
         retry=0;
#ifdef FIFO_ON_BUFF_OVERFLOW
         while ((this->sendmessage() == -1) && retry++ < 3) {
#if !defined(NEEDVIRTUALMETHODS)
            printf("Mail::sendmessage %d - Unable to send mail message\n",
                  (int) retry );
#endif
            if (retry < 3) sleep(1);
         }
#else
         while ((this->sendmessage() == -1) && retry++ < 3) {
            if (retry == 3) {
#if !defined(NEEDVIRTUALMETHODS)
               printf("Mail::sendmessage %d - Unable to send mail message.  Contents discarded.\n", (int) retry);
#endif
            } else {
#if !defined(NEEDVIRTUALMETHODS)
               printf("Mail::sendmessage %d - Unable to send mail message\n", (int) retry );
#endif
               sleep( 1 );
            }
         }
         strcpy(messagetext,"");
#endif
      }
   }
#endif
}

void MailMessage::addtomessage(char *txt ) {
   if (messagelen != 0) {
      if ((strlen(messagetext)+strlen(txt)+1) >= MAILBUFFSIZE) { // drop old mail due to overflow?
#ifdef FIFO_ON_BUFF_OVERFLOW
        {
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
        }
#else
            strcpy(messagetext,"");
#endif
      }
      if (strlen(messagetext) == 0) {
         this->inittext(0);
      }
      strcat(messagetext,txt);
      if ('\r' == txt[strlen(txt)-1]) {
         strcat(messagetext,"\n");
      }
      if (strlen(messagetext) >= (messagelen+5000)) { // Although messages are sent when
                                         // Update() is called, this ensures that
                                         // mail is still send when using a shared buffer
                                         // (in which case, Update() may never get called)
         checktosend(1);
      }
   }
}

int MailMessage::inittext(int out)
{
  if (messagelen != 0) {
    if (messagelen<500) {
       messagelen=500;
    }
    if (messagelen>MAXMAILSIZE) {
       messagelen=MAXMAILSIZE;
    }
  }
#if !defined(NEEDVIRTUALMETHODS)
  if (out == 1) {
    printf("Mail server:port is %s:%d\n", smtp, (int) port);
    printf("Mail id is %s\n", fromid);
    printf("Destination is %s\n", destid);
    printf("Message length set to %d\n", (int) messagelen);
    printf("RC5id set to %s\n", rc5id);
  }
#endif

#ifndef NONETWORK
  gethostname(my_hostname, 255);   //This should be in network.cpp
#else
  strcpy(my_hostname, "No Network");
#endif

#ifndef GEN_HEADER_AT_SEND
  char dttext[255];
  time_t time_of_day;

  time_of_day = time( NULL );
  #if defined(_SUNOS3_)
    {
      struct tm *now = localtime(&time_of_day);
      const char *day;
      const char *month;
      switch (now->tm_wday)
      {
        case 0: day = "Sun"; break;
        case 1: day = "Mon"; break;
        case 2: day = "Tue"; break;
        case 3: day = "Wed"; break;
        case 4: day = "Thu"; break;
        case 5: day = "Fri"; break;
        default:
        case 6: day = "Sat"; break;
      }
      switch (now->tm_mon)
      {
        case 0: month = "Jan"; break;
        case 1: month = "Feb"; break;
        case 2: month = "Mar"; break;
        case 3: month = "Apr"; break;
        case 4: month = "May"; break;
        case 5: month = "Jun"; break;
        case 6: month = "Jul"; break;
        case 7: month = "Aug"; break;
        case 8: month = "Sep"; break;
        case 9: month = "Oct"; break;
        case 10: month = "Nov"; break;
        default:
        case 11: month = "Dec"; break;
      }
      sprintf(dttext, "%s, %02d %s %04d %02d:%02d:%02d", day, now->tm_mday,
              month, (now->tm_year+1900), now->tm_hour, now->tm_min,
              now->tm_sec);
    }
  #else
     strftime( dttext, 80, "%a, %d %b %Y %H:%M:%S",
                localtime( &time_of_day ) );
  #endif

   strcpy(messagetext,"Subject: RC5DES stats (");
   strcat(messagetext,my_hostname);
   strcat(messagetext,":");
   strcat(messagetext,rc5id);
   strcat(messagetext,")\r\nFrom: ");
   strcat(messagetext,fromid);
   if (!strchr(fromid, '@')) { strcat(messagetext,"@"); strcat(messagetext,smtp); }
   strcat(messagetext,"\r\nTo: ");
   strcat(messagetext,destid);
   strcat(messagetext,"\r\nDate: ");
   // Store the position where the date belongs...
   timeoffset = strlen(messagetext);
   strcat(messagetext,rfc822Date() /* dttext */);
   strcat(messagetext,"\r\n\r\n");

#endif // GEN_HEADER_AT_SEND
   return(0);
}

#ifndef NONETWORK
int MailMessage::sendmessage()
{
  // Get the SMTP server name, destination mailid from the INI file...
  s32 retry;
  Network *net;

#if (CLIENT_OS == OS_NETWARE)
  if ( !CliIsNetworkAvailable(0) )    //This should be made a generic
    return 0;                         //function in network.cpp
#endif


#ifndef GEN_HEADER_AT_SEND
// First, update the time in the header.  Otherwise, it will have the time that
// composition of the message started, instead of now.
  char *temptime = rfc822Date();
  for (u32 i = 0; i < 29; i++)
    messagetext[timeoffset+i] = *(temptime+i);
#endif

//   printf("Server: %s\n",smtp);
//   printf("Port:   %d\n",port);
//   printf("From:   %s\n",fromid);
//   printf("Dest:   %s\n",destid);

  if (messagelen != 0)
  {
    net = new Network( (const char *) smtp , (const char *) smtp, (s16) port );
    net->quietmode = quietmode;

    retry=0;
    while ((net->Open()) && retry++ < 3)
    {
      if (retry == 3)
      {
#ifndef FIFO_ON_BUFF_OVERFLOW
        strcpy(messagetext,"");
#endif
        delete net;
        return(-1);
      }
#if !defined(NEEDVIRTUALMETHODS)
      printf("Network::MailMessage %d - Unable to open connection to smtp server\n", (int) retry );
#endif
      sleep( 3 );
      // Unable to open network.
    }

    net->MakeBlocking(); // This message is sent byte-by-byte -- there's no reason to be non-blocked

    if (prepare_smtp_message(net) != -1)
    {
      if (0 == send_smtp_edit_data(net))
      {
        finish_smtp_message(net);
#if !defined(NEEDVIRTUALMETHODS)
        printf("Mail message has been sent.\n");
#endif
        net->Close();
        delete net;
        return(0);
      } else {
        net->Close();
        delete net;
        return(-1);
      }
    } else {
#if !defined(NEEDVIRTUALMETHODS)
      printf("Error in prepare_smtp_message\n");
#endif
      net->Close();
      return(-1);
    }
  }
  return(0);
}

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

  sprintf( out_data, "HELO %s\r\n", my_hostname );
  if (0 != put_smtp_line( out_data, strlen (out_data), net ))  {
     smtp_error(net,"SMTP server error");
     return -1;
  }

  if ( get_smtp_line(net) != 250 )
  {
    smtp_error (net,"SMTP server error");
    return -1;
  }

  sprintf (out_data, "MAIL From:<%s>\r\n", fromid);
  if (0 != put_smtp_line( out_data, strlen (out_data) , net))  {
     smtp_error(net,"SMTP server error");
     return -1;
  }

  if (get_smtp_line(net) != 250)
  {
    smtp_error (net,"The mail server doesn't like the sender name,\nhave you set your mail address correctly?\n");
    return -1;
  }

  // do a series of RCPT lines for each name in address line
  strcpy( (char *) &destidcopy, (char const *) &destid );
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
    if (0 != put_smtp_line( out_data, strlen (out_data) , net ))  {
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
  if (0 != put_smtp_line( out_data, strlen (out_data) , net))  {
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

int MailMessage::get_smtp_line( Network * net )
{
  char in_data[10];
  int index = 0;
  in_data[3] = 0;

  while (1)
    {
    if ( net->Get( 1, &in_data[index], 2*NETTIMEOUT ) != 1)
      {
#if !defined(NEEDVIRTUALMETHODS)
      printf("recv error\n");
#endif
      return(-1);
      }
#ifdef SHOWMAIL
printf("%s%c", ((index==0)?("GET: "):("")), in_data[index] );
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

int MailMessage::put_smtp_line( const char * line, unsigned int nchars , Network *net)
{

#ifdef SHOWMAIL
printf("PUT: %s",line);
#endif
//  delay(1); //Some servers can't talk too fast.
  if ( net->Put( nchars, line ) )
  {
#if !defined(NEEDVIRTUALMETHODS)
    printf("send error\n");
#endif
    return -1;
  }
  return (0);
}

int MailMessage::finish_smtp_message(Network * net)
{
  return put_smtp_line( "QUIT\r\n", 6 , net);
}

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
   if ( net->Put( 25, "\r\nSubject: RC5DES stats (" ) ) return(-1);
   if ( fromhost && *fromhost)
     {
     if ( net->Put( strlen(fromhost), fromhost ) ) return(-1);
     if ( net->Put( 1, ":" ) ) return(-1);
     }
   if ( statsid && *statsid )
     {
     if ( net->Put( strlen(statsid), statsid ) ) return(-1);
     }
   if ( net->Put( 5, ")\r\n\r\n" ) ) return(-1);

   return 0;
}

int MailMessage::transform_and_send_edit_data(Network * net)
{
  char *index;
  char *header_end;
  char previous_char = 'x';
  char this_char;
  unsigned int send_len;
  bool done = 0;

//  send_len = lstrlen(messagetext);
  send_len = strlen(messagetext);
  index = messagetext;

#ifdef GEN_HEADER_AT_SEND
  if ( send_message_header( net, smtp,   my_hostname,
                          destid, fromid, rc5id, rfc822Date() ) )
    return -1;
  header_end = messagetext;
#else
  header_end = strstr (messagetext, "\r\n\r\n");
#endif

  while (!done)
  {
    // room for extra char for double dot on end case
    while ((unsigned int) (index - messagetext) < send_len)
    {
      this_char = *index;
#if defined(SHOWMAIL) && !defined(NEEDVIRTUALMETHODS)
printf("%c",this_char);
#endif
//      delay(1); //Some servers can't talk too fast.
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

void MailMessage::smtp_error (Network *net, const char * message)
{
#if !defined(NEEDVIRTUALMETHODS)
  printf("%s\n",message);
#endif
  put_smtp_line("QUIT\r\n", 6,net);
  net->Close();
}

#endif
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

