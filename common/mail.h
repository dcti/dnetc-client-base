// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#ifndef MAIL_H
#define MAIL_H

#if (CLIENT_OS == OS_AMIGAOS)
extern "C" {
#endif

#include <string.h>

#if (CLIENT_OS == OS_AMIGAOS)
}
#endif

#if (CLIENT_OS == OS_WIN16)
  #define MAILBUFFSIZE 32000
  #define MAXMAILSIZE 31000
#else
  #define MAILBUFFSIZE 150000
  #define MAXMAILSIZE 125000
#endif

class MailMessage
{
public:
  u32 messagelen;
  int port;
  char fromid[255];
  char smtp[255];
  char destid[255];
  char my_hostname[255];
  char rc5id[255];
  s32 quietmode;
protected:
  char messagetext[MAILBUFFSIZE];
  int timeoffset;
public:
  MailMessage(void);
  ~MailMessage(void);
  int inittext(int x);
  void checktosend( u32 forcesend );
  void addtomessage(char *txt );
#ifndef NONETWORK
  int sendmessage(void);
  int prepare_smtp_message(Network * net);
  int send_smtp_edit_data (Network * net);
  int get_smtp_line(Network * net);
  int put_smtp_line(char * line, unsigned int nchars , Network * net);
  void smtp_error (Network * net, char * message);
  int finish_smtp_message(Network * net);
  int transform_and_send_edit_data(Network * net);
#endif
};

#endif

