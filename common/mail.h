// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: mail.h,v $
// Revision 1.10  1998/08/15 21:31:05  jlawson
// new mail code using autobuffer
//
// Revision 1.8  1998/08/10 20:29:36  cyruspatel
// Call to gethostname() is now a call to Network::GetHostName(). Updated
// send routine to reflect new NetworkInitialize()/NetworkDeinitialize()
// requirements. Removed all references to NO!NETWORK.
//
// Revision 1.7  1998/07/07 21:55:44  cyruspatel
// client.h has been split into client.h baseincs.h
//
// Revision 1.6  1998/06/14 08:12:59  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef __MAIL_H__
#define __MAIL_H__

#include "cputypes.h"
#include "autobuff.h"

class MailMessage
{
public:
  u32 sendthreshold;   //send threshold
  
  AutoBuffer spoolbuff;
  char fromid[256];
  char destid[256];
  char rc5id[256];
  char smtphost[256];
  s16 smtpport;

  int append(const char *txt);
  int send(void);

  u32 clear(void)
    { spoolbuff.Clear(); return 0; } 
  u32 countspooled(void)
    { return spoolbuff.GetLength(); }
  int checktosend(int force)
    {
      return ((force || (countspooled() > ((sendthreshold/10)*9)) ) ?
          (send()):(0) );
    }
  MailMessage(void)
    {
      fromid[0]=destid[0]=rc5id[0]=smtphost[0]=0;
      sendthreshold=0;
      smtpport=0;
    }
  ~MailMessage(void)
    {
      send();
    }
};

#endif //__MAIL_H__

