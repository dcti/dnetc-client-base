// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: mail.h,v $
// Revision 1.9  1998/08/15 18:11:27  cyruspatel
// Adjusted for mail.cpp changes.
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

//#define MAILTEST

#ifndef __MAIL_H__
#define __MAIL_H__

#include <limits.h>
#define  MAILBUFFSIZE 150000L
#if (UINT_MAX < MAILBUFFSIZE) 
  //was #if (INTSIZES == 422) //#if (CLIENT_OS==OS_WIN16)
  #undef MAILBUFFSIZE
  #define MAILBUFFSIZE 32000
#endif

#ifdef  MAILTEST
#undef  MAILBUFFSIZE
#define MAILBUFFSIZE 1024
#define SHOWMAIL
#endif

class MailMessage
{
public:
  size_t sendthreshold;   //send threshold
  const size_t spoolbuffmaxsize = MAILBUFFSIZE; //so the code uses a variable
    
  char spoolbuff[MAILBUFFSIZE];
  char fromid[256];
  char destid[256];
  char rc5id[256];
  char smtphost[256];
  unsigned int smtpport;

  int append(const char *txt);
  int send(void);

  int clear(void)            { spoolbuff[0]=0; return 0; } 
  size_t countspooled(void)  { return strlen(spoolbuff); }
  int checktosend(int force) { return ((force || (countspooled()>
                               ((sendthreshold/10)*9)))?(send()):(0)); }
  MailMessage(void)          { fromid[0]=destid[0]=rc5id[0]=smtphost[0]=0;
			       #if defined(MAILTEST)
			       printf("** MAILTEST:ON spoolmaxsize=%u **\n",
			         spoolbuffmaxsize);
			       #endif
                               spoolbuff[0]=0;sendthreshold=0;smtpport=0;}
  ~MailMessage(void)         { send(); }
};

#undef MAILBUFFSIZE
#endif //__MAIL_H__
