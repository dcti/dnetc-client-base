// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: mail.h,v $
// Revision 1.15  1999/01/29 04:10:27  cyp
// default nettimeout for mail is -1
//
// Revision 1.14  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.13  1998/08/24 23:49:58  cyp
// added mailmessage.clear() so logstuff can clear the spool if necessary.
//
// Revision 1.12  1998/08/21 00:05:50  cyruspatel
// Added a sendpendingflag so that smtp_deinitialize() (or the MailMessage
// destructor) will attempt a send() before the spool is destroyed/cleared.
//
// Revision 1.11  1998/08/20 19:24:58  cyruspatel
// Restored spooling via static buffer until Autobuffer growth can be
// limited.
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

#include "logstuff.h"

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
#else //if defined(MAILSPOOL_IS_MALLOCBUFFER)
  //
  #define MAILSPOOL_IS_MALLOCBUFFER
#endif  

struct mailmessage
{
  unsigned long sendthreshold; 
  unsigned long maxspoolsize;
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

extern int smtp_initialize_message( struct mailmessage *msg, 
         unsigned long sendthresh, const char *smtphost, unsigned int smtpport,
         const char *fromid, const char *destid, const char *rc5id );
extern int smtp_deinitialize_message( struct mailmessage *msg );
extern int smtp_append_message( struct mailmessage *msg, const char *txt );
extern int smtp_send_message( struct mailmessage *msg );
extern int smtp_clear_message( struct mailmessage *msg );
extern unsigned long smtp_countspooled( struct mailmessage *msg );

class MailMessage
{
public:
  struct mailmessage msg;

  int Initialize( unsigned long _sendthresh, const char *_smtphost, 
      unsigned int _smtpport, const char *_fromid, const char *_destid, 
      const char *_rc5id )    { return smtp_initialize_message( &msg, 
                                _sendthresh, _smtphost, _smtpport,
                                _fromid,  _destid, _rc5id );               }
        
  int Deinitialize(void)      { return smtp_deinitialize_message( &msg );  }
                              //Deinitialize() send()s if necessary

  int append(const char *txt) { return smtp_append_message( &msg, txt );   }
  
  int send(void)              { return smtp_send_message( &msg );          }

  int clear(void)             { return smtp_clear_message( &msg );         }
  
  unsigned long countspooled(void) { return smtp_countspooled( &msg );     }

  int checktosend(int force)  { return ((force || (countspooled()>
                                ((msg.sendthreshold/10)*9)))?(send()):(0));}

  MailMessage(void)           { memset( (void *)(&msg), 0, sizeof(msg) );  }

  ~MailMessage(void)          { Deinitialize();                            }
};

#endif //__MAIL_H__
