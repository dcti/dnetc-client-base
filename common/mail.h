/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __MAIL_H__
#define __MAIL_H__ "@(#)$Id: mail.h,v 1.17 2000/07/11 04:01:19 mfeiri Exp $"

extern void * smtp_construct_message( unsigned long sendthresh, 
                                   const char *smtphost, unsigned int smtpport,
                                   const char *fromid, const char *destid, 
                                   const char *rc5id );
extern int smtp_destruct_message( void *msghandle );
extern int smtp_deinitialize_message( void *msghandle );
extern int smtp_append_message( void *msghandle, const char *txt );
extern int smtp_send_message( void *msghandle );   /* if anything in spool */
extern int smtp_send_if_needed( void *msghandle ); /* if >= 90% of thresh */
extern int smtp_clear_message( void *msghandle );
extern unsigned long smtp_countspooled( void *msghandle );

#endif /* __MAIL_H__ */
