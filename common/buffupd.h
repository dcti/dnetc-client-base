/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BUFFUPD_H__
#define __BUFFUPD_H__ "@(#)$Id: buffupd.h,v 1.11 2000/06/02 06:24:53 jlawson Exp $"

#define BUFFERUPDATE_FETCH 0x01
#define BUFFERUPDATE_FLUSH 0x02

#include "client.h" /* forward reference */
int BufferUpdate( Client *client, int updatereq_flags, int interactive );

    // pass flags ORd with BUFFERUPDATE_FETCH/*_FLUSH. 
    // if interactive, prints "Input buffer full. No fetch required" etc.
    // returns updated flags or < 0 if failed. (offlinemode!=0/NetOpen() etc)

#endif /* __BUFFUPD_H__ */
