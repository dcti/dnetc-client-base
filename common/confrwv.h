/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __CONFRWV_H__
#define __CONFRWV_H__ "@(#)$Id: confrwv.h,v 1.10.2.2 2000/09/17 11:46:30 cyp Exp $"

int  ConfigRead(Client *client);
int  ConfigWrite(Client *client);
void ConfigWriteUniversalNews( Client *client );

#endif /* __CONFRWV_H__ */
