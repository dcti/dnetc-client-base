/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __CONFRWV_H__
#define __CONFRWV_H__ "@(#)$Id: confrwv.h,v 1.17 2003/09/12 22:29:25 mweiser Exp $"

int  ConfigRead(Client *client);
int  ConfigWrite(Client *client);
void ConfigWriteUniversalNews( Client *client );

#endif /* __CONFRWV_H__ */
