/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __CONFRWV_H__
#define __CONFRWV_H__ "@(#)$Id: confrwv.h,v 1.20 2010/02/15 19:44:26 stream Exp $"

int  ConfigRead(Client *client);
int  ConfigWrite(Client *client);
void ConfigWriteUniversalNews( Client *client );
int  ConfigWriteRandomSubspace( Client *client, int contestid, u32 subspace );

#endif /* __CONFRWV_H__ */
