/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __CONFRWV_H__
#define __CONFRWV_H__ "@(#)$Id: confrwv.h,v 1.16 2002/09/02 01:15:45 andreasb Exp $"

int  ConfigRead(Client *client);
int  ConfigWrite(Client *client);
void ConfigWriteUniversalNews( Client *client );

#endif /* __CONFRWV_H__ */
