/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __CONFRWV_H__
#define __CONFRWV_H__ "@(#)$Id: confrwv.h,v 1.9.2.1 1999/04/13 19:45:20 jlawson Exp $"

int ReadConfig(Client *client);
int WriteConfig(Client *client, int writefull /* defaults to 0*/);
void RefreshRandomPrefix( Client *client );

#endif /* __CONFRWV_H__ */
