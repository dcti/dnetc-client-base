/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __CONFRWV_H__
#define __CONFRWV_H__ "@(#)$Id: confrwv.h,v 1.14 2000/06/02 06:24:55 jlawson Exp $"

int ReadConfig(Client *client);
int WriteConfig(Client *client, int writefull /* defaults to 0*/);
void RefreshRandomPrefix( Client *client );


#endif /* __CONFRWV_H__ */
