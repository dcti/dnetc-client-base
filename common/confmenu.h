/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __CONFMENU_H__
#define __CONFMENU_H__ "@(#)$Id: confmenu.h,v 1.1.2.2 2000/09/17 11:46:29 cyp Exp $"

/* returns <0=error, 0=exit+nosave, >0=exit+save */
int Configure( Client *sample_client, int nottycheck ); 

#endif /* __CONFMENU_H__ */
