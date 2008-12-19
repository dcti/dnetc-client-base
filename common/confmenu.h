/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __CONFMENU_H__
#define __CONFMENU_H__ "@(#)$Id: confmenu.h,v 1.9 2008/12/19 11:10:58 andreasb Exp $"

/* returns <0=error, 0=exit+nosave, >0=exit+save */
int Configure( Client *sample_client, int nottycheck );

#endif /* __CONFMENU_H__ */
