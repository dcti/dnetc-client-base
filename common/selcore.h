/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.3.2.1 1999/09/19 16:04:38 cyp Exp $"

/* returns name for cputype (from .ini), 0... or "" if no such cputype */
const char *selcoreUserGetCPUNameFromCPUType( int user_cputype );
#define GetCoreNameFromCoreType(x) selcoreUserGetCPUNameFromCPUType(x)

/* this is called from Client::Main() (or COMPAT: Client::SelectCore()) */
int selcoreInitialize( int user_cputype );

/* this is called from Problem::LoadState() */
int selcoreGetSelectedCoreForContest( unsigned int contestid );

#endif /* __SELCORE_H__ */
