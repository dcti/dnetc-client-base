/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELCORE_H__
#define __SELCORE_H__ "@(#)$Id: selcore.h,v 1.2.2.1 1999/04/13 19:45:30 jlawson Exp $"

/* returns name for core number (0...) or "" if no such core */
const char *GetCoreNameFromCoreType( unsigned int coretype ); 

#endif /* __SELCORE_H__ */
