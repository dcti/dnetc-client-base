/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __SELFTEST_H__
#define __SELFTEST_H__ "@(#)$Id: selftest.h,v 1.4.2.1 1999/04/13 19:45:31 jlawson Exp $"

/* returns number of tests if all passed or negative number if a test failed */
int SelfTest( unsigned int contest, int cputype );

#endif /* __SELFTEST_H__ */
