/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __SELFTEST_H__
#define __SELFTEST_H__ "@(#)$Id: selftest.h,v 1.15 2012/01/13 01:05:22 snikkel Exp $"

#include "client.h"

/* returns number of tests if all passed or negative number if a test failed */
long SelfTest( Client *client, unsigned int contest );
long StressTest( Client *client, unsigned int contest );

#if defined(HAVE_RC5_72_CORES)
long StressRC5_72( Client *client );  /* RC5-72/stress.cpp */
#endif

#endif /* __SELFTEST_H__ */
