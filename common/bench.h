/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __BENCH_H__
#define __BENCH_H__ "@(#)$Id: bench.h,v 1.20 2012/01/13 01:05:21 snikkel Exp $"

#include "client.h"

/* returns "rate", or 0 if not supported or core error */
#ifdef HAVE_I64
ui64 TBenchmark( Client *client, unsigned int contestid, unsigned int numsecs, int flags );
#else
long TBenchmark( Client *client, unsigned int contestid, unsigned int numsecs, int flags );
#endif
#define TBENCHMARK_QUIET       0x01
#define TBENCHMARK_IGNBRK      0x02
// do not use 0x80, it's internal to TBenchmark
//#define TBENCHMARK_CALIBRATION 0x80
#ifdef HAVE_I64
ui64 BenchGetBestRate(Client *client, unsigned int contestid);
#else
unsigned long BenchGetBestRate(Client *client, unsigned int contestid);
#endif
void BenchResetStaticVars(void);

#endif /* __BENCH_H__ */
