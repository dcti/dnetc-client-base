/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __BENCH_H__
#define __BENCH_H__ "@(#)$Id: bench.h,v 1.21 2012/05/13 09:32:54 stream Exp $"

#include "client.h"

/* returns "rate", or -1 if core error/^C, or 0 if not supported */
long TBenchmark( Client *client, unsigned int contestid, unsigned int numsecs, int flags, u32 *p_ratehi, u32 *p_ratelo );
#define TBENCHMARK_QUIET       0x01
#define TBENCHMARK_IGNBRK      0x02
// do not use 0x80, it's internal to TBenchmark
//#define TBENCHMARK_CALIBRATION 0x80
void BenchGetBestRate(Client *client, unsigned int contestid, u32 *p_ratehi, u32 *p_ratelo);
void BenchResetStaticVars(void);

#endif /* __BENCH_H__ */
