/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#ifndef __BENCH_H__
#define __BENCH_H__ "@(#)$Id: bench.h,v 1.16 2008/12/19 11:10:58 andreasb Exp $"

/* returns "rate", or -1 if core error/^C, or 0 if not supported */
long TBenchmark( unsigned int contestid, unsigned int numsecs, int flags );
#define TBENCHMARK_QUIET       0x01
#define TBENCHMARK_IGNBRK      0x02
// do not use 0x80, it's internal to TBenchmark
//#define TBENCHMARK_CALIBRATION 0x80
unsigned long BenchGetBestRate(unsigned int contestid);
void BenchResetStaticVars(void);

#endif /* __BENCH_H__ */
