/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BENCH_H__
#define __BENCH_H__ "@(#)$Id: bench.h,v 1.7.2.1 1999/09/19 15:58:49 cyp Exp $"

/* returns keys/sec or 0 if break */
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype, int *numblocks );

/* returns "rate", or -1 if core error/^C, or 0 if not supported */
long TBenchmark( unsigned int contestid, unsigned int numsecs, 
                 int cputype, int flags );
#define TBENCHMARK_QUIET  0x01
#define TBENCHMARK_IGNBRK 0x02

#endif /* __BENCH_H__ */
