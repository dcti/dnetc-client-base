/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BENCH_H__
#define __BENCH_H__ "@(#)$Id: bench.h,v 1.11 2000/06/02 06:24:52 jlawson Exp $"

/* returns "rate", or -1 if core error/^C, or 0 if not supported */
long TBenchmark( unsigned int contestid, unsigned int numsecs, int flags );
#define TBENCHMARK_QUIET       0x01
#define TBENCHMARK_IGNBRK      0x02
// do not use 0x80, it's internal to TBenchmark
//#define TBENCHMARK_CALIBRATION 0x80

#endif /* __BENCH_H__ */
