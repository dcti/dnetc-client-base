/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__ "@(#)$Id: bench.h,v 1.6 1999/04/06 10:20:47 cyp Exp $"

/* returns keys/sec or 0 if break */
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype, int *numblocks );

#endif /* __BENCHMARK_H__ */
