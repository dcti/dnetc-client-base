/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BENCH_H__
#define __BENCH_H__ "@(#)$Id: bench.h,v 1.5.4.2 1999/04/13 19:45:11 jlawson Exp $"

/* returns keys/sec or 0 if break */
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype, int *numblocks );

#endif /* __BENCH_H__ */
