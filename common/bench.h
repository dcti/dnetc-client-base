// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: bench.h,v $
// Revision 1.2  1998/10/11 00:46:30  cyp
// Benchmark() is now standalone.
//
// Revision 1.1  1998/09/28 01:38:05  cyp
// Spun off from client.cpp  Note: the problem object is local so it does not
// need to be assigned from the problem table. Another positive side effect
// is that benchmarks can be run without shutting down the client.
//
//

#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

/* returns keys/sec or 0 if break */
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype );

#endif /* __BENCHMARK_H__ */
