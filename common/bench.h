// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: bench.h,v $
// Revision 1.5.2.1  1999/01/30 15:44:04  remi
// Forked a branch, because we don't need AutoSetThresold() here, and we
// don't have any confrwv.* in the public code.
//
// Revision 1.5  1999/01/17 23:18:14  silby
// AutoSetThreshold added.
//
// Revision 1.4  1999/01/15 20:21:59  michmarc
// Fix the fact that Benchmark() in bench.cpp changed its prototype
//
// Revision 1.3  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
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
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype, int *numblocks );

#endif /* __BENCHMARK_H__ */
