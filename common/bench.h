// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

/* returns keys/sec or 0 if break */
u32 Benchmark( unsigned int contest, u32 numkeys, int cputype, int *numblocks );

/* determines optimal buffer sizes based on benchmarked rate */
void AutoSetThreshold( Client *clientp, unsigned int contest,
                       unsigned int inbuffer, unsigned int outbuffer );


#endif /* __BENCHMARK_H__ */
