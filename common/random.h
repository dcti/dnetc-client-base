// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Id: random.h,v 1.4 1999/04/05 14:34:58 jlawson Exp $

#ifndef __RANDOM_H__
#define __RANDOM_H__

u32  Random( const u32 * u32data, unsigned int u32count );
  // count = # of u32s of data...
  // calling it with ( NULL, 0 ) is OK...
  // calls InitRandom() and/or InitRandom2() if they haven't been called yet 
  // Returns: a random u32, mangled slightly with data...

void InitRandom(void);
  // Initialize random number generator (added 12.15.97)

void InitRandom2(const char *p);
  // Initialize random number generator, using string p to 
  // influence seed value.

#endif /* __RANDOM_H__ */
