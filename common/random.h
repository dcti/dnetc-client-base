/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * $Log: random.h,v $
 * Revision 1.1  1999/03/20 07:04:46  cyp
 * Split random/randomizing functions to stand alone.
 *
*/ 

#ifndef __RANDOM_H__
#define __RANDOM_H__

u32  Random( u32 * data, u32 length );
  // length = # of u32s of data...
  // calling it with ( NULL, 0 ) is OK...
  // Returns: a random u32, mangled slightly with data...

void InitRandom();
  // Initialize random number generator (added 12.15.97)

void InitRandom2(char *p);
  // Initialize random number generator, using string p to 
  // influence seed value.

#endif
