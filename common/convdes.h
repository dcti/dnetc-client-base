/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ---------------------------------------------------------------
 * Routines to convert DES key # from/to incremental format
 * by Remi Guyomarche
 * ---------------------------------------------------------------
*/ 
#ifndef CONVDES_H
#define CONVDES_H "@(#)$Id: convdes.h,v 1.8 1999/04/06 10:20:48 cyp Exp $"

// odd_parity[n] = (n & 0xFE) | b
// b set so that odd_parity[n] has an odd number of bits
extern const u8 odd_parity[256];

// convert to/from two different key formats
extern void convert_key_from_des_to_inc (u32 *deshi, u32 *deslo);
extern void convert_key_from_inc_to_des (u32 *deshi, u32 *deslo);

#endif /* CONVDES_H */

