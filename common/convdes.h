// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Log: convdes.h,v $
// Revision 1.5  1998/06/14 08:12:49  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef CONVDES_H
#define CONVDES_H

#include "cputypes.h"

// odd_parity[n] = (n & 0xFE) | b
// b set so that odd_parity[n] has an odd number of bits
extern const u8 odd_parity[256];

// convert to/from two different key formats
extern void convert_key_from_des_to_inc (u32 *deshi, u32 *deslo);
extern void convert_key_from_inc_to_des (u32 *deshi, u32 *deslo);

#endif

