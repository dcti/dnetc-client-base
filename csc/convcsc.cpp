/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/ 
const char *convcsc_cpp(void) {
return "@(#)$Id: convcsc.cpp,v 1.1.2.1 1999/10/07 18:41:13 cyp Exp $"; }

/* CSC conversion routines */

#include <stdio.h>
#include <string.h>
#include "problem.h"
#include "convcsc.h"

/*
 * cs-cipher is a 64-bit block cipher, with variable key length (0..128)
 * Internally, the key is viewed as two 64-bit numbers, independantly from
 * the real key-length. For example, a 56-bit CSC key might be :
 *   0x21, 0x53, 0xad, 0xf9, 0x46, 0x7c, 0xb9, 0x00,
 *   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
 *
 * Since we're on a 56-bit contest, we will only convert the first 64-bit number
 *
 * Here's how I'm counting bits in all CSC stuff :
 * 0x21 == bits 63..56, 0x21 & 0x80 == bit 63
 * 0x53 == bits 55..48, 0x53 & 0x01 == bit 48
 * etc...
 *
 * NOTE: Those routines are probably slower than their DES conterparts
 * but I think they will be far easier to maintain if we want to change
 * the bit order.
 */

const int csc_bit_order[64] = {
  // bits internal to the bitslicer and/or the driver
  22,30,38,46,54,62, 8,9,10,11,12,13,
  // other bits
                    14,15,
  16,17,18,19,20,21,   23,
  24,25,26,27,28,29,   31,
  32,33,34,35,36,37,   39,
  40,41,42,43,44,45,   47,
  48,49,50,51,52,53,   55,
  56,57,58,59,60,61,   63,
  // unused bits (in case of a 56-bit key)
   0, 1, 2, 3, 4, 5, 6, 7,
};

// ------------------------------------------------------------------
// Convert a key from CSC format to incrementable format
//
void convert_key_from_csc_to_inc (u32 *cschi, u32 *csclo)
{
  u32 tmphi, tmplo;
  
  tmphi = tmplo = 0;
  for( int i=0; i<64; i++ ) {
    int src = csc_bit_order[i];
    u32 bitval;
    if( src <= 31 )
      bitval = (*csclo >> src) & 1;
    else
      bitval = (*cschi >> (src-32)) & 1;
    if( i <= 31 )
      tmplo |= bitval << i;
    else
      tmphi |= bitval << (i-32);
  }
  *cschi = tmphi;
  *csclo = tmplo;
}

// ------------------------------------------------------------------
// Convert a key from incrementable format to CSC format
//
void convert_key_from_inc_to_csc (u32 *cschi, u32 *csclo)
{
  u32 tmphi, tmplo;
  
  tmphi = tmplo = 0;
  for( int i=0; i<64; i++ ) {
    u32 bitval;
    int src;
    for( src=0; src<64 && csc_bit_order[src] != i; src++ );
    if( src <= 31 )
      bitval = (*csclo >> src) & 1;
    else
      bitval = (*cschi >> (src-32)) & 1;
    if( i <= 31 )
      tmplo |= bitval << i;
    else
      tmphi |= bitval << (i-32);
  }
  *cschi = tmphi;
  *csclo = tmplo;
}
