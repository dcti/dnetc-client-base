// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// encapsulate Meggs' bitslicer
//

#if (!defined(lint) && defined(__showids__))
const char *des_slice_meggs_cpp(void) {
return "@(#)$Id: des-slice-meggs.cpp,v 1.25.2.1 1999/12/07 23:56:27 cyp Exp $"; }
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include "problem.h"
#include "convdes.h"

#if defined(BIT_32) || defined(BIT_64)
  #error "Remove BIT_32/BIT_64 defines from your makefile"
#endif  

#if defined(MMX_BITSLICER) || defined(NTALPHA)
  #define BIT_64
  #if defined(__GNUC__)
    #define BASIC_SLICE_TYPE unsigned long long
    #define NOTZERO ~(0ull)
    #define BIGNUM(n) n##ull
  #elif defined(__WATCOMC__)
    #define BASIC_SLICE_TYPE __int64
    #define NOTZERO ~((__int64)0)
    #define BIGNUM(n) n##ul
  #elif (defined(_MSC_VER) && (_MSC_VER >= 11)) // VC++ >= 5.0
    #define BASIC_SLICE_TYPE __int64
    #define NOTZERO ~((__int64)0)
    #define BIGNUM(n) n##ul
  #else
    #error "What's the 64-bit type on your compiler ?"
  #endif 
#elif (ULONG_MAX == 0xfffffffful)
  #define BIT_32
  #define BASIC_SLICE_TYPE unsigned long
  #define NOTZERO ~(0ul)
  #define BIGNUM(n) n##ul
#elif (ULONG_MAX == 0xfffffffffffffffful)
  #define BIT_64
  #define BASIC_SLICE_TYPE unsigned long
  #define NOTZERO ~(0ul)
  #define BIGNUM(n) n##ul
#else  
  #error "Cannot determine 32/64 bittedness"
#endif

#ifdef BIT_32
#define BITS_PER_SLICE 32
#else //if defined(BIT_64)
#define BITS_PER_SLICE 64
#endif

#if defined(MMX_BITSLICER) /* mmx-bitslice */
extern "C" BASIC_SLICE_TYPE whack16 (BASIC_SLICE_TYPE *plain,
            BASIC_SLICE_TYPE *cypher,
            BASIC_SLICE_TYPE *key, char *coremem);
#elif (CLIENT_OS == OS_BEOS) || \
      ((CLIENT_OS == OS_MACOS) && defined(MRCPP_FOR_DES))
extern "C" BASIC_SLICE_TYPE whack16 (BASIC_SLICE_TYPE *plain,
            BASIC_SLICE_TYPE *cypher,
            BASIC_SLICE_TYPE *key);
#else
extern BASIC_SLICE_TYPE whack16 (BASIC_SLICE_TYPE *plain,
            BASIC_SLICE_TYPE *cypher,
            BASIC_SLICE_TYPE *key);
#endif

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'iterations' if no key found, 'iterations-something' else
//         the adjusted number of iterations is stored back in the param
//         (can't be less than 1<<19 when BIT_32 is defined
//          and can't be less than 1<<20 when BIT_64)

// rc5unitwork.LO in lo:hi 24+32 incrementable format

#if defined(MMX_BITSLICER)
u32 des_unit_func_mmx( RC5UnitWork * rc5unitwork, u32 *iterations, char *coremem)
#elif defined(DEC_UNIX_CPU_SELECT)
u32 des_alpha_osf_ev4( RC5UnitWork * rc5unitwork, u32 *iterations, char *)
#else
u32 des_unit_func_meggs( RC5UnitWork * rc5unitwork, u32 *iterations, char *)
#endif
{
  BASIC_SLICE_TYPE key[56];
  BASIC_SLICE_TYPE plain[64];
  BASIC_SLICE_TYPE cypher[64];
  u32 i, nbbits;

#ifdef BIT_32
  assert( sizeof(BASIC_SLICE_TYPE) == 4);
  nbbits = 19; //minimum and maximum
#else //BIT_64
  assert( sizeof(BASIC_SLICE_TYPE) == 8);
  nbbits = 20; //minimum and maximum
#endif  
#if defined(MMX_BITSLICER) && defined(BITSLICER_WITH_LESS_BITS)
  nbbits = 16; //mmx-bitslice allows this
#endif    

  *iterations = (1ul << nbbits);

  // convert the starting key from incrementable format
  // to DES format
  u32 keyhi = rc5unitwork->L0.hi;
  u32 keylo = rc5unitwork->L0.lo;
  convert_key_from_inc_to_des (&keyhi, &keylo);

  // convert plaintext and cyphertext to slice mode
  u32 pp = rc5unitwork->plain.lo;
  u32 cc = rc5unitwork->cypher.lo;
  u32 mask = 1;
  for (i=0; i<64; i++) {
    plain[i] = (pp & mask) ? NOTZERO : 0;
    cypher[i] = (cc & mask) ? NOTZERO : 0;
    if ((u32)(mask <<= 1) == 0) {
      pp = rc5unitwork->plain.hi;
      cc = rc5unitwork->cypher.hi;
      mask = 1;
    }
  }

  // are we testing complementary keys ?
  bool complement = false;

  redo:
  // convert key to slice mode
  // keybyte[0] & 0x80  -->>  key[55]
  // keybyte[7] & 0x02  -->>  key[0]
  // we are counting bits from right to left
  u32 kk = keylo;
  mask = 1;
  for (i=0; i<56; i++) {
    if ((i % 7) == 0) mask <<= 1;
    key[i] = (kk & mask) ? NOTZERO : 0;
    if ((mask <<= 1) == 0) {
      kk = keyhi;
      mask = 1;
    }
  }


    // now we must generate 32/64 different keys with
    // bits not used by Meggs core
#ifdef BIT_32
  key[40] = 0xAAAAAAAAul;
  key[41] = 0xCCCCCCCCul;
  key[ 0] = 0xF0F0F0F0ul;
  key[ 1] = 0xFF00FF00ul;
  key[ 2] = 0xFFFF0000ul;
#elif defined(BIT_64)
  #if defined(MMX_BITSLICER) && defined(BITSLICER_WITH_LESS_BITS)
  key[40] = BIGNUM(0xAAAAAAAAAAAAAAAA);
  key[41] = BIGNUM(0xCCCCCCCCCCCCCCCC);
  key[12] = BIGNUM(0x0F0F0F0FF0F0F0F0);
  key[15] = BIGNUM(0xFF00FF00FF00FF00);
  key[45] = BIGNUM(0xFFFF0000FFFF0000);
  key[50] = BIGNUM(0xFFFFFFFF00000000);
  #else
  key[40] = BIGNUM(0xAAAAAAAAAAAAAAAA);
  key[41] = BIGNUM(0xCCCCCCCCCCCCCCCC);
  key[ 0] = BIGNUM(0x0F0F0F0FF0F0F0F0);
  key[ 1] = BIGNUM(0xFF00FF00FF00FF00);
  key[ 2] = BIGNUM(0xFFFF0000FFFF0000);
  key[ 4] = BIGNUM(0xFFFFFFFF00000000);
  #endif
#endif
  
#if defined(DEBUG) 
  #if defined(BIT_32)
  for (i=0; i<64; i++) printf ("bit %02d of plain  = %08X\n", i, plain[i]);
  for (i=0; i<64; i++) printf ("bit %02d of cypher = %08X\n", i, cypher[i]);
  for (i=0; i<56; i++) printf ("bit %02d of key    = %08X\n", i, key[i]);
  #elif defined(MMX_BITSLICER) //mmx bitslice
  for (i=0; i<64; i++) printf ("bit %02ld of plain  = %08X%08X\n", i, (unsigned)(plain[i] >> 32), (unsigned)(plain[i] & 0xFFFFFFFF));
  for (i=0; i<64; i++) printf ("bit %02ld of cypher = %08X%08X\n", i, (unsigned)(cypher[i] >> 32), (unsigned)(cypher[i] & 0xFFFFFFFF));
  for (i=0; i<56; i++) printf ("bit %02ld of key    = %08X%08X\n", i, (unsigned)(key[i] >> 32), (unsigned)(key[i] & 0xFFFFFFFF));
  #else
  for (i=0; i<64; i++) printf ("bit %02d of plain  = %016X\n", i, plain[i]);
  for (i=0; i<64; i++) printf ("bit %02d of cypher = %016X\n", i, cypher[i]);
  for (i=0; i<56; i++) printf ("bit %02d of key    = %016X\n", i, key[i]);
  #endif
#endif  

  // Zero out all the bits that are to be varied
#if defined(MMX_BITSLICER) && defined(BITSLICER_WITH_LESS_BITS)
  key[ 3]=key[ 5]=key[ 8]=key[10]=key[11]=
  key[18]=key[42]=key[43]        =key[46]=key[49]        =0;
#else
  key[ 3]=key[ 5]=key[ 8]=key[10]=key[11]=key[12]=key[15]=
  key[18]=key[42]=key[43]=key[45]=key[46]=key[49]=key[50]=0;
#endif    

  // Launch a crack session
//printf("beginning whack\n");  

#if defined(MMX_BITSLICER)
  BASIC_SLICE_TYPE result = whack16( plain, cypher, key, coremem);
#else  
  BASIC_SLICE_TYPE result = whack16( plain, cypher, key);
#endif  

//printf("ended whack\n");  

  // Test also the complementary key
  if (result == 0 && complement == false) {
    keyhi = ~keyhi;
    keylo = ~keylo;
    complement = true;
    goto redo;
  }

  // have we found something ?
  if (result != 0) {

#ifdef DEBUG
    // print all keys in binary format
    for (i=0; i<56; i++) {
      printf ("key[%02ld] = ", i);
      for (int j=0; j<32; j++)
	printf ((key[i] & ((BASIC_SLICE_TYPE)1 << (BITS_PER_SLICE-1-j))) ? "1":"0");
      printf ("\n");
    }
#endif

    // which one is the good key ?
    // search the first bit set to 1 in result
    int numkeyfound = -1;
    for (i=0; i<BITS_PER_SLICE; i++)
      if ((result & ((BASIC_SLICE_TYPE)1 << i)) != 0) numkeyfound = i;
#ifdef DEBUG
    printf ("result = ");
    for (i=0; i<BITS_PER_SLICE; i++)
      printf (result & ((BASIC_SLICE_TYPE)1 << (BITS_PER_SLICE-1-i)) ? "1":"0");
    printf ("\n");
#endif

    // convert winning key from slice mode to DES format (with parity)
    keylo = keyhi = 0;
    for (int j=55; j>0; j-=7) {
      u32 byte = odd_parity [
	(((key[j-0] >> numkeyfound) & 1) << 7) |
	(((key[j-1] >> numkeyfound) & 1) << 6) |
	(((key[j-2] >> numkeyfound) & 1) << 5) |
	(((key[j-3] >> numkeyfound) & 1) << 4) |
	(((key[j-4] >> numkeyfound) & 1) << 3) |
	(((key[j-5] >> numkeyfound) & 1) << 2) |
	(((key[j-6] >> numkeyfound) & 1) << 1)];
      int numbyte = (j+1) / 7 - 1;
      if (numbyte >= 4)
	keyhi |= byte << ((numbyte-4)*8);
      else
	keylo |= byte << (numbyte*8);
    }

    // have we found the complementary key ?
    if (complement) {
      keyhi = ~keyhi;
      keylo = ~keylo;
    }
#ifdef DEBUG
    printf (complement ?
	    "  key %02d = %08X:%08X (C) " : "  key %02d = %08X:%08X (n) ",
	    numkeyfound, keyhi, keylo);
#endif
    // convert key from 64 bits DES ordering with parity
    // to incrementable format
    convert_key_from_des_to_inc (&keyhi, &keylo);
  
    u32 nbkeys = keylo - rc5unitwork->L0.lo;
    rc5unitwork->L0.lo = keylo;
    rc5unitwork->L0.hi = keyhi;

    return nbkeys;

  }
  else
  {
    rc5unitwork->L0.lo += 1 << nbbits; // Increment lower 32 bits
    if (rc5unitwork->L0.lo < (u32)(1 << nbbits) )
      rc5unitwork->L0.hi++; // Carry to high 32 bits if needed
    return 1 << nbbits;
  }
}

