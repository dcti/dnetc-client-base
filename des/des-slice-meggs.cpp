// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-slice-meggs.cpp,v $
// Revision 1.16  1998/07/14 10:43:39  remi
// Added support for a minimum timeslice value of 16 instead of 20 when
// using BIT_64, which is needed by MMX_BITSLICER. Will help some platforms
// like Netware or Win16. I added support in deseval-meggs3.cpp, but it's just
// for completness, Alphas don't need this patch.
//
// Important note : this patch **WON'T** work with deseval-meggs2.cpp, but
// according to the configure script it isn't used anymore. If you compile
// des-slice-meggs.cpp and deseval-meggs2.cpp with BIT_64 and
// BITSLICER_WITH_LESS_BITS, the DES self-test will fail.
//
// Revision 1.15  1998/07/13 03:48:48  cyruspatel
// Converted an 'if (sizeof(BASIC_SLICE_TYPE)!=8)' check to an assert() to
// squelch a compiler warning.
//
// Revision 1.14  1998/07/12 23:52:14  foxyloxy
// Fixed typo (changed NOTSZERO to NOTZERO) to allow compile to work on IRIX
// (and probably other platforms).
//
// Revision 1.13  1998/07/10 20:08:25  cyruspatel
// Added support for Watcom compilers (__int64) to mmx bitslice stuff
//
// Revision 1.12  1998/07/08 23:42:05  remi
// Added support for CliIdentifyModules().
//
// Revision 1.11  1998/07/08 16:26:22  remi
// Added support for MS-VC++ 5.0
//
// Revision 1.10  1998/07/08 10:06:20  remi
// Another RCS-id tweaking.
//
// Revision 1.9  1998/07/08 10:02:46  remi
// Declare whack16() with "C" linkage. Will help MS platforms.
//
// Revision 1.8  1998/07/08 10:00:31  remi
// Added support for the MMX bitslicer.
//
// Revision 1.7  1998/06/14 08:27:02  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.6  1998/06/14 08:13:17  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

// encapsulate Meggs' bitslicer

#if (!defined(lint) && defined(__showids__))
const char *des_slice_meggs_cpp(void) {
return "@(#)$Id: des-slice-meggs.cpp,v 1.16 1998/07/14 10:43:39 remi Exp $"; }
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "problem.h"
#include "convdes.h"

//#define DEBUG

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

#if defined(MMX_BITSLICER)
  #if defined(__GNUC__)
    #define BASIC_SLICE_TYPE unsigned long long
    #define NOTZERO ~(0ull)
  #elif defined(__WATCOMC__) || (_MSC_VER >= 11) // VC++ 5.0
    #define BASIC_SLICE_TYPE __int64
    #define NOTZERO ~((__int64)0)
  #elif
    #error "What's the 64-bit type on your compiler ?"
  #endif
#else
  #define BASIC_SLICE_TYPE unsigned long
  #define NOTZERO ~(0ul)
#endif

#ifdef BIT_32
#define BITS_PER_SLICE 32
#elif BIT_64
#define BITS_PER_SLICE 64
#else
// make sure the size of a "long" was specified
#error "You must define BIT_32 or BIT_64"
#endif

#if (CLIENT_OS == OS_BEOS) || defined(MMX_BITSLICER)
extern "C" BASIC_SLICE_TYPE whack16 (BASIC_SLICE_TYPE *plain,
            BASIC_SLICE_TYPE *cypher,
            BASIC_SLICE_TYPE *key);
#else
extern BASIC_SLICE_TYPE whack16 (BASIC_SLICE_TYPE *plain,
            BASIC_SLICE_TYPE *cypher,
            BASIC_SLICE_TYPE *key);
#endif

#if defined(MMX_BITSLICER)
  #define des_unit_func des_unit_func_mmx
#elif (CLIENT_CPU == CPU_X86)
  #define des_unit_func des_unit_func_slice
#endif

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 nbbits )
{
  BASIC_SLICE_TYPE key[56];
  BASIC_SLICE_TYPE plain[64];
  BASIC_SLICE_TYPE cypher[64];
  u32 i;

  // check if we have the right BIT_xx define
  // this should be phased out by an optimizing compiler
  // if the right BIT_xx is defined
#ifdef BIT_32
  assert( sizeof(BASIC_SLICE_TYPE) == 4);
  //  if (sizeof(BASIC_SLICE_TYPE) != 4) {
  //printf ("Bad BIT_32 define !\n");
  //exit (-1);
  // }
#elif BIT_64
  assert( sizeof(BASIC_SLICE_TYPE) == 8);
  // if (sizeof(BASIC_SLICE_TYPE) != 8) {
  // printf ("Bad BIT_64 define !\n");
  // exit (-1);
  // }
#endif

  // check nbbits
  if (nbbits != MIN_DES_BITS) {
    printf ("Bad nbbits ! (%d)\n", (int)nbbits);
    exit (-1);
  }

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

#if defined(MMX_BITSLICER)
  #if defined(__GNUC__)
    #define BIGNUM(n) n##ull
  #elif (defined(__WATCOMC__) || (_MSC_VER>=11))
    #define BIGNUM(n) n##ul
  #else
    #error "Write this macro for your compiler"
  #endif
#else
  #define BIGNUM(n) n##ul
#endif // MMX_BITSLICER

    // now we must generate 32/64 different keys with
    // bits not used by Meggs core
#ifdef BIT_32
  key[40] = 0xAAAAAAAAul;
  key[41] = 0xCCCCCCCCul;
  key[ 0] = 0xF0F0F0F0ul;
  key[ 1] = 0xFF00FF00ul;
  key[ 2] = 0xFFFF0000ul;
#elif defined(BITSLICER_WITH_LESS_BITS) && defined(BIT_64)
  key[40] = BIGNUM(0xAAAAAAAAAAAAAAAA);
  key[41] = BIGNUM(0xCCCCCCCCCCCCCCCC);
  key[12] = BIGNUM(0x0F0F0F0FF0F0F0F0);
  key[15] = BIGNUM(0xFF00FF00FF00FF00);
  key[45] = BIGNUM(0xFFFF0000FFFF0000);
  key[50] = BIGNUM(0xFFFFFFFF00000000);
#elif defined(BIT_64)
  key[40] = BIGNUM(0xAAAAAAAAAAAAAAAA);
  key[41] = BIGNUM(0xCCCCCCCCCCCCCCCC);
  key[ 0] = BIGNUM(0x0F0F0F0FF0F0F0F0);
  key[ 1] = BIGNUM(0xFF00FF00FF00FF00);
  key[ 2] = BIGNUM(0xFFFF0000FFFF0000);
  key[ 4] = BIGNUM(0xFFFFFFFF00000000);
#endif
  
#if defined(DEBUG) && defined(BIT_32)
  for (i=0; i<64; i++) printf ("bit %02d of plain  = %08X\n", i, plain[i]);
  for (i=0; i<64; i++) printf ("bit %02d of cypher = %08X\n", i, cypher[i]);
  for (i=0; i<56; i++) printf ("bit %02d of key    = %08X\n", i, key[i]);
#elif defined(DEBUG) && defined(MMX_BITSLICER)
  for (i=0; i<64; i++) printf ("bit %02ld of plain  = %08X%08X\n", i, (unsigned)(plain[i] >> 32), (unsigned)(plain[i] & 0xFFFFFFFF));
  for (i=0; i<64; i++) printf ("bit %02ld of cypher = %08X%08X\n", i, (unsigned)(cypher[i] >> 32), (unsigned)(cypher[i] & 0xFFFFFFFF));
  for (i=0; i<56; i++) printf ("bit %02ld of key    = %08X%08X\n", i, (unsigned)(key[i] >> 32), (unsigned)(key[i] & 0xFFFFFFFF));
#elif defined(DEBUG) && defined(BIT_64)
  for (i=0; i<64; i++) printf ("bit %02d of plain  = %016X\n", i, plain[i]);
  for (i=0; i<64; i++) printf ("bit %02d of cypher = %016X\n", i, cypher[i]);
  for (i=0; i<56; i++) printf ("bit %02d of key    = %016X\n", i, key[i]);
#endif

  // Zero out all the bits that are to be varied
#if defined(BITSLICER_WITH_LESS_BITS)
  key[ 3]=key[ 5]=key[ 8]=key[10]=key[11]=
  key[18]=key[42]=key[43]        =key[46]=key[49]        =0;
#else
  key[ 3]=key[ 5]=key[ 8]=key[10]=key[11]=key[12]=key[15]=
  key[18]=key[42]=key[43]=key[45]=key[46]=key[49]=key[50]=0;
#endif

  // Launch a crack session
  BASIC_SLICE_TYPE result = whack16( plain, cypher, key);
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

  } else {
    rc5unitwork->L0.lo += 1 << nbbits;
    return 1 << nbbits;
  }
}

