// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// encapsulate UltraSparc bitslice code

// $Log: des-slice-ultrasparc.cpp,v $
// Revision 1.1  1998/06/14 14:23:52  remi
// Initial revision
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "problem.h"
#include "convdes.h"

//#define DEBUG

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

static char *id="@(#)$Id: des-slice-ultrasparc.cpp,v 1.1 1998/06/14 14:23:52 remi Exp $";

typedef unsigned long long base_slice_type;

#if defined(LOW_WORD_VALID) || defined(HIGH_WORD_VALID)
#define BITS_PER_SLICE 32
#else
#define BITS_PER_SLICE 64
#endif

extern "C" unsigned long whack16 (
    base_slice_type *plain, base_slice_type *cypher, base_slice_type *key);

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 nbbits )
{
    base_slice_type key[56];
    base_slice_type plain[64];
    base_slice_type cypher[64];
    u32 i;

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
	plain[i] = (pp & mask) ? ~(0ull) : 0;
	cypher[i] = (cc & mask) ? ~(0ull) : 0;
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
	key[i] = (kk & mask) ? ~(0ull) : 0;
	if ((mask <<= 1) == 0) {
	    kk = keyhi;
	    mask = 1;
	}
    }

    // now we must generate 32/64 different keys with
    // bits not used by Meggs core
    // ..ull means unsigned long long constant, gcc-isme
// no need to do so
// already done in whack16()
    //key[40] = 0xAAAAAAAAAAAAAAAAull;
    //key[41] = 0xCCCCCCCCCCCCCCCCull;
    //key[ 0] = 0x0F0F0F0FF0F0F0F0ull;
    //key[ 1] = 0xFF00FF00FF00FF00ull;
    //key[ 2] = 0xFFFF0000FFFF0000ull;
    //key[ 4] = 0xFFFFFFFF00000000ull;
	
#if defined(DEBUG) && defined(BIT_64)
    // note that %016XL is not ansi-c and so not portable
    // it means "long long" and it works under Linux
    //for (i=0; i<64; i++) printf ("bit %02d of plain  = %016LX\n", i, plain[i]);
    //for (i=0; i<64; i++) printf ("bit %02d of cypher = %016LX\n", i, cypher[i]);
    //for (i=0; i<56; i++) printf ("bit %02d of key    = %016LX\n", i, key[i]);
#endif

    // Zero out all the bits that are to be varied
// no need to do so
// already done in whack16()
    //key[ 3]=key[ 5]=key[ 8]=key[10]=key[11]=key[12]=key[15]=key[18]=
    //key[42]=key[43]=key[45]=key[46]=key[49]=key[50]=0;

    // Launch a crack session
    base_slice_type result = whack16( plain, cypher, key);
    //printf ("result = %016LX\n", result);
#ifdef HIGH_WORD_VALID
    result >>= 32;
#elif LOW_WORD_VALID
    result &= 0xFFFFFFFFull;
#endif
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
	    for (int j=0; j<BITS_PER_SLICE; j++)
		printf ((key[i] & (1ull << (BITS_PER_SLICE-1-j))) ? "1":"0");
	    printf ("\n");
	}
#endif

	// which one is the good key ?
	// search the first bit set to 1 in result
	int numkeyfound = -1;
	for (i=0; i<BITS_PER_SLICE; i++)
	    if ((result & (1ull << i)) != 0) numkeyfound = i;
#ifdef DEBUG
	printf ("result  = ");
	for (i=0; i<BITS_PER_SLICE; i++)
	    printf (result & (1ull << (BITS_PER_SLICE-1-i)) ? "1":"0");
	printf ("\n");
#endif

	// convert winning key from slice mode to DES format (with parity)
	keylo = keyhi = 0;
	for (int j=55; j>0; j-=7) {
#ifdef HIGH_WORD_VALID
	    u32 byte = odd_parity [
		(((key[j-0] >> (numkeyfound+32)) & 1) << 7) |
		(((key[j-1] >> (numkeyfound+32)) & 1) << 6) |
		(((key[j-2] >> (numkeyfound+32)) & 1) << 5) |
		(((key[j-3] >> (numkeyfound+32)) & 1) << 4) |
		(((key[j-4] >> (numkeyfound+32)) & 1) << 3) |
		(((key[j-5] >> (numkeyfound+32)) & 1) << 2) |
		(((key[j-6] >> (numkeyfound+32)) & 1) << 1)];
#else
	    u32 byte = odd_parity [
		(((key[j-0] >> numkeyfound) & 1) << 7) |
		(((key[j-1] >> numkeyfound) & 1) << 6) |
		(((key[j-2] >> numkeyfound) & 1) << 5) |
		(((key[j-3] >> numkeyfound) & 1) << 4) |
		(((key[j-4] >> numkeyfound) & 1) << 3) |
		(((key[j-5] >> numkeyfound) & 1) << 2) |
		(((key[j-6] >> numkeyfound) & 1) << 1)];
#endif
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
