// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// encapsulate the bitslice SolNET code

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (CLIENT_OS == OS_AMIGAOS)
#include "common/problem.h"
#include "common/convdes.h"
#else
#include "../common/problem.h"
#include "../common/convdes.h"
#endif
#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif

#ifdef BIT_32
#define BITS_PER_SLICE 32
#elif BIT_64
#define BITS_PER_SLICE 64
#else
// make sure the size of a "long" was specified
#error "You must define BIT_32 or BIT_64"
#endif

#if (CLIENT_OS == OS_BEOS)
extern "C" unsigned long Bwhack16 (unsigned long *plain,
			      unsigned long *cypher,
			      unsigned long *key);
#else
extern unsigned long Bwhack16 (unsigned long *plain,
			      unsigned long *cypher,
			      unsigned long *key);
#endif

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 Bdes_unit_func( RC5UnitWork * rc5unitwork, u32 nbbits )
{
    unsigned long key[56];
    unsigned long plain[64];
    unsigned long cypher[64];
    u32 i;

    // check if we have the right BIT_xx define
    // this should be phased out by an optimizing compiler
    // if the right BIT_xx is defined
#ifdef BIT_32
    if (sizeof(unsigned long) != 4) {
	printf ("Bad BIT_32 define !\n");
	exit (-1);
    }
#elif BIT_64
    if (sizeof(unsigned long) != 8) {
	printf ("Bad BIT_64 define !\n");
	exit (-1);
    }
#endif

    // check nbbits
    if (nbbits != MIN_DES_BITS) {
	printf ("Bad nbbits ! (%d)\n", nbbits);
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
	plain[i] = (pp & mask) ? ~(0ul) : 0;
	cypher[i] = (cc & mask) ? ~(0ul) : 0;
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
	key[i] = (kk & mask) ? ~(0ul) : 0;
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
#elif BIT_64
    key[40] = 0xAAAAAAAAAAAAAAAAul;
    key[41] = 0xCCCCCCCCCCCCCCCCul;
    key[ 0] = 0x0F0F0F0FF0F0F0F0ul;
    key[ 1] = 0xFF00FF00FF00FF00ul;
    key[ 2] = 0xFFFF0000FFFF0000ul;
    key[ 4] = 0xFFFFFFFF00000000ul;
#endif
	
#if defined(DEBUG) && defined(BIT_32)
    for (i=0; i<64; i++) printf ("bit %02d of plain  = %08X\n", plain[i]);
    for (i=0; i<64; i++) printf ("bit %02d of cypher = %08X\n", cypher[i]);
    for (i=0; i<56; i++) printf ("bit %02d of key    = %08X\n", key[i]);
#elif defined(DEBUG) && defined(BIT_64)
    for (i=0; i<64; i++) printf ("bit %02d of plain  = %016X\n", plain[i]);
    for (i=0; i<64; i++) printf ("bit %02d of cypher = %016X\n", cypher[i]);
    for (i=0; i<56; i++) printf ("bit %02d of key    = %016X\n", key[i]);
#endif

    // Zero out all the bits that are to be varied
    key[ 3]=key[ 5]=key[ 8]=key[10]=key[11]=key[12]=key[15]=key[18]=
    key[42]=key[43]=key[45]=key[46]=key[49]=key[50]=0;

    // Launch a crack session
    unsigned long result = Bwhack16( plain, cypher, key);
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
	    printf ("key[%02d] = ", i);
	    for (int j=0; j<32; j++)
		printf ((key[i] & (1ul << (BITS_PER_SLICE-1-j))) ? "1":"0");
	    printf ("\n");
	}
#endif

	// which one is the good key ?
	// search the first bit set to 1 in result
	int numkeyfound = -1;
	for (i=0; i<BITS_PER_SLICE; i++)
	    if ((result & (1ul << i)) != 0) numkeyfound = i;
#ifdef DEBUG
	printf ("result = ");
	for (i=0; i<BITS_PER_SLICE; i++)
	    printf (result & (1ul << (BITS_PER_SLICE-1-i)) ? "1":"0");
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

