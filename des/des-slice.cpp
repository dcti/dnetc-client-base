// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-slice.cpp,v $
// Revision 1.5  1998/07/01 03:12:46  blast
// AmigaOS changes...
//
// Revision 1.4  1998/06/20 10:04:18  cyruspatel
// Modified so x86 make with /DKWAN will work: Renamed des_unit_func() in
// des_slice to des_unit_func_slice() to resolve conflict with (*des_unit_func)().
// Added prototype in problem.h, cliconfig x86/SelectCore() is /DKWAN aware.
//
// Revision 1.3  1998/06/14 08:27:03  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:18  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

// encapsulate the bitslice SolNET code

static char *id="@(#)$Id: des-slice.cpp,v 1.5 1998/07/01 03:12:46 blast Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "problem.h"
#include "convdes.h"

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

extern unsigned long deseval (const unsigned long *plain,
			      const unsigned long *cypher,
			      const unsigned long *key);

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : timeslice will be rounded to the upper power of two
//        and can't be less than 256

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 des_unit_func_slice( RC5UnitWork * rc5unitwork, u32 nbbits )
{
    unsigned long key[56];
    unsigned long plain[64];
    unsigned long cypher[64];
    u32 i;

      // check if we have the right BIT_xx define
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

      // convert the starting key from incrementable format
      // to DES format
    u32 keyhi = rc5unitwork->L0.hi;
    u32 keylo = rc5unitwork->L0.lo;
    convert_key_from_inc_to_des (&keyhi, &keylo);

      // convert key to slice mode
      // keybyte[0] & 0x80  -->>  key[55]
      // keybyte[7] & 0x02  -->>  key[0]
      // we are counting bits from right to left
    u32 kk = keylo;
    u32 mask = 0x00000001;
    for (i=0; i<56; i++) {
	if ((i % 7) == 0) mask <<= 1;
	key[i] = (kk & mask) ? ~(0ul) : 0;
	if ((mask <<= 1) == 0) {
	    kk = keyhi;
	    mask = 0x00000001;
	}
    }

      // convert plaintext and cyphertext to slice mode
    u32 pp = rc5unitwork->plain.lo;
    u32 cc = rc5unitwork->cypher.lo;
    mask = 1;
    for (i=0; i<64; i++) {
	plain[i] = (pp & mask) ? ~(0ul) : 0;
	cypher[i] = (cc & mask) ? ~(0ul) : 0;
	if ((u32)(mask <<= 1) == 0) {
	    pp = rc5unitwork->plain.hi;
	    cc = rc5unitwork->cypher.hi;
	    mask = 1;
	}
    }

      // now we must generate 32/64 different keys with
      // the least significant bits we will modify
      // key[] & 0x00000001 == first key
      // key[] & 0x00000002 == second key
      // key[] & 0x00000004 == third key
      // key[] & 0x00000008 == fourth key
      // etc ...
#ifdef BIT_32
    key[ 3] = 0xAAAAAAAAul;
    key[ 5] = 0xCCCCCCCCul;
    key[ 8] = 0xF0F0F0F0ul;
    key[10] = 0xFF00FF00ul;
    key[11] = 0xFFFF0000ul;
#elif BIT_64
    key[ 3] = 0xAAAAAAAAAAAAAAAAul;
    key[ 5] = 0xCCCCCCCCCCCCCCCCul;
    key[ 8] = 0x0F0F0F0FF0F0F0F0ul;
    key[10] = 0xFF00FF00FF00FF00ul;
    key[11] = 0xFFFF0000FFFF0000ul;
    key[12] = 0xFFFFFFFF00000000ul;
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

      // first three/two bits are BrydDES fixed bits
      // the 8 followings are Meggs' fixed bits
      // the remaining bits are just here if we got a timeslice > 65536
      // (up to a timelice of 2^24, that should be ok...)
      // (these bits needs to be compatible with the convertion routines)
#ifdef BIT_32
    static u8 twiddles[] = {12,15,18, 40,41,42,43,45,46,49,50, 0,1,2,4,6,7,9,13 };
    u32 nbbits_to_twiddle = nbbits - 5;
#elif BIT_64
    static u8 twiddles[] = {15,18, 40,41,42,43,45,46,49,50, 0,1,2,4,6,7,9,13 };
    u32 nbbits_to_twiddle = nbbits - 6;
#endif

      // Zero out all the bits that are to be varied
    for (i=0; i<nbbits_to_twiddle; i++) key[twiddles[i]] = 0;

      // now we iterates, checking 32/64 keys each time
      // we call deseval()
    bool complement = false;
    u32 bkey = 0;
    u32 gkey = 0;
    unsigned long result;
    for (;;) {
	  // check 32/64 keys
	result = deseval(plain, cypher, key);
	if (result != 0) break;
	  // increment the binary order key and check
	  // if we need to do the complement
	if (++bkey >= (1ul << nbbits_to_twiddle))
	    if (!complement) {
		  // reset counters
		complement = true;
		bkey = gkey = 0;
		  // Invert keys
		for (i=0; i<56; i++) key[i] ^= ~(0ul);
		  // Zero out all the bits that are to be varied
		for (i=0; i<nbbits_to_twiddle; i++) key[twiddles[i]] = 0;
		  // restart
		continue;
	    } else
		break;
	  // Increment the gray-order key and find out which bit changed
	u32 tmp = gkey;
	gkey = bkey ^ (bkey>>1);
	tmp ^= gkey;
	for (i = 0; i<nbbits_to_twiddle && ((tmp>>i)&1)==0; i++);
	  // Update the keys
	key[twiddles[i]] ^= ~(0ul);
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

#ifdef DEBUG
	  // check to see if we really have consecutive keys
	  // and if the winning key is at the right place
	for (i=0; i<BITS_PER_SLICE; i++) {
	    keylo = keyhi = 0;
	    for (int j=55; j>0; j-=7) {
		u32 byte = odd_parity [
		    (((key[j-0] >> i) & 1) << 7) |
		    (((key[j-1] >> i) & 1) << 6) |
		    (((key[j-2] >> i) & 1) << 5) |
		    (((key[j-3] >> i) & 1) << 4) |
		    (((key[j-4] >> i) & 1) << 3) |
		    (((key[j-5] >> i) & 1) << 2) |
		    (((key[j-6] >> i) & 1) << 1)];
		int numbyte = (j+1) / 7 - 1;
		if (numbyte >= 4)
		    keyhi |= byte << ((numbyte-4)*8);
		else
		    keylo |= byte << (numbyte*8);
	    }
	    if (complement) {
		keyhi ^= 0xFFFFFFFFu;
		keylo ^= 0xFFFFFFFFu;
	    }
	    convert_key_from_des_to_inc (&keyhi, &keylo);
	    printf ("key %08X:%08X%c\n", keyhi, keylo, numkeyfound==i ? '*':' ');
	}
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
	    keyhi ^= 0xFFFFFFFFu;
	    keylo ^= 0xFFFFFFFFu;
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

