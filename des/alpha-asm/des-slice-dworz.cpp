// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-slice-dworz.cpp,v $
// Revision 1.2  1999/01/23 14:47:14  remi
// Even faster DES bitslicer for Alpha machines.
// Works with deseval-dworz3.S, not deseval-dworz2.c
//
// Revision 1.1  1999/01/18 18:37:37  remi
// Added Christoph Dworzak new alpha bitslicer.
//


#if (!defined(lint) && defined(__showids__))
const char *des_slice_dworz_cpp(void) {
return "@(#)$Id: des-slice-dworz.cpp,v 1.2 1999/01/23 14:47:14 remi Exp $"; }
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "problem.h"
#include "convdes.h"

extern unsigned long PT[64], CT[64];
extern "C" unsigned long checkKey (unsigned long *K, unsigned long init);

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 nbbits )
{
    unsigned long K[56];
    unsigned long i, j, l, m, result, SK, EK;

    // check nbbits
    if (nbbits != 20) {
	printf ("Bad nbbits ! (%d)\n", nbbits);
	exit (-1);
    }

	j = (unsigned long)rc5unitwork->plain.hi<<32|(unsigned long)rc5unitwork->plain.lo;
	for (i=0; i<64; i++) if (j & (1UL << i)) PT[i] = ~0UL; else PT[i] = 0;
	j = (unsigned long)rc5unitwork->cypher.hi<<32|(unsigned long)rc5unitwork->cypher.lo;
	for (i=0; i<64; i++) if (j & (1UL << i)) CT[i] = ~0UL; else CT[i] = 0;

    // convert the starting key from incrementable format
    // to DES format
    u32 keyhi = rc5unitwork->L0.hi;
    u32 keylo = rc5unitwork->L0.lo;
    convert_key_from_inc_to_des (&keyhi, &keylo);


	SK = ((unsigned long)keylo)|((unsigned long)keyhi<<32);
	SK = ((SK&0xFEUL)>>1) |
		 ((SK&0xFE00UL)>>2) |
		 ((SK&0xFE0000UL)>>3) |
		 ((SK&0xFE000000UL)>>4) |
		 ((SK&0xFE00000000UL)>>5) |
		 ((SK&0xFE0000000000UL)>>6) |
		 ((SK&0xFE000000000000UL)>>7) |
		 ((SK&0xFE00000000000000UL)>>8);
    for (j=2;j;j--){
  		for (i=0; i<56; i++) if (SK & (1UL << i)) K[i] = ~0UL; else K[i] = 0;

	    K[ 0] = 0xFFFFFFFF00000000UL; K[ 1] = 0xFFFF0000FFFF0000UL;
	    K[ 2] = 0xFF00FF00FF00FF00UL; K[40] = 0xF0F0F0F0F0F0F0F0UL;
	    K[ 4] = 0xCCCCCCCCCCCCCCCCUL; K[41] = 0xAAAAAAAAAAAAAAAAUL;

	    K[ 3] = 0; K[ 5] = 0; K[ 8] = 0; K[10] = 0;
		K[11] = 0; K[12] = 0; K[15] = 0; K[18] = 0;
		K[42] = 0; K[43] = 0; K[45] = 0; K[46] = 0;
		K[49] = 0; K[50] = 0;

		l=0x7f; m=0xff;
next:
  		K[43] = ~K[43]; if ((result = checkKey(K, l|0x00f80UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x01200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x02100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x03200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x04400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x05200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x06100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x07200UL))!=0) goto found;
		K[11] = ~K[11]; if ((result = checkKey(K, l|0x08200UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x09200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x0a100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x0b200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x0c400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x0d200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x0e100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x0f200UL))!=0) goto found;
		K[42] = ~K[42]; if ((result = checkKey(K, l|0x10800UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x11200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x12100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x13200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x14400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x15200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x16100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x17200UL))!=0) goto found;
		K[11] = ~K[11]; if ((result = checkKey(K, l|0x18200UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x19200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x1a100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x1b200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x1c400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x1d200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x1e100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x1f200UL))!=0) goto found;
		K[43] = ~K[43]; if ((result = checkKey(K, l|0x20400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x21200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x22100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x23200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x24400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x25200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x26100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x27200UL))!=0) goto found;
		K[11] = ~K[11]; if ((result = checkKey(K, l|0x28200UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x29200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x2a100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x2b200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x2c400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x2d200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x2e100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x2f200UL))!=0) goto found;
		K[42] = ~K[42]; if ((result = checkKey(K, l|0x30800UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x31200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x32100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x33200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x34400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x35200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x36100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x37200UL))!=0) goto found;
		K[11] = ~K[11]; if ((result = checkKey(K, l|0x38200UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x39200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x3a100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x3b200UL))!=0) goto found;
		K[ 8] = ~K[ 8]; if ((result = checkKey(K, l|0x3c400UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x3d200UL))!=0) goto found;
		K[ 5] = ~K[ 5]; if ((result = checkKey(K, l|0x3e100UL))!=0) goto found;
		K[ 3] = ~K[ 3]; if ((result = checkKey(K, l|0x3f200UL))!=0) goto found;

		if (m&0x01){ K[10] = ~K[10]; l=0x02; m^=0x01; goto next; }
		if (m&0x02){ K[18] = ~K[18]; l=0x02; m^=0x03; goto next; }
		if (m&0x04){ K[45] = ~K[45]; l=0x20; m^=0x07; goto next; }
		if (m&0x08){ K[49] = ~K[49]; l=0x20; m^=0x0f; goto next; }
		if (m&0x10){ K[12] = ~K[12]; l=0x08; m^=0x1f; goto next; }
		if (m&0x20){ K[15] = ~K[15]; l=0x10; m^=0x3f; goto next; }
		if (m&0x40){ K[46] = ~K[46]; l=0x04; m^=0x7f; goto next; }
		if (m&0x80){ K[50] = ~K[50]; l=0x40; m^=0xff; goto next; }

	    SK = ~SK;
    }
	rc5unitwork->L0.lo += 1 << nbbits;
	return 1 << nbbits;

found:
    for (i=EK=0UL;i<56;i++)
	if (K[i]&result)
	    EK |= 1UL<<i;
	if ((EK^SK)&(1UL<<63)) EK = ~EK;
	EK =  (unsigned long)odd_parity[EK<<1&0xFEUL] |
		  (unsigned long)odd_parity[EK>>6&0xFEUL]<<8 |
		  (unsigned long)odd_parity[EK>>13&0xFEUL]<<16 |
		  (unsigned long)odd_parity[EK>>20&0xFEUL]<<24 |
		  (unsigned long)odd_parity[EK>>27&0xFEUL]<<32 |
		  (unsigned long)odd_parity[EK>>34&0xFEUL]<<40 |
		  (unsigned long)odd_parity[EK>>41&0xFEUL]<<48 |
		  (unsigned long)odd_parity[EK>>48&0xFEUL]<<56;
#if defined(DEBUG)
    printf("-> Key=%014lX\n", EK);
#endif
	keyhi = EK>>32;
	keylo = EK&0xffffffff;
	// convert key from 64 bits DES ordering with parity
	// to incrementable format
	convert_key_from_des_to_inc (&keyhi, &keylo);
	
	u32 nbkeys = keylo - rc5unitwork->L0.lo;
	rc5unitwork->L0.lo = keylo;
	rc5unitwork->L0.hi = keyhi;

	return nbkeys;
}

