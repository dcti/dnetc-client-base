// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-slice-dworz.cpp,v $
// Revision 1.1  1999/01/18 18:37:37  remi
// Added Christoph Dworzak new alpha bitslicer.
//
// Revision 1.2  1999/01/17 08:26:59  dworz
//


//static char *id="@(#)$Id: des-slice-dworz.cpp,v 1.1 1999/01/18 18:37:37 remi Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/problem.h"
#include "../common/convdes.h"

extern unsigned long PT[64], CT[64];
extern "C" unsigned long checkKey (unsigned long *K, unsigned long init);

unsigned long scramble(unsigned long l) {
    unsigned long binary;

    binary  = ((l >>  7) & 1UL) << (15+ 0);
    binary |= ((l >> 15) & 1UL) << ( 5+ 0);
    binary |= ((l >> 23) & 1UL) << (16+ 0);
    binary |= ((l >> 31) & 1UL) << (20+ 0);
    binary |= ((l >> 39) & 1UL) << (28+ 0);
    binary |= ((l >> 47) & 1UL) << ( 8+ 0);
    binary |= ((l >> 55) & 1UL) << (27+ 0);
    binary |= ((l >> 63) & 1UL) << (19+ 0);
    binary |= ((l >>  5) & 1UL) << ( 2+ 0);
    binary |= ((l >> 13) & 1UL) << (14+ 0);
    binary |= ((l >> 21) & 1UL) << (22+ 0);
    binary |= ((l >> 29) & 1UL) << (25+ 0);
    binary |= ((l >> 37) & 1UL) << ( 4+ 0);
    binary |= ((l >> 45) & 1UL) << (17+ 0);
    binary |= ((l >> 53) & 1UL) << (29+ 0);
    binary |= ((l >> 61) & 1UL) << (10+ 0);
    binary |= ((l >>  3) & 1UL) << ( 0+ 0);
    binary |= ((l >> 11) & 1UL) << ( 6+ 0);
    binary |= ((l >> 19) & 1UL) << (21+ 0);
    binary |= ((l >> 27) & 1UL) << (13+ 0);
    binary |= ((l >> 35) & 1UL) << (31+ 0);
    binary |= ((l >> 43) & 1UL) << (26+ 0);
    binary |= ((l >> 51) & 1UL) << ( 3+ 0);
    binary |= ((l >> 59) & 1UL) << (11+ 0);
    binary |= ((l >>  1) & 1UL) << (18+ 0);
    binary |= ((l >>  9) & 1UL) << (12+ 0);
    binary |= ((l >> 17) & 1UL) << (30+ 0);
    binary |= ((l >> 25) & 1UL) << ( 7+ 0);
    binary |= ((l >> 33) & 1UL) << (23+ 0);
    binary |= ((l >> 41) & 1UL) << ( 9+ 0);
    binary |= ((l >> 49) & 1UL) << ( 1+ 0);
    binary |= ((l >> 57) & 1UL) << (24+ 0);
    binary |= ((l >>  6) & 1UL) << (15+32);
    binary |= ((l >> 14) & 1UL) << ( 5+32);
    binary |= ((l >> 22) & 1UL) << (16+32);
    binary |= ((l >> 30) & 1UL) << (20+32);
    binary |= ((l >> 38) & 1UL) << (28+32);
    binary |= ((l >> 46) & 1UL) << ( 8+32);
    binary |= ((l >> 54) & 1UL) << (27+32);
    binary |= ((l >> 62) & 1UL) << (19+32);
    binary |= ((l >>  4) & 1UL) << ( 2+32);
    binary |= ((l >> 12) & 1UL) << (14+32);
    binary |= ((l >> 20) & 1UL) << (22+32);
    binary |= ((l >> 28) & 1UL) << (25+32);
    binary |= ((l >> 36) & 1UL) << ( 4+32);
    binary |= ((l >> 44) & 1UL) << (17+32);
    binary |= ((l >> 52) & 1UL) << (29+32);
    binary |= ((l >> 60) & 1UL) << (10+32);
    binary |= ((l >>  2) & 1UL) << ( 0+32);
    binary |= ((l >> 10) & 1UL) << ( 6+32);
    binary |= ((l >> 18) & 1UL) << (21+32);
    binary |= ((l >> 26) & 1UL) << (13+32);
    binary |= ((l >> 34) & 1UL) << (31+32);
    binary |= ((l >> 42) & 1UL) << (26+32);
    binary |= ((l >> 50) & 1UL) << ( 3+32);
    binary |= ((l >> 58) & 1UL) << (11+32);
    binary |= ((l >>  0) & 1UL) << (18+32);
    binary |= ((l >>  8) & 1UL) << (12+32);
    binary |= ((l >> 16) & 1UL) << (30+32);
    binary |= ((l >> 24) & 1UL) << ( 7+32);
    binary |= ((l >> 32) & 1UL) << (23+32);
    binary |= ((l >> 40) & 1UL) << ( 9+32);
    binary |= ((l >> 48) & 1UL) << ( 1+32);
    binary |= ((l >> 56) & 1UL) << (24+32);
    return(binary);
}

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 nbbits )
{
    unsigned long K[56];
    unsigned long i, j, result, SK, EK;

    // check nbbits
    if (nbbits != 20) {
	printf ("Bad nbbits ! (%d)\n", nbbits);
	exit (-1);
    }

	j = scramble(((unsigned long)rc5unitwork->plain.hi<<32)|(unsigned long)rc5unitwork->plain.lo);
	for (i=0; i<64; i++) if (j & (1UL << i)) PT[i] = ~0UL; else PT[i] = 0;
	j = scramble(((unsigned long)rc5unitwork->cypher.hi<<32)|(unsigned long)rc5unitwork->cypher.lo);
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
	    K[0] = 0xFFFFFFFF00000000UL; K[1] = 0xFFFF0000FFFF0000UL;
	    K[2] = 0xFF00FF00FF00FF00UL; K[3] = 0xF0F0F0F0F0F0F0F0UL;
	    K[4] = 0xCCCCCCCCCCCCCCCCUL; K[5] = 0xAAAAAAAAAAAAAAAAUL;
	    K[ 8] = ~0UL; K[10] = ~0UL; K[11] = ~0UL; K[12] = ~0UL;
	    K[15] = ~0UL; K[18] = ~0UL; K[42] = ~0UL; K[43] = ~0UL;
	    K[45] = ~0UL; K[46] = ~0UL; K[49] = ~0UL; K[50] = ~0UL;
	    K[40] = ~0UL; K[41] = ~0UL;
	    do{do{do{do{do{do{do{do{do{
  			if ((result=checkKey(K, 0))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[45] = ~K[45]; if ((result=checkKey(K, 2))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[49] = ~K[49]; if ((result=checkKey(K, 2))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[45] = ~K[45]; if ((result=checkKey(K, 2))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[46] = ~K[46]; if ((result=checkKey(K, 4))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[45] = ~K[45]; if ((result=checkKey(K, 2))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[49] = ~K[49]; if ((result=checkKey(K, 2))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[45] = ~K[45]; if ((result=checkKey(K, 2))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
			K[18] = ~K[18]; if ((result=checkKey(K, 1))!=0) goto found;
			K[10] = ~K[10]; if ((result=checkKey(K, 1))!=0) goto found;
	    K[12] = ~K[12];} while (~K[12]);
	    K[15] = ~K[15];} while (~K[15]);
	    K[50] = ~K[50];} while (~K[50]);
  		K[ 8] = ~K[ 8];} while (~K[ 8]);
	    K[11] = ~K[11];} while (~K[11]);
	    K[42] = ~K[42];} while (~K[42]);
	    K[43] = ~K[43];} while (~K[43]);
	    K[40] = ~K[40];} while (~K[40]);
	    K[41] = ~K[41];} while (~K[41]);
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

