// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-slice-dworz.cpp,v $
// Revision 1.3  1999/01/26 17:30:25  michmarc
// Made thread safe and Alpha/NT compatable
//
// Revision 1.2  1999/01/23 14:47:14  remi
// Even faster DES bitslicer for Alpha machines.
// Works with deseval-dworz3.S, not deseval-dworz2.c
//
// Revision 1.1  1999/01/18 18:37:37  remi
// Added Christoph Dworzak new alpha bitslicer.
//


#if (!defined(lint) && defined(__showids__))
const char *des_slice_dworz_cpp(void) {
return "@(#)$Id: des-slice-dworz.cpp,v 1.3 1999/01/26 17:30:25 michmarc Exp $"; }
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "problem.h"
#include "convdes.h"
#include "logstuff.h"

#ifndef DWORZ
#error "You must compile with -DDWORZ.  Set this and then recompile cliconfig.cpp"
#endif

#if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32) && defined(BIT_64) && (_MSC_VER >= 11)
typedef unsigned __int64 WORD_TYPE;
#else
typedef unsigned long WORD_TYPE;
#endif

// Use these symbols to force 64bit versions of these constants.  1UL<<63 is zero
// under NT/Alpha since 1UL is treated as a 32bit int, and doesn't get promoted until
// __int64 until too late.
const WORD_TYPE ONE = 1UL;
const WORD_TYPE NOTZERO = ~(WORD_TYPE)0;

struct DesWorkStruct {
   WORD_TYPE K[56], PT[64], CT[64], ct[64], t[64], pt[64*64];
};

//extern "C" { WORD_TYPE PT[64], CT[64], ct[64], pt[64*64], t[64]; };
extern "C" WORD_TYPE checkKey (DesWorkStruct *dws, WORD_TYPE init);

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

u32 des_unit_func( RC5UnitWork * rc5unitwork, u32 nbbits )
{
    WORD_TYPE i, j, l, m, result, SK, EK;
    DesWorkStruct dws;

    // check nbbits
    if (nbbits != 20) {
	Log ("Bad nbbits ! (%d)\n", nbbits);
	exit (-1);
    }

	j = (WORD_TYPE)rc5unitwork->plain.hi<<32|(WORD_TYPE)rc5unitwork->plain.lo;
#ifdef DEBUG
        Log("PT=%08x%08x\n",j>>32,j);
#endif
	for (i=0; i<64; i++) dws.PT[i] = ((j & (ONE << i))) ? NOTZERO : 0;
	j = (WORD_TYPE)rc5unitwork->cypher.hi<<32|(WORD_TYPE)rc5unitwork->cypher.lo;
#ifdef DEBUG
        Log(" CT=%08x%08x\n",j>>32,j);
#endif
	for (i=0; i<64; i++) dws.CT[i] = ((j & (ONE << i))) ? NOTZERO : 0;

    // convert the starting key from incrementable format
    // to DES format
    u32 keyhi = rc5unitwork->L0.hi;
    u32 keylo = rc5unitwork->L0.lo;
    convert_key_from_inc_to_des (&keyhi, &keylo);


	SK = ((WORD_TYPE)keylo)|((WORD_TYPE)keyhi<<32);
	SK = ((SK&0xFEUL)>>1) |
		 ((SK&0xFE00UL)>>2) |
		 ((SK&0xFE0000UL)>>3) |
		 ((SK&0xFE000000UL)>>4) |
		 ((SK&0xFE00000000UL)>>5) |
		 ((SK&0xFE0000000000UL)>>6) |
		 ((SK&0xFE000000000000UL)>>7) |
		 ((SK&0xFE00000000000000UL)>>8);
#ifdef DEBUG
    Log(" SK=%08x%08x\n",SK>>32,SK);
#endif
    for (j=2;j;j--){
  		for (i=0; i<56; i++) if (SK & (ONE << i)) dws.K[i] = NOTZERO; else dws.K[i] = 0;

	    dws.K[ 0] = 0xFFFFFFFF00000000UL; dws.K[ 1] = 0xFFFF0000FFFF0000UL;
	    dws.K[ 2] = 0xFF00FF00FF00FF00UL; dws.K[40] = 0xF0F0F0F0F0F0F0F0UL;
	    dws.K[ 4] = 0xCCCCCCCCCCCCCCCCUL; dws.K[41] = 0xAAAAAAAAAAAAAAAAUL;

	    dws.K[ 3] = 0; dws.K[ 5] = 0; dws.K[ 8] = 0; dws.K[10] = 0;
		dws.K[11] = 0; dws.K[12] = 0; dws.K[15] = 0; dws.K[18] = 0;
		dws.K[42] = 0; dws.K[43] = 0; dws.K[45] = 0; dws.K[46] = 0;
		dws.K[49] = 0; dws.K[50] = 0;

		l=0x7f; m=0xff;
next:
  		dws.K[43] = ~dws.K[43]; if ((result = checkKey(&dws, l|0x00f80UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x01200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x02100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x03200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x04400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x05200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x06100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x07200UL))!=0) goto found;
		dws.K[11] = ~dws.K[11]; if ((result = checkKey(&dws, l|0x08200UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x09200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x0a100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x0b200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x0c400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x0d200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x0e100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x0f200UL))!=0) goto found;
		dws.K[42] = ~dws.K[42]; if ((result = checkKey(&dws, l|0x10800UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x11200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x12100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x13200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x14400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x15200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x16100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x17200UL))!=0) goto found;
		dws.K[11] = ~dws.K[11]; if ((result = checkKey(&dws, l|0x18200UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x19200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x1a100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x1b200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x1c400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x1d200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x1e100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x1f200UL))!=0) goto found;
		dws.K[43] = ~dws.K[43]; if ((result = checkKey(&dws, l|0x20400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x21200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x22100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x23200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x24400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x25200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x26100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x27200UL))!=0) goto found;
		dws.K[11] = ~dws.K[11]; if ((result = checkKey(&dws, l|0x28200UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x29200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x2a100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x2b200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x2c400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x2d200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x2e100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x2f200UL))!=0) goto found;
		dws.K[42] = ~dws.K[42]; if ((result = checkKey(&dws, l|0x30800UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x31200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x32100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x33200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x34400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x35200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x36100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x37200UL))!=0) goto found;
		dws.K[11] = ~dws.K[11]; if ((result = checkKey(&dws, l|0x38200UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x39200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x3a100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x3b200UL))!=0) goto found;
		dws.K[ 8] = ~dws.K[ 8]; if ((result = checkKey(&dws, l|0x3c400UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x3d200UL))!=0) goto found;
		dws.K[ 5] = ~dws.K[ 5]; if ((result = checkKey(&dws, l|0x3e100UL))!=0) goto found;
		dws.K[ 3] = ~dws.K[ 3]; if ((result = checkKey(&dws, l|0x3f200UL))!=0) goto found;

		if (m&0x01){ dws.K[10] = ~dws.K[10]; l=0x02; m^=0x01; goto next; }
		if (m&0x02){ dws.K[18] = ~dws.K[18]; l=0x02; m^=0x03; goto next; }
		if (m&0x04){ dws.K[45] = ~dws.K[45]; l=0x20; m^=0x07; goto next; }
		if (m&0x08){ dws.K[49] = ~dws.K[49]; l=0x20; m^=0x0f; goto next; }
		if (m&0x10){ dws.K[12] = ~dws.K[12]; l=0x08; m^=0x1f; goto next; }
		if (m&0x20){ dws.K[15] = ~dws.K[15]; l=0x10; m^=0x3f; goto next; }
		if (m&0x40){ dws.K[46] = ~dws.K[46]; l=0x04; m^=0x7f; goto next; }
		if (m&0x80){ dws.K[50] = ~dws.K[50]; l=0x40; m^=0xff; goto next; }

	    SK = ~SK;
    }
#ifdef DEBUG
    Log(" -> EK not found\n");
#endif
	rc5unitwork->L0.lo += 1 << nbbits;
	return 1 << nbbits;

found:
    for (i=EK=0UL;i<56;i++)
	if (dws.K[i]&result)
	    EK |= ONE<<i;
	if ((EK^SK)&(ONE<<63)) EK = ~EK;
	EK =  (WORD_TYPE)odd_parity[EK<<1&0xFEUL] |
		  (WORD_TYPE)odd_parity[EK>>6&0xFEUL]<<8 |
		  (WORD_TYPE)odd_parity[EK>>13&0xFEUL]<<16 |
		  (WORD_TYPE)odd_parity[EK>>20&0xFEUL]<<24 |
		  (WORD_TYPE)odd_parity[EK>>27&0xFEUL]<<32 |
		  (WORD_TYPE)odd_parity[EK>>34&0xFEUL]<<40 |
		  (WORD_TYPE)odd_parity[EK>>41&0xFEUL]<<48 |
		  (WORD_TYPE)odd_parity[EK>>48&0xFEUL]<<56;
#ifdef DEBUG
    Log("-> Key=%08x%08x\n", EK>>32,EK);
#endif
	keyhi = (u32)(EK>>32);
	keylo = (u32)(EK&0xffffffff);
	// convert key from 64 bits DES ordering with parity
	// to incrementable format
	convert_key_from_des_to_inc (&keyhi, &keylo);
	
	u32 nbkeys = keylo - rc5unitwork->L0.lo;
	rc5unitwork->L0.lo = keylo;
	rc5unitwork->L0.hi = keyhi;

	return nbkeys;
}
