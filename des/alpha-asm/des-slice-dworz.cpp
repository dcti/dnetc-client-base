// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: des-slice-dworz.cpp,v $
// Revision 1.7  1999/04/04 16:41:21  dworz
// replaced (1UL << i) with (ONE << i)...
//
// Revision 1.6  1999/04/01 07:18:40  jlawson
// changed core function name to des_unit_func_alpha_dworz
//
// Revision 1.5  1999/02/21 09:53:20  silby
// Changes for large block support.
//
// Revision 1.4  1999/02/09 04:15:36  dworz
// moved keyschedule to deseval-dworz3.S
// changed the setup of PT and CT
//
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
return "@(#)$Id: des-slice-dworz.cpp,v 1.7 1999/04/04 16:41:21 dworz Exp $"; }
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "problem.h"
#include "convdes.h"
#include "logstuff.h"

#ifndef DWORZ
#error "You must compile with -DDWORZ.  Set this and then recompile."
#endif

#if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32) && defined(BIT_64) && (_MSC_VER >= 11)
typedef unsigned __int64 WORD_TYPE;
#else
typedef unsigned long WORD_TYPE;
#endif

// Use these symbols to force 64bit versions of these constants.
// 1UL<<63 is zero under NT/Alpha since 1UL is treated as a 32bit int,
// and doesn't get promoted until __int64 until too late.
const WORD_TYPE ONE = 1UL;
const WORD_TYPE NOTZERO = ~(WORD_TYPE)0;

struct DesWorkStruct {
   WORD_TYPE K[56], PT[64], CT[64], ct[96], t[64], pt[96*64];
};

extern "C" WORD_TYPE checkKey (DesWorkStruct *dws);

// ------------------------------------------------------------------
// Input : 56 bit key, plain & cypher text, timeslice
// Output: key incremented, return 'timeslice' if no key found, 
//         'timeslice-something' else
// note : nbbits can't be less than 19 when BIT_32 is defined
// and can't be less than 20 when BIT_64

// rc5unitwork.LO in lo:hi 24+32 incrementable format

extern "C"
u32 des_unit_func_alpha_dworz( RC5UnitWork * rc5unitwork, u32 nbbits )
{
  WORD_TYPE i, j, result, SK, EK;
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
  for (i=0; i<32; i++){
    dws.PT[i   ] = ((j&(ONE<<(2*i  )))) ? NOTZERO : 0;
    dws.PT[i+32] = ((j&(ONE<<(2*i+1)))) ? NOTZERO : 0;
  }
  
  j = (WORD_TYPE)rc5unitwork->cypher.hi<<32|(WORD_TYPE)rc5unitwork->cypher.lo;
#ifdef DEBUG
  Log(" CT=%08x%08x\n",j>>32,j);
#endif
  for (i = 0; i < 32; i++){
    dws.CT[i   ] = ((j&(ONE<<(2*i  )))) ? NOTZERO : 0;
    dws.CT[i+32] = ((j&(ONE<<(2*i+1)))) ? NOTZERO : 0;
  }
  
  // convert the starting key from incrementable format
  // to DES format
  u32 keyhi = rc5unitwork->L0.hi;
  u32 keylo = rc5unitwork->L0.lo;
  convert_key_from_inc_to_des (&keyhi, &keylo);


  SK = ((WORD_TYPE)keylo)|((WORD_TYPE)keyhi<<32);
  SK = ((SK & 0xFEUL)>>1) |
    ((SK & 0xFE00UL)>>2) |
    ((SK & 0xFE0000UL)>>3) |
    ((SK & 0xFE000000UL)>>4) |
    ((SK & 0xFE00000000UL)>>5) |
    ((SK & 0xFE0000000000UL)>>6) |
    ((SK & 0xFE000000000000UL)>>7) |
    ((SK & 0xFE00000000000000UL)>>8);
#ifdef DEBUG
  Log(" SK=%08x%08x\n",SK>>32,SK);
#endif
  for (i = 0; i < 56; i++) dws.K[i] = (SK & (ONE << i)) ? NOTZERO : 0;
  
  dws.K[ 0] = 0xFFFFFFFF00000000UL; dws.K[ 1] = 0xFFFF0000FFFF0000UL;
  dws.K[ 2] = 0xFF00FF00FF00FF00UL; dws.K[40] = 0xF0F0F0F0F0F0F0F0UL;
  dws.K[ 4] = 0xCCCCCCCCCCCCCCCCUL; dws.K[41] = 0xAAAAAAAAAAAAAAAAUL;
  
  dws.K[ 3] = 0; dws.K[ 5] = 0; dws.K[ 8] = 0; dws.K[10] = 0;
  dws.K[11] = 0; dws.K[12] = 0; dws.K[15] = 0; dws.K[18] = 0;
  dws.K[42] = 0; dws.K[43] = 0; dws.K[45] = 0; dws.K[46] = 0;
  dws.K[49] = 0; dws.K[50] = 0;
  
  if ((result = checkKey(&dws))!=0) goto found;
  
#ifdef DEBUG
  Log(" -> EK not found\n");
#endif
  rc5unitwork->L0.lo += 1 << nbbits; // Increment lower 32 bits
  if (rc5unitwork->L0.lo < (u32)(1 << nbbits) )
    rc5unitwork->L0.hi++; // Carry to high 32 bits if needed
  return 1 << nbbits;
  
 found:
  for (i = EK = 0UL; i < 56; i++)
    if (dws.K[i] & result)
      EK |= ONE << i;
  if ((EK ^ SK) & (ONE << 55)) EK = ~EK;
  EK =  (WORD_TYPE)odd_parity[(EK << 1) & 0xFEUL] |
    (WORD_TYPE)odd_parity[(EK >> 6) & 0xFEUL] << 8 |
    (WORD_TYPE)odd_parity[(EK >> 13) & 0xFEUL] << 16 |
    (WORD_TYPE)odd_parity[(EK >> 20) & 0xFEUL] << 24 |
    (WORD_TYPE)odd_parity[(EK >> 27) & 0xFEUL] << 32 |
    (WORD_TYPE)odd_parity[(EK >> 34) & 0xFEUL] << 40 |
    (WORD_TYPE)odd_parity[(EK >> 41) & 0xFEUL] << 48 |
    (WORD_TYPE)odd_parity[(EK >> 48) & 0xFEUL] << 56;
#ifdef DEBUG
  Log("-> Key=%08x%08x\n", EK >> 32, EK);
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
