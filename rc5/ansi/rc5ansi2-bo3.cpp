// dual-key, mixed round 3 and encryption, direct use of last value,
// non-arrayed S1/S2 tables

// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: rc5ansi2-bo3.cpp,v $
// Revision 1.4  1998/06/14 08:13:50  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


/*  This file is included from rc5.cpp so we can use __inline__.  */

#include "problem.h"
#include "rotate.h"


#if (PIPELINE_COUNT != 2)
#error "Expecting pipeline count of 2"
#endif

#ifndef _CPU_32BIT_
#error "everything assumes a 32bit CPU..."
#endif


#define P     0xB7E15163
#define Q     0x9E3779B9

#define S_not(n)      P+Q*n


// rc5_unit will get passed an RC5WorkUnit to complete
// this is where all the actually work occurs, this is where you optimize.
// assembly gurus encouraged.
// Returns: 0 - nothing found, 1 - found on pipeline 1,
//   2 - found pipeline 2, 3 - ... etc ...

static __inline__
u32 rc5_unit_func( RC5UnitWork * rc5unitwork )
{
    /* Plaintext and Ciphertext word pairs */
    u32 PlainLo, PlainHi;
    u32 CypherLo, CypherHi;
    u32 Llo1, Lhi1;
    u32 Llo2, Lhi2;
    u32 Elo1, Ehi1;
    u32 Elo2, Ehi2;

    u32 S1_00,S1_01,S1_02,S1_03,S1_04,S1_05,S1_06,S1_07,S1_08,S1_09,S1_10,S1_11,S1_12,
	     S1_13,S1_14,S1_15,S1_16,S1_17,S1_18,S1_19,S1_20,S1_21,S1_22,S1_23,S1_24,S1_25;

    u32 S2_00,S2_01,S2_02,S2_03,S2_04,S2_05,S2_06,S2_07,S2_08,S2_09,S2_10,S2_11,S2_12,
	     S2_13,S2_14,S2_15,S2_16,S2_17,S2_18,S2_19,S2_20,S2_21,S2_22,S2_23,S2_24,S2_25;


    PlainLo = rc5unitwork->plain.lo;
    PlainHi = rc5unitwork->plain.hi;

    CypherLo = rc5unitwork->cypher.lo;
    CypherHi = rc5unitwork->cypher.lo;

    Llo2 = Llo1 = rc5unitwork->L0.lo;
    Lhi2 = (Lhi1 = rc5unitwork->L0.hi) + 0x01000000;


    /* Begin round 1 of key expansion */


    /*  Special case while A and B are known to be zero.  */
    S1_00 = ROTL3(S_not(0));
    Llo1 = ROTL(Llo1 + S1_00, S1_00);
    S2_00 = ROTL3(S_not(0));
    Llo2 = ROTL(Llo2 + S2_00, S2_00);

    S1_01 = ROTL3(S_not(1) + S1_00 + Llo1);
    S2_01 = ROTL3(S_not(1) + S2_00 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);

    S1_02 = ROTL3(S_not(2) + S1_01 + Lhi1);
    S2_02 = ROTL3(S_not(2) + S2_01 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_02 + Lhi1, S1_02 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_02 + Lhi2, S2_02 + Lhi2);

    S1_03 = ROTL3(S_not(3) + S1_02 + Llo1);
    S2_03 = ROTL3(S_not(3) + S2_02 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_03 + Llo1, S1_03 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_03 + Llo2, S2_03 + Llo2);

    S1_04 = ROTL3(S_not(4) + S1_03 + Lhi1);
    S2_04 = ROTL3(S_not(4) + S2_03 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_04 + Lhi1, S1_04 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_04 + Lhi2, S2_04 + Lhi2);

    S1_05 = ROTL3(S_not(5) + S1_04 + Llo1);
    S2_05 = ROTL3(S_not(5) + S2_04 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_05 + Llo1, S1_05 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_05 + Llo2, S2_05 + Llo2);

    S1_06 = ROTL3(S_not(6) + S1_05 + Lhi1);
    S2_06 = ROTL3(S_not(6) + S2_05 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_06 + Lhi1, S1_06 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_06 + Lhi2, S2_06 + Lhi2);

    S1_07 = ROTL3(S_not(7) + S1_06 + Llo1);
    S2_07 = ROTL3(S_not(7) + S2_06 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_07 + Llo1, S1_07 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_07 + Llo2, S2_07 + Llo2);

    S1_08 = ROTL3(S_not(8) + S1_07 + Lhi1);
    S2_08 = ROTL3(S_not(8) + S2_07 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_08 + Lhi1, S1_08 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_08 + Lhi2, S2_08 + Lhi2);

    S1_09 = ROTL3(S_not(9) + S1_08 + Llo1);
    S2_09 = ROTL3(S_not(9) + S2_08 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_09 + Llo1, S1_09 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_09 + Llo2, S2_09 + Llo2);
    S1_10 = ROTL3(S_not(10) + S1_09 + Lhi1);
    S2_10 = ROTL3(S_not(10) + S2_09 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_10 + Lhi1, S1_10 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_10 + Lhi2, S2_10 + Lhi2);
    S1_11 = ROTL3(S_not(11) + S1_10 + Llo1);
    S2_11 = ROTL3(S_not(11) + S2_10 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_11 + Llo1, S1_11 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_11 + Llo2, S2_11 + Llo2);
    S1_12 = ROTL3(S_not(12) + S1_11 + Lhi1);
    S2_12 = ROTL3(S_not(12) + S2_11 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_12 + Lhi1, S1_12 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_12 + Lhi2, S2_12 + Lhi2);
    S1_13 = ROTL3(S_not(13) + S1_12 + Llo1);
    S2_13 = ROTL3(S_not(13) + S2_12 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_13 + Llo1, S1_13 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_13 + Llo2, S2_13 + Llo2);
    S1_14 = ROTL3(S_not(14) + S1_13 + Lhi1);
    S2_14 = ROTL3(S_not(14) + S2_13 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_14 + Lhi1, S1_14 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_14 + Lhi2, S2_14 + Lhi2);
    S1_15 = ROTL3(S_not(15) + S1_14 + Llo1);
    S2_15 = ROTL3(S_not(15) + S2_14 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_15 + Llo1, S1_15 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_15 + Llo2, S2_15 + Llo2);
    S1_16 = ROTL3(S_not(16) + S1_15 + Lhi1);
    S2_16 = ROTL3(S_not(16) + S2_15 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_16 + Lhi1, S1_16 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_16 + Lhi2, S2_16 + Lhi2);
    S1_17 = ROTL3(S_not(17) + S1_16 + Llo1);
    S2_17 = ROTL3(S_not(17) + S2_16 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_17 + Llo1, S1_17 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_17 + Llo2, S2_17 + Llo2);
    S1_18 = ROTL3(S_not(18) + S1_17 + Lhi1);
    S2_18 = ROTL3(S_not(18) + S2_17 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_18 + Lhi1, S1_18 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_18 + Lhi2, S2_18 + Lhi2);
    S1_19 = ROTL3(S_not(19) + S1_18 + Llo1);
    S2_19 = ROTL3(S_not(19) + S2_18 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_19 + Llo1, S1_19 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_19 + Llo2, S2_19 + Llo2);
    S1_20 = ROTL3(S_not(20) + S1_19 + Lhi1);
    S2_20 = ROTL3(S_not(20) + S2_19 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_20 + Lhi1, S1_20 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_20 + Lhi2, S2_20 + Lhi2);
    S1_21 = ROTL3(S_not(21) + S1_20 + Llo1);
    S2_21 = ROTL3(S_not(21) + S2_20 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_21 + Llo1, S1_21 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_21 + Llo2, S2_21 + Llo2);
    S1_22 = ROTL3(S_not(22) + S1_21 + Lhi1);
    S2_22 = ROTL3(S_not(22) + S2_21 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_22 + Lhi1, S1_22 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_22 + Lhi2, S2_22 + Lhi2);
    S1_23 = ROTL3(S_not(23) + S1_22 + Llo1);
    S2_23 = ROTL3(S_not(23) + S2_22 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_23 + Llo1, S1_23 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_23 + Llo2, S2_23 + Llo2);
    S1_24 = ROTL3(S_not(24) + S1_23 + Lhi1);
    S2_24 = ROTL3(S_not(24) + S2_23 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_24 + Lhi1, S1_24 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_24 + Lhi2, S2_24 + Lhi2);
    S1_25 = ROTL3(S_not(25) + S1_24 + Llo1);
    S2_25 = ROTL3(S_not(25) + S2_24 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_25 + Llo1, S1_25 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_25 + Llo2, S2_25 + Llo2);

        /* Begin round 2 of key expansion */
    S1_00 = ROTL3(S1_00 + S1_25 + Lhi1);
    S2_00 = ROTL3(S2_00 + S2_25 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    S1_01 = ROTL3(S1_01 + S1_00 + Llo1);
    S2_01 = ROTL3(S2_01 + S2_00 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    S1_02 = ROTL3(S1_02 + S1_01 + Lhi1);
    S2_02 = ROTL3(S2_02 + S2_01 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_02 + Lhi1, S1_02 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_02 + Lhi2, S2_02 + Lhi2);
    S1_03 = ROTL3(S1_03 + S1_02 + Llo1);
    S2_03 = ROTL3(S2_03 + S2_02 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_03 + Llo1, S1_03 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_03 + Llo2, S2_03 + Llo2);
    S1_04 = ROTL3(S1_04 + S1_03 + Lhi1);
    S2_04 = ROTL3(S2_04 + S2_03 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_04 + Lhi1, S1_04 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_04 + Lhi2, S2_04 + Lhi2);
    S1_05 = ROTL3(S1_05 + S1_04 + Llo1);
    S2_05 = ROTL3(S2_05 + S2_04 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_05 + Llo1, S1_05 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_05 + Llo2, S2_05 + Llo2);
    S1_06 = ROTL3(S1_06 + S1_05 + Lhi1);
    S2_06 = ROTL3(S2_06 + S2_05 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_06 + Lhi1, S1_06 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_06 + Lhi2, S2_06 + Lhi2);
    S1_07 = ROTL3(S1_07 + S1_06 + Llo1);
    S2_07 = ROTL3(S2_07 + S2_06 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_07 + Llo1, S1_07 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_07 + Llo2, S2_07 + Llo2);
    S1_08 = ROTL3(S1_08 + S1_07 + Lhi1);
    S2_08 = ROTL3(S2_08 + S2_07 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_08 + Lhi1, S1_08 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_08 + Lhi2, S2_08 + Lhi2);
    S1_09 = ROTL3(S1_09 + S1_08 + Llo1);
    S2_09 = ROTL3(S2_09 + S2_08 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_09 + Llo1, S1_09 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_09 + Llo2, S2_09 + Llo2);
    S1_10 = ROTL3(S1_10 + S1_09 + Lhi1);
    S2_10 = ROTL3(S2_10 + S2_09 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_10 + Lhi1, S1_10 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_10 + Lhi2, S2_10 + Lhi2);
    S1_11 = ROTL3(S1_11 + S1_10 + Llo1);
    S2_11 = ROTL3(S2_11 + S2_10 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_11 + Llo1, S1_11 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_11 + Llo2, S2_11 + Llo2);
    S1_12 = ROTL3(S1_12 + S1_11 + Lhi1);
    S2_12 = ROTL3(S2_12 + S2_11 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_12 + Lhi1, S1_12 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_12 + Lhi2, S2_12 + Lhi2);
    S1_13 = ROTL3(S1_13 + S1_12 + Llo1);
    S2_13 = ROTL3(S2_13 + S2_12 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_13 + Llo1, S1_13 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_13 + Llo2, S2_13 + Llo2);
    S1_14 = ROTL3(S1_14 + S1_13 + Lhi1);
    S2_14 = ROTL3(S2_14 + S2_13 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_14 + Lhi1, S1_14 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_14 + Lhi2, S2_14 + Lhi2);
    S1_15 = ROTL3(S1_15 + S1_14 + Llo1);
    S2_15 = ROTL3(S2_15 + S2_14 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_15 + Llo1, S1_15 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_15 + Llo2, S2_15 + Llo2);
    S1_16 = ROTL3(S1_16 + S1_15 + Lhi1);
    S2_16 = ROTL3(S2_16 + S2_15 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_16 + Lhi1, S1_16 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_16 + Lhi2, S2_16 + Lhi2);
    S1_17 = ROTL3(S1_17 + S1_16 + Llo1);
    S2_17 = ROTL3(S2_17 + S2_16 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_17 + Llo1, S1_17 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_17 + Llo2, S2_17 + Llo2);
    S1_18 = ROTL3(S1_18 + S1_17 + Lhi1);
    S2_18 = ROTL3(S2_18 + S2_17 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_18 + Lhi1, S1_18 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_18 + Lhi2, S2_18 + Lhi2);
    S1_19 = ROTL3(S1_19 + S1_18 + Llo1);
    S2_19 = ROTL3(S2_19 + S2_18 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_19 + Llo1, S1_19 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_19 + Llo2, S2_19 + Llo2);
    S1_20 = ROTL3(S1_20 + S1_19 + Lhi1);
    S2_20 = ROTL3(S2_20 + S2_19 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_20 + Lhi1, S1_20 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_20 + Lhi2, S2_20 + Lhi2);
    S1_21 = ROTL3(S1_21 + S1_20 + Llo1);
    S2_21 = ROTL3(S2_21 + S2_20 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_21 + Llo1, S1_21 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_21 + Llo2, S2_21 + Llo2);
    S1_22 = ROTL3(S1_22 + S1_21 + Lhi1);
    S2_22 = ROTL3(S2_22 + S2_21 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_22 + Lhi1, S1_22 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_22 + Lhi2, S2_22 + Lhi2);
    S1_23 = ROTL3(S1_23 + S1_22 + Llo1);
    S2_23 = ROTL3(S2_23 + S2_22 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_23 + Llo1, S1_23 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_23 + Llo2, S2_23 + Llo2);
    S1_24 = ROTL3(S1_24 + S1_23 + Lhi1);
    S2_24 = ROTL3(S2_24 + S2_23 + Lhi2);
    Llo1 = ROTL(Llo1 + S1_24 + Lhi1, S1_24 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_24 + Lhi2, S2_24 + Lhi2);
    S1_25 = ROTL3(S1_25 + S1_24 + Llo1);
    S2_25 = ROTL3(S2_25 + S2_24 + Llo2);
    Lhi1 = ROTL(Lhi1 + S1_25 + Llo1, S1_25 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_25 + Llo2, S2_25 + Llo2);

        /* Begin round 3 of key expansion */
        /* mixed with encryption */

    Elo1 = PlainLo               + (S1_00 = ROTL3(S1_00 + S1_25 + Lhi1));
    Elo2 = PlainLo               + (S2_00 = ROTL3(S2_00 + S2_25 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = PlainHi               + (S1_01 = ROTL3(S1_01 + S1_00 + Llo1));
    Ehi2 = PlainHi               + (S2_01 = ROTL3(S2_01 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_02 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_02 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_03 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_03 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_04 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_04 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_05 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_05 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_06 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_06 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_07 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_07 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_08 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_08 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_09 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_09 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_10 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_10 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_11 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_11 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_12 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_12 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_13 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_13 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_14 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_14 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_15 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_15 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_16 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_16 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_17 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_17 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_18 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_18 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_19 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_19 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_20 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_20 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_21 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_21 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_22 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_22 + S2_01 + Lhi2));
    Llo1 = ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1);
    Llo2 = ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2);
    Ehi1 = ROTL(Ehi1 ^ Elo1, Elo1) + (S1_01 = ROTL3(S1_23 + S1_00 + Llo1));
    Ehi2 = ROTL(Ehi2 ^ Elo2, Elo2) + (S2_01 = ROTL3(S2_23 + S2_00 + Llo2));
    Lhi1 = ROTL(Lhi1 + S1_01 + Llo1, S1_01 + Llo1);
    Lhi2 = ROTL(Lhi2 + S2_01 + Llo2, S2_01 + Llo2);
    Elo1 = ROTL(Elo1 ^ Ehi1, Ehi1) + (S1_00 = ROTL3(S1_24 + S1_01 + Lhi1));
    Elo2 = ROTL(Elo2 ^ Ehi2, Ehi2) + (S2_00 = ROTL3(S2_24 + S2_01 + Lhi2));

    if (CypherLo == Elo1)
    {
        if (CypherHi == (ROTL(Ehi1 ^ Elo1, Elo1) + ROTL3(S1_25 + S1_00 + ROTL(Llo1 + S1_00 + Lhi1, S1_00 + Lhi1))))
            return 1;
    }
    if (CypherLo == Elo2)
    {
        if (CypherHi == (ROTL(Ehi2 ^ Elo2, Elo2) + ROTL3(S2_25 + S2_00 + ROTL(Llo2 + S2_00 + Lhi2, S2_00 + Lhi2))))
        return 2;
    }
    return (0);
}


