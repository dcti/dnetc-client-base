/* s6_6n.h v3.0 */
/* TOADS not done */
/* does function, links to next function */
/* CRITICAL! This is storing results from one iteration, and fetching
 * operands for the next iteration.  Therefore, there is the possibility
 * that this will fetch a result BEFORE it is stored.  This must be
 * avoided.  The S variables must be compared to the D variables,
 * and where an S fetches from under a D, the load must be later.
 *
 * For example, function 1 writes with D11 as 8.  Function 2 reads
 * from S26 as 8!  So the load should come long after the store.
 *
 * This code specifically does NOT support back-to-back updates to the
 * same data area.  It can only be used when applied to separate data items.
 */

/* $Log: s6_6n.h,v $
/* Revision 1.1  1998/06/14 14:23:50  remi
/* Initial revision
/* */


#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_SLICE T1, T2, T3, T4, T5, T6, T7, T8;
    register INNER_LOOP_SLICE T9, T10, T11, T12, T13, T14;

/* FREE */
#define x36	a1	/* GROSS, but ran out of registers */
#define x37	a1

/* FREE */
#define x28	T14
#define x29	T14
#define x30	T14
#define x31	T14

/* FREE */
#define x38	T13
#define x39	T13
#define x40	T13
/* FREE */
#define x41	T13

/* FREE */
#define x16	T12
#define x17	T12
#define x18	T12
#define x19	T12
#define x20	T12
#define x21	T12
/* FREE */
#define Preload4_6 T12
/* FREE */
#define Preload3_6 T12

/* FREE */
#define x13	T11
#define x14	T11
/* FREE */
#define Preload1_6 T11
/* FREE */
#define x22	T11
#define x23	T11

/* FREE */
#define x5	T10

#define key2	T9
/* FREE but hold */
#define k2	T9
/* FREE */
#define x11	T9
#define x12	T9
/* FREE */
#define x26	T9
/* FREE */
#define t_a5	T9

/* FREE */
#define key3	T8
/* FREE but hold */
#define k3	T8
/* FREE */
#define t_key5	T8
/* FREE but hold */
#define t_k5	T8

#define x7	T7
#define x42	T7
#define x43	T7
#define x44	T7
#define x45	T7

#define x3	T6
#define x32	T6
#define x33	T6
#define x48	T6
#define x49	T6
#define x50	T6
#define x51	T6
#define x52	T6
#define x53	T6

#define x1	T5
#define x27	T5
/* FREE */
#define x34	T5
#define x35	T5
#define x46	T5
#define x47	T5
/* FREE */
#define Preload2_6 T5

#define x2	T4
/* FREE */
#define t_a1	T4

#define x4	T3
#define x15	T3
/* FREE */
#define t_key1	T3
/* FREE but hold */
#define t_k1	T3

/* FREE */
#define x6	T2
/* FREE */
#define t_key6	T2
/* FREE but hold */
#define t_k6	T2

/* FREE */
#define x8	T1
#define x9	T1
#define x10	T1
/* FREE */
#define x24	T1
#define x25	T1
/* FREE */
#define t_key4	T1
/* FREE but hold */
#define t_k4	T1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_SLICE x1, x2, x3, x4, x5, x6, x7, x8;
    INNER_LOOP_SLICE x9, x10, x11, x12, x13, x14, x15, x16;
    INNER_LOOP_SLICE x17, x18, x19, x20, x21, x22, x23, x24;
    INNER_LOOP_SLICE x25, x26, x27, x28, x29, x30, x31, x32;
    INNER_LOOP_SLICE x33, x34, x35, x36, x37, x38, x39, x40;
    INNER_LOOP_SLICE x41, x42, x43, x44, x45, x46, x47, x48;
    INNER_LOOP_SLICE x49, x50, x51, x52, x53;
    INNER_LOOP_SLICE Preload1_6, Preload2_6, Preload3_6, Preload4_6;

    INNER_LOOP_SLICE k2, k3;
    INNER_LOOP_SLICE t_a1, t_a5;
    INNER_LOOP_SLICE t_k1, t_k4, t_k5, t_k6;

    INNER_LOOP_SLICE *key2, *key3;
    INNER_LOOP_SLICE *t_key1, *t_key4, *t_key5, *t_key6;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a5, a1, a6, a4 valid */

	ASM_A_LOAD (key2, offsets->Key_Ptrs[OFFSET6 + 2]);

	ASM_XOR (x1, a5, a1);			/*              N */
/**/	ASM_AND (x3, a1, a6);			/*          J     */
	ASM_D_LOAD (a2, in[S62]);

	ASM_AND_NOT (x4, x3, a5);		/*              N */
	ASM_XOR (x7, a6, x3);			/*          J     */
	ASM_D_LOAD (k2, ((INNER_LOOP_SLICE *)key2)[0]);

	ASM_XOR (x2, x1, a6);			/*              N */
	ASM_OR (x8, x4, x7);			/*          J     */
	ASM_A_LOAD (key3, offsets->Key_Ptrs[OFFSET6 + 3]);

	ASM_AND_NOT (x5, a4, x4);		/*              N */
	ASM_AND_NOT (x9, x8, a4);		/*          J     */
	ASM_D_LOAD (a3, in[S63]);

	ASM_XOR (x6, x2, x5);			/*              N */
	ASM_XOR (x10, x7, x9);			/*          J     */
	ASM_D_LOAD (k3, ((INNER_LOOP_SLICE *)key3)[0]);	

	ASM_OR (x13, a6, x6);			/*             M  */
	ASM_OR (x15, x4, x10);			/*          J     */
/* FREE MEMORY REF */

	ASM_XOR (a2, a2, k2);
	ASM_XOR (a3, a3, k3);
/* FREE MEMORY REF */

	ASM_AND_NOT (x14, x13, a5);		/*             M  */
	ASM_AND_NOT (x16, a2, x15);		/*          J     */
    ASM_A_LOAD (in, offsets->Next_Source);		/* PIPELINE */

	ASM_AND (x11, a2, x10);			/*              N */
	ASM_XOR (x17, x14, x16);		/*          J  M  */
    ASM_A_LOAD (offsets, offsets->Next_Offsets);	/* PIPELINE */

	ASM_XOR (x12, x6, x11);			/*             M  */
	ASM_AND_NOT (x18, x17, a3);		/*          J     */
	ASM_D_LOAD (Preload1_6, merge[D61]);	/*                */

	ASM_XOR_NOT (x19, x12, x18);		/*          J   N */
	ASM_OR (x26, a5, a6);			/*    D           */
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET6 + 5]);	/* PIPELINE */

	ASM_XOR (Preload1_6, Preload1_6, x19);	/*          J     */
	ASM_D_STORE (out[D61], Preload1_6);	/*          J     */
	ASM_AND_NOT (x20, x19, x1);		/*          J     */

	ASM_XOR (x21, x20, x15);		/*          J     */
	ASM_AND_NOT (x27, x26, x1);		/*    D           */
/* FREE MEMORY REF */

	ASM_AND_NOT (x22, a6, x21);		/*          J     */
	ASM_XOR (x32, x3, x6);			/*     E          */
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);		/* PIPELINE */

	ASM_XOR (x23, x22, x6);			/*          J     */
	ASM_AND_NOT (x33, x32, x10);		/*     E          */
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET6 + 1]);	/* PIPELINE */

	ASM_AND_NOT (x24, a2, x23);		/*          J     */
	ASM_AND_NOT (x38, x21, a5);		/*        H J     */
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET6 + 6]);	/* PIPELINE */

	ASM_AND_NOT (x28, a2, x24);		/*          J L   */
	ASM_XOR (x25, x21, x24);		/*          J     */
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);		/* PIPELINE */

	ASM_XOR (x29, x27, x28);		/*    D       L   */
	ASM_XOR (x34, a6, x25);			/*         I      */
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);		/* PIPELINE */

	ASM_AND_NOT (x35, a5, x34);		/*         I      */
	ASM_AND_NOT (x30, a3, x29);		/*            L   */
	ASM_D_LOAD (Preload4_6, merge[D64]);	/*                */

	ASM_XOR_NOT (x31, x25, x30);		/*          J L   */
/**/	ASM_AND_NOT (x36, a2, x35);		/*         I      */
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET6 + 4]);	/* PIPELINE */

	ASM_XOR (Preload4_6, Preload4_6, x31);	/*            L   */
	ASM_D_STORE (out[D64], Preload4_6);	/*            L   */
	ASM_XOR (x37, x33, x36);		/*     E   I      */

	ASM_OR (x39, a3, x38);			/*        H       */
	ASM_AND (x42, a5, x7);			/*  B             */
	ASM_D_LOAD (Preload3_6, merge[D63]);	/*                */

	ASM_XOR_NOT (x40, x37, x39);		/*        HI      */
	ASM_AND_NOT (x43, a4, x42);		/*  B             */
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]);		/* PIPELINE */

	ASM_XOR (Preload3_6, Preload3_6, x40);	/*        H       */
	ASM_D_STORE (out[D63], Preload3_6);	/*        H       */
	ASM_OR (x41, x35, x2);			/*       G        */

	ASM_OR (x44, a2, x43);			/*  B             */
	ASM_AND (x48, x26, x33);		/* A  DE          */
    ASM_D_LOAD (t_a5, in[S65]);				/* PIPELINE */

	ASM_XOR (x45, x41, x44);		/*  B    G        */
	ASM_XOR (x49, x48, x2);			/* A              */
    ASM_D_LOAD (t_a1, in[S61]);				/* PIPELINE */

	ASM_OR (x46, x23, x35);			/*   C   G  J     */
	ASM_AND (x50, a2, x49);			/* A              */
    ASM_D_LOAD (a6, in[S66]);				/* PIPELINE */

/* FREE ALU OP */
	ASM_XOR (x47, x46, x5);			/*   C            */
/* FREE MEMORY REF */

/* FREE ALU OP */
	ASM_XOR (x51, x47, x50);		/* A C            */
    ASM_D_LOAD (a4, in[S64]);				/* PIPELINE */

/* FREE ALU OP */
	ASM_AND_NOT (x52, a3, x51);		/* A              */
	ASM_D_LOAD (Preload2_6, merge[D62]);	/*                */

    ASM_XOR (a5, t_a5, t_k5);				/* PIPELINE */
	ASM_XOR_NOT (x53, x45, x52);		/* AB             */
ASM_A_LOAD (merge, offsets->Merge);			/* PIPELINE */

	ASM_XOR (Preload2_6, Preload2_6, x53);	/* A              */
	ASM_D_STORE (out[D62], Preload2_6);	/* A              */
    ASM_XOR (a1, t_a1, t_k1);				/* PIPELINE */

    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */
    ASM_XOR (a4, a4, t_k4);				/* PIPELINE */

    ASM_COMMENT_END_INCLUDE (end_of_s6_6n);

#include "s0.h"
/* end of s6_6n.h */
