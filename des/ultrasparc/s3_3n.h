/* s3_3n.h v3.0 */
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

/*
 * $Log: s3_3n.h,v $
 * Revision 1.2  1998/06/14 15:19:01  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:50  remi
 * Initial integration.
 *
 */


#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_SLICE T1, T2, T3, T4, T5, T6, T7, T8;
    register INNER_LOOP_SLICE T9, T10, T11, T12, T13, T14, T15;

/* FREE */
#define x27	a5	/* GROSS, but ran out of registers */
#define x28	a5
#define x29	a5
#define x30	a5

#define x4	T15
#define x24	T15
/* FREE */
#define Preload3_3 T15

#define x5	T14
/* FREE */
#define Preload1_3 T14

#define x10	T13

#define x31	T12
#define x32	T12
/* FREE */
#define Preload2_3 T12

#define x12	T11
#define x13	T11
#define x14	T11
#define x15	T11
/* FREE */
#define x23	T11

/* FREE */
#define Preload4_3 T10
/* FREE */
#define x18	T10

/* FREE */
#define x11	T9
#define x25	T9
#define x26	T9
#define x36	T9
/* FREE */
#define t_a2	T9

/* FREE */
#define t_key2	T8
/* FREE but hold */
#define t_k2	T8

/* FREE */
#define x45	T7
#define x46	T7
#define x47	T7
#define x48	T7

/* FREE */
#define x7	T6
#define x33	T6
#define x34	T6
#define x35	T6
/* FREE */
#define t_a3	T6

#define key1	T5
/* FREE but hold */
#define k1	T5
/* FREE */
#define x6	T5
/* FREE */
#define t_key3	T5
/* FREE but hold */
#define t_k3	T5

#define x16	T4
#define x17	T4
#define x37	T4
#define x38	T4
#define x39	T4
#define x40	T4
#define x41	T4
#define x42	T4
#define x49	T4
#define x50	T4
#define x51	T4
#define x52	T4
#define x53	T4

#define x1	T3
#define x2	T3
#define x19	T3
#define x43	T3
#define x44	T3

#define x3	T2
/* FREE */
#define t_key6	T2
/* FREE but hold */
#define t_k6	T2

#define key4	T1
/* FREE but hold */
#define k4	T1
/* FREE */
#define x8	T1
#define x9	T1
/* FREE */
#define x20	T1
#define x21	T1
#define x22	T1
/* FREE */
#define t_key5	T1
/* FREE but hold */
#define t_k5	T1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_SLICE x1, x2, x3, x4, x5, x6, x7, x8;
    INNER_LOOP_SLICE x9, x10, x11, x12, x13, x14, x15, x16;
    INNER_LOOP_SLICE x17, x18, x19, x20, x21, x22, x23, x24;
    INNER_LOOP_SLICE x25, x26, x27, x28, x29, x30, x31, x32;
    INNER_LOOP_SLICE x33, x34, x35, x36, x37, x38, x39, x40;
    INNER_LOOP_SLICE x41, x42, x43, x44, x45, x46, x47, x48;
    INNER_LOOP_SLICE x49, x50, x51, x52, x53;
    INNER_LOOP_SLICE Preload1_3, Preload2_3, Preload3_3, Preload4_3;

    INNER_LOOP_SLICE k1, k4;
    INNER_LOOP_SLICE t_a2, t_a3;
    INNER_LOOP_SLICE t_k2, t_k3, t_k5, t_k6;

    INNER_LOOP_SLICE *t_key2, *t_key3, *t_key5, *t_key6;
    INNER_LOOP_SLICE *key1, *key4;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a2, a3, a6, a5 valid */

	ASM_A_LOAD (key1, offsets->Key_Ptrs[OFFSET3 + 1]);

	ASM_XOR (x1, a2, a3);			/*                P */
	ASM_AND (x16, a3, a6);			/*             M  */
	ASM_D_LOAD (a1, in[S31]);

	ASM_XOR (x2, x1, a6);			/*                P */
	ASM_AND (x31, a3, a5);			/*         I      */
	ASM_D_LOAD (k1, ((INNER_LOOP_SLICE *)key1)[0]);	

	ASM_AND (x3, a2, x2);			/*                P */
	ASM_XOR (x32, x31, x2);			/*         I      */
	ASM_A_LOAD (key4, offsets->Key_Ptrs[OFFSET3 + 4]);

	ASM_OR (x4, a5, x3);			/*                P */
	ASM_AND_NOT (x10, a6, x3);		/*              N */
	ASM_D_LOAD (a4, in[S34]);

	ASM_XOR (x5, x2, x4);			/*                P */
	ASM_XOR (x11, x10, a5);			/*              N */
	ASM_D_LOAD (k4, ((INNER_LOOP_SLICE *)key4)[0]);	

	ASM_XOR (a1, a1, k1);
	ASM_XOR (x6, a3, x3);			/*               O */
/* FREE MEMORY REF */

	ASM_AND (x12, a1, x11);			/*              N */
	ASM_XOR (a4, a4, k4);
/* FREE MEMORY REF */

	ASM_AND_NOT (x7, x6, a5);		/*               O */
	ASM_XOR (x13, a5, x12);			/*              N */
    ASM_A_LOAD (in, offsets->Next_Source);		/* PIPELINE */

	ASM_OR (x8, a1, x7);			/*               O */
	ASM_OR (x14, a4, x13);			/*              N */
    ASM_A_LOAD (offsets, offsets->Next_Offsets);	/* PIPELINE */

	ASM_AND_NOT (x19, x2, x7);		/*           K    */
	ASM_XOR (x9, x5, x8);			/*               O */
	ASM_D_LOAD (Preload4_3, merge[D34]);	/*                */

	ASM_XOR (x15, x9, x14);			/*              N */
	ASM_XOR (x20, x19, x16);		/*           K    */
/* FREE MEMORY REF */

	ASM_OR (x25, x11, x19);			/*          J     */
	ASM_AND (x45, a6, x15);			/*  B  E          */
    ASM_A_LOAD (t_key2, offsets->Key_Ptrs[OFFSET3 + 2]);	/* PIPELINE */

	ASM_XOR (Preload4_3, Preload4_3, x15);	/*              N */
	ASM_D_STORE (out[D34], Preload4_3);	/*              N */
	ASM_XOR (x46, x45, x6);			/*  B             */

	ASM_OR (x17, x16, x3);			/*             M  */
	ASM_OR (x21, a1, x20);			/*           K    */
    ASM_D_LOAD (t_k2, ((INNER_LOOP_SLICE *)t_key2)[0]);		/* PIPELINE */

	ASM_XOR (x26, x25, x17);		/*          J  M  */
/**/	ASM_XOR (x18, x17, a5);			/*             M  */
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET3 + 3]);	/* PIPELINE */

/**/	ASM_OR (x27, a1, x26);			/*          J     */
	ASM_OR (x23, a2, x7);			/*            L   */
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET3 + 6]);	/* PIPELINE */

	ASM_XOR (x22, x18, x21);		/*           K M  */
	ASM_XOR (x24, x23, x4);			/*            L   */
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);		/* PIPELINE */

/* FREE ALU OP */
	ASM_XOR (x28, x24, x27);		/*          J L   */
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);		/* PIPELINE */

	ASM_AND_NOT (x29, a4, x28);		/*          J     */
	ASM_AND_NOT (x33, x7, a3);		/*      F         */
	ASM_D_LOAD (Preload3_3, merge[D33]);	/*                */

	ASM_XOR_NOT (x30, x22, x29);		/*          JK    */
	ASM_XOR (x37, a6, x17);			/*   C    H       */
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET3 + 5]);	/* PIPELINE */

	ASM_XOR (Preload3_3, Preload3_3, x30);	/*          J     */
	ASM_D_STORE (out[D33], Preload3_3);	/*          J     */
	ASM_AND_NOT (x38, x37, x5);		/*   C            */

	ASM_OR (x36, x10, x26);			/*       G        */
	ASM_AND (x39, a1, x38);			/*   C            */
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);		/* PIPELINE */

	ASM_OR (x34, a1, x33);			/*      F         */
	ASM_XOR (x40, x36, x39);		/*   C   G        */
    ASM_D_LOAD (t_a2, in[S32]);				/* PIPELINE */

	ASM_XOR (x35, x32, x34);		/*      F  I      */
	ASM_AND (x41, a4, x40);			/*   C            */
	ASM_D_LOAD (Preload2_3, merge[D32]);	/*                */

	ASM_OR (x43, a2, x19);			/*    D      K    */
	ASM_XOR (x42, x35, x41);		/*   C  F         */
    ASM_D_LOAD (t_a3, in[S33]);				/* PIPELINE */

	ASM_XOR (Preload2_3, Preload2_3, x42);	/*   C            */
	ASM_D_STORE (out[D32], Preload2_3);	/*   C            */
	ASM_AND_NOT (x49, x42, x23);		/* A C            */

	ASM_AND_NOT (x47, x46, a1);		/*  B             */
	ASM_OR (x50, a1, x49);			/* A              */
    ASM_D_LOAD (a6, in[S36]);				/* PIPELINE */

	ASM_XOR (x44, x43, x18);		/*    D           */
	ASM_XOR (x51, x47, x50);		/* AB             */
    ASM_D_LOAD (a5, in[S35]);				/* PIPELINE */

	ASM_XOR (x48, x44, x47);		/*  B D           */
	ASM_AND (x52, a4, x51);			/* A              */
	ASM_D_LOAD (Preload1_3, merge[D31]);	/*                */

    ASM_XOR (a2, t_a2, t_k2);				/* PIPELINE */
	ASM_XOR_NOT (x53, x48, x52);		/* AB             */
    ASM_A_LOAD (merge, offsets->Merge);			/* PIPELINE */

	ASM_XOR (Preload1_3, Preload1_3, x53);	/* A              */
	ASM_D_STORE (out[D31], Preload1_3);	/* A              */
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */

    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */
    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */

    ASM_COMMENT_END_INCLUDE (end_of_s3_3);

#include "s0.h"
/* end of s3_3n.h */
