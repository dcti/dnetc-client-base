/* s1_1n.h v3.0 */
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

/* $Log: s1_1n.h,v $
/* Revision 1.1.1.1  1998/06/14 14:23:49  remi
/* Initial integration.
/* */

#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_SLICE T1, T2, T3, T4, T5, T6, T7, T8;
    register INNER_LOOP_SLICE T9, T10, T11, T12, T13, T14, T15;

/* FREE */
#define x34	a4	/* GROSS, but ran out of registers */
#define x35	a4

/* FREE */
#define x30	T15
#define x31	T15
#define x40	T15

/* FREE */
#define Preload4_1 T14
/* FREE */
#define x43	T14
#define x44	T14
#define x45	T14
#define x46	T14

/* FREE */
#define x16	T13
#define x17	T13
#define x18	T13
/* FREE */
#define Preload1_1 T13

#define x4	T12

#define key1	T11
/* FREE but hold */
#define k1	T11
/* FREE */
#define x9	T11

#define key2	T10
/* FREE but hold */
#define k2	T10
/* FREE */
#define x28	T10
#define x36	T10
#define x49	T10
#define x50	T10

#define x2	T9
#define x25	T9
#define x29	T9
/* FREE */
#define x37	T9
/* FREE */
#define t_a3	T9

/* FREE */
#define x7	T8
#define x10	T8
#define x11	T8
#define x12	T8
/* FREE */
#define t_key3	T8
/* FREE but hold */
#define t_k3	T8

#define x13	T7
#define x14	T7
/* FREE */
#define x32	T7
#define x33	T7
#define x47	T7
#define x48	T7

#define x1	T6
#define x23	T6
#define x24	T6
#define x41	T6
#define x42	T6
#define x51	T6
#define x52	T6
#define x53	T6
#define x54	T6
#define x55	T6
#define x56	T6

#define x5	T5
/* FREE */
#define Preload3_1 T5

#define x3	T4
#define x38	T4
#define x39	T4
/* FREE */
#define t_a5	T4

#define x15	T3
/* FREE */
#define x19	T3
#define x20	T3
#define x21	T3
#define x22	T3
/* FREE */
#define t_key5	T3
/* FREE but hold */
#define t_k5	T3

#define x6	T2
/* FREE */
#define Preload2_1 T2
/* FREE */
#define t_key4	T2
/* FREE but hold */
#define t_k4	T2

#define x8	T1
/* FREE */
#define x26	T1
#define x27	T1
/* FREE */
#define t_key6	T1
/* FREE but hold */
#define t_k6	T1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_SLICE x1, x2, x3, x4, x5, x6, x7, x8;
    INNER_LOOP_SLICE x9, x10, x11, x12, x13, x14, x15, x16;
    INNER_LOOP_SLICE x17, x18, x19, x20, x21, x22, x23, x24;
    INNER_LOOP_SLICE x25, x26, x27, x28, x29, x30, x31, x32;
    INNER_LOOP_SLICE x33, x34, x35, x36, x37, x38, x39, x40;
    INNER_LOOP_SLICE x41, x42, x43, x44, x45, x46, x47, x48;
    INNER_LOOP_SLICE x49, x50, x51, x52, x53, x54, x55, x56;
    INNER_LOOP_SLICE Preload1_1, Preload2_1, Preload3_1, Preload4_1;

    INNER_LOOP_SLICE k1, k2;
    INNER_LOOP_SLICE t_a3, t_a5;
    INNER_LOOP_SLICE t_k3, t_k4, t_k5, t_k6;

    INNER_LOOP_SLICE *key1, *key2;
    INNER_LOOP_SLICE *t_key3, *t_key4, *t_key5, *t_key6;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a3, a5, a4, a6 valid */

	ASM_A_LOAD (key2, offsets->Key_Ptrs[OFFSET1 + 2]);

	ASM_AND_NOT (x1, a3, a5);		/*               */
	ASM_AND_NOT (x3, a3, a4);		/*               */
	ASM_D_LOAD (a2, in[S12]);

	ASM_OR (x4, x3, a5);			/*               */
	ASM_XOR (x8, a3, a4);			/*               O */
	ASM_D_LOAD (k2, ((INNER_LOOP_SLICE *)key2)[0]);	

	ASM_XOR (x2, x1, a4);			/*               */
	ASM_AND (x5, a6, x4);			/*               */
	ASM_A_LOAD (key1, offsets->Key_Ptrs[OFFSET1 + 1]);

	ASM_XOR (x6, x2, x5);			/*               */
	ASM_XOR (x13, a5, x5);			/*               O */
	ASM_D_LOAD (a1, in[S11]);

	ASM_AND (x14, x13, x8);			/*               O */
	ASM_AND_NOT (x15, a5, a4);		/*              N  */
	ASM_D_LOAD (k1, ((INNER_LOOP_SLICE *)key1)[0]);

/**/	ASM_AND_NOT (x7, a4, a5);		/*               */
	ASM_XOR (x16, x3, x14);			/*              N  */
/* FREE MEMORY REF */

	ASM_XOR (a2, a2, k2);
	ASM_OR (x28, x6, x7);			/*             M  */
/* FREE MEMORY REF */

	ASM_XOR (a1, a1, k1);
	ASM_OR (x17, a6, x16);			/*              N  */
/* FREE MEMORY REF */

	ASM_AND_NOT (x9, a6, x8);		/*               O */
	ASM_XOR (x18, x15, x17);		/*              N  */
/* FREE MEMORY REF */

	ASM_XOR (x10, x7, x9);			/*               O */
	ASM_OR (x19, a2, x18);			/*              N  */
    ASM_A_LOAD (in, offsets->Next_Source);		/* PIPELINE */

	ASM_OR (x11, a2, x10);			/*               O */
	ASM_XOR (x20, x14, x19);		/*              NO */
    ASM_A_LOAD (offsets, offsets->Next_Offsets);	/* PIPELINE */

	ASM_XOR (x12, x6, x11);			/*               O */
	ASM_AND (x21, a1, x20);			/*              N  */
	ASM_D_LOAD (Preload2_1, merge[D12]);	/*                 */

	ASM_XOR_NOT (x22, x12, x21);		/*              N  */
	ASM_OR (x23, x1, x5);			/*            L   */
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET1 + 3]);	/* PIPELINE */

	ASM_XOR (Preload2_1, Preload2_1, x22);	/*              N  */
	ASM_D_STORE (out[D12], Preload2_1);	/*              N  */
	ASM_AND_NOT (x25, x18, x2);		/*            L   */

	ASM_XOR (x24, x23, x8);			/*            L   */
	ASM_AND_NOT (x26, a2, x25);		/*            L   */
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);		/* PIPELINE */

	ASM_XOR (x27, x24, x26);		/*            L   */
	ASM_XOR (x30, x9, x24);			/*           K    */
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET1 + 5]);	/* PIPELINE */

	ASM_AND_NOT (x31, x18, x30);		/*           K    */
	ASM_XOR (x29, x28, x25);		/*             M  */
    ASM_A_LOAD (t_key4, offsets->Key_Ptrs[OFFSET1 + 4]);	/* PIPELINE */

	ASM_AND (x32, a2, x31);			/*           K    */
	ASM_AND (x36, a3, x28);			/*        H       */
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);		/* PIPELINE */

	ASM_XOR (x33, x29, x32);		/*           K M  */
	ASM_OR (x40, a3, x31);			/*         I      */
    ASM_D_LOAD (t_k4, ((INNER_LOOP_SLICE *)t_key4)[0]);		/* PIPELINE *//* FREE MEMORY REF */

/**/	ASM_AND (x34, a1, x33);			/*           K    */
	ASM_AND_NOT (x37, x18, x36);		/*        H       */
	ASM_D_LOAD (Preload4_1, merge[D14]);	/*                */

	ASM_XOR (x35, x27, x34);		/*           KL   */
	ASM_AND_NOT (x41, x24, x37);		/*       G  J     */
    ASM_A_LOAD (t_key6, offsets->Key_Ptrs[OFFSET1 + 6]);	/* PIPELINE */

	ASM_XOR (Preload4_1, Preload4_1, x35);	/*           K    */
	ASM_D_STORE (out[D14], Preload4_1);	/*           K    */
	ASM_OR (x42, x41, x3);			/*       G        */

	ASM_AND_NOT (x43, x42, a2);		/*       G        */
	ASM_OR (x38, a2, x3);			/*        H       */
    ASM_D_LOAD (t_k6, ((INNER_LOOP_SLICE *)t_key6)[0]);		/* PIPELINE */

	ASM_XOR (x44, x40, x43);		/*       G I      */
	ASM_XOR (x39, x37, x38);		/*        H       */
    ASM_D_LOAD (t_a3, in[S13]);				/* PIPELINE */

	ASM_OR (x51, x42, x18);			/* A  D           */
	ASM_AND_NOT (x45, a1, x44);		/*       G        */
	ASM_D_LOAD (Preload1_1, merge[D11]);	/*                */

	ASM_XOR_NOT (x46, x39, x45);		/*       GH       */
	ASM_XOR (x52, x51, a5);			/* A              */
    ASM_D_LOAD (a6, in[S16]);				/* PIPELINE */

	ASM_XOR (Preload1_1, Preload1_1, x46);	/*       G        */
	ASM_D_STORE (out[D11], Preload1_1);	/*       G        */
	ASM_XOR (x49, x4, x36);			/*   C E          */

	ASM_AND_NOT (x50, x49, x5);		/*   C            */
	ASM_AND_NOT (x53, a2, x52);		/* A              */
	ASM_D_LOAD (Preload3_1, merge[D13]);	/*                */

	ASM_AND_NOT (x47, x33, x9);		/*  B   F         */
	ASM_XOR (x54, x50, x53);		/* A C            */
    ASM_D_LOAD (a4, in[S14]);				/* PIPELINE */

	ASM_XOR (x48, x47, x39);		/*  B     H       */
	ASM_OR (x55, a1, x54);			/* A              */
    ASM_D_LOAD (t_a5, in[S15]);				/* PIPELINE */

    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */
	ASM_XOR_NOT (x56, x48, x55);		/* AB             */
    ASM_A_LOAD (merge, offsets->Merge);			/* PIPELINE */

	ASM_XOR (Preload3_1, Preload3_1, x56);	/* A              */
	ASM_D_STORE (out[D13], Preload3_1);	/* A              */
    ASM_XOR (a5, t_a5, t_k5);				/* PIPELINE */

    ASM_XOR (a4, a4, t_k4);				/* PIPELINE */
    ASM_XOR (a6, a6, t_k6);				/* PIPELINE */

    ASM_COMMENT_END_INCLUDE (end_of_s1_1n);

#include "s0.h"
/* end of s1_1n.h */
