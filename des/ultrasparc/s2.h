/* s2.h v3.0 */

/*
 * $Log: s2.h,v $
 * Revision 1.2  1998/06/14 15:18:55  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:49  remi
 * Initial integration.
 *
 */


#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_SLICE T1, T2, T3, T4, T5, T6, T7, T8;
    register INNER_LOOP_SLICE T9, T10, T11, T12, T13, T14, T15;

#define x23	a1	/* GROSS, but ran out of registers */

/* FREE */
#define x30	T15
#define x31	T15

/* FREE */
#define x6	T14
#define x26	T14
#define x27	T14
#define x34	T14
#define x35	T14
#define x36	T14

/* FREE */
#define x13	T13
#define x14	T13
#define x15	T13
#define x16	T13
#define x17	T13
/* FREE */
#define x24	T13
#define x37	T13
#define x38	T13
#define x39	T13
#define x40	T13

/* FREE */
#define x9	T12
#define x10	T12
/* FREE */
#define Preload3_2 T12
/* FREE */
#define Preload4_2 T12

#define x28	T11
#define x29	T11

/* FREE */
#define x11	T10
#define x19	T10
#define x20	T10
#define x46	T10

#define x12	T9
/* FREE */
#define t_a1	T9

#define x1	T8
/* FREE */
#define t_key1	T8
/* FREE but hold */
#define t_k1	T8

#define x2	T7
#define x42	T7
#define x43	T7
#define x44	T7
#define x45	T7

#define key3	T6
/* FREE but hold */
#define k3	T6
/* FREE */
#define x18	T6
#define x32	T6
#define x33	T6
/* FREE */
#define t_a6	T6

/* FREE */
#define key4	T5
/* FREE but hold */
#define k4	T5
/* FREE */
#define Preload1_2 T5
/* FREE */
#define t_key6	T5
/* FREE but hold */
#define t_k6	T5

#define x3	T4
#define x47	T4
#define x48	T4
#define x49	T4
#define x50	T4

#define x41	T3
/* FREE */
#define Preload2_2 T3

#define x4	T2
/* FREE */
#define x21	T2
#define x22	T2
/* FREE */
#define t_key5	T2
/* FREE but hold */
#define t_k5	T2

/* FREE */
#define x5	T1
#define x7	T1
#define x8	T1
#define x25	T1
/* FREE */
#define t_key2	T1
/* FREE but hold */
#define t_k2	T1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_SLICE x1, x2, x3, x4, x5, x6, x7, x8;
    INNER_LOOP_SLICE x9, x10, x11, x12, x13, x14, x15, x16;
    INNER_LOOP_SLICE x17, x18, x19, x20, x21, x22, x23, x24;
    INNER_LOOP_SLICE x25, x26, x27, x28, x29, x30, x31, x32;
    INNER_LOOP_SLICE x33, x34, x35, x36, x37, x38, x39, x40;
    INNER_LOOP_SLICE x41, x42, x43, x44, x45, x46, x47, x48;
    INNER_LOOP_SLICE x49, x50;
    INNER_LOOP_SLICE Preload1_2, Preload2_2, Preload3_2, Preload4_2;

    INNER_LOOP_SLICE *key3, *key4;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a1, a6, a5, a2 valid */

	ASM_A_LOAD (key3, offsets->Key_Ptrs[OFFSET2 + 3]);

	ASM_XOR (x1, a1, a6);			/*               */
	ASM_AND (x3, a6, a5);			/*    D           */
	ASM_D_LOAD (a3, in[S23]);

	ASM_XOR (x2, x1, a5);			/*               */
	ASM_AND_NOT (x4, a1, x3);		/*    D           */
	ASM_D_LOAD (k3, ((INNER_LOOP_SLICE *)key3)[0]);

	ASM_AND_NOT (x5, a2, x4);		/*    D           */
	ASM_AND_NOT (x11, a5, x4);		/*               */
	ASM_A_LOAD (key4, offsets->Key_Ptrs[OFFSET2 + 4]);

	ASM_XOR (x6, x2, x5);			/*               */
	ASM_OR (x7, x3, x5);			/*    D           */
	ASM_D_LOAD (a4, in[S24]);

	ASM_AND_NOT (x8, x7, x1);		/*    D           */
	ASM_OR (x12, x11, a2);			/*               */
	ASM_D_LOAD (k4, ((INNER_LOOP_SLICE *)key4)[0]);

	ASM_XOR (a3, a3, k3);
	ASM_AND_NOT (x18, a6, x4);		/*              N */
/* FREE MEMORY REF */

	ASM_OR (x9, a3, x8);			/*    D           */
	ASM_XOR (a4, a4, k4);
/* FREE MEMORY REF */

	ASM_XOR (x10, x6, x9);			/*    D           */
	ASM_AND (x13, a4, x12);			/*               */
	ASM_D_LOAD (Preload1_2, merge[D21]);	/*               */

	ASM_XOR_NOT (x14, x10, x13);		/*    D           */
	ASM_XOR (x19, x6, x11);			/*             M  */
/* FREE MEMORY REF */

	ASM_XOR (Preload1_2, Preload1_2, x14);	/*    D           */
	ASM_D_STORE (out[D21], Preload1_2);	/*    D           */
	ASM_XOR (x15, x4, x14);			/*               */

	ASM_AND_NOT (x16, x15, a2);		/*               */
	ASM_AND (x20, a2, x19);			/*             M  */
/* FREE MEMORY REF */

	ASM_XOR (x17, x2, x16);			/*               */
	ASM_XOR (x21, x18, x20);		/*             MN */
/* FREE MEMORY REF */

	ASM_AND (x22, a3, x21);			/*             M  */
/**/	ASM_OR (x26, x6, a1);			/*            L   */
/* FREE MEMORY REF */

/**/	ASM_XOR (x23, x17, x22);		/*             M  */
	ASM_XOR (x24, a5, a2);			/*           K    */
/* FREE MEMORY REF */

	ASM_AND_NOT (x25, x24, x8);		/*           K    */
	ASM_XOR (x27, x26, a2);			/*            L   */
/* FREE MEMORY REF */

	ASM_AND_NOT (x28, a3, x27);		/*            L   */
	ASM_OR (x32, x18, x25);			/*       G   K    */
/* FREE MEMORY REF */

	ASM_XOR (x29, x25, x28);		/*           KL   */
	ASM_OR (x34, x27, x20);			/*         I      */
/* FREE MEMORY REF */

	ASM_OR (x30, a4, x29);			/*            L   */
	ASM_XOR (x33, x32, x10);		/*       G        */
	ASM_D_LOAD (Preload3_2, merge[D23]);	/*                */

	ASM_XOR (x31, x23, x30);		/*           KL   */
	ASM_AND (x37, x24, x34);		/*        H  K    */
/* FREE MEMORY REF */

	ASM_XOR (Preload3_2, Preload3_2, x31);	/*            L   */
	ASM_D_STORE (out[D23], Preload3_2);	/*            L   */
	ASM_AND_NOT (x38, x12, x37);		/*        H J     */

	ASM_AND (x35, a3, x34);			/*         I      */
	ASM_OR (x39, a4, x38);			/*        H       */
/* FREE MEMORY REF */

	ASM_XOR (x36, x33, x35);		/*       G I      */
	ASM_XOR (x41, a2, x2);			/*     E          */
	ASM_D_LOAD (Preload4_2, merge[D24]);	/*                */

	ASM_XOR_NOT (x40, x36, x39);		/*        HI      */
	ASM_AND_NOT (x42, x41, x33);		/*     E G        */
/* FREE MEMORY REF */

	ASM_XOR (Preload4_2, Preload4_2, x40);	/*        H       */
	ASM_D_STORE (out[D24], Preload4_2);	/*        H       */
	ASM_XOR (x43, x42, x29);		/*     EF         */

	ASM_OR (x46, x3, x20);			/*   CD           */
	ASM_AND (x47, a3, x3);			/*  B             */
/* FREE MEMORY REF */

	ASM_AND_NOT (x44, a3, x43);		/*     E          */
	ASM_XOR (x48, x46, x47);		/*  BC            */
/* FREE MEMORY REF */

	ASM_XOR (x45, x41, x44);		/*     E          */
	ASM_AND_NOT (x49, a4, x48);		/*  B             */
	ASM_D_LOAD (Preload2_2, merge[D22]);	/*                */

/* FREE ALU OP */
	ASM_XOR_NOT (x50, x45, x49);		/* AB             */
/* FREE MEMORY REF */

/* FREE ALU OP */
	ASM_XOR (Preload2_2, Preload2_2, x50);	/* A              */
	ASM_D_STORE (out[D22], Preload2_2);	/* A              */

    ASM_COMMENT_END_INCLUDE (end_of_s5_6);

#include "s0.h"
/* end of s2.h */
