/* s7.h v3.0 */

/*
 * $Log: s7.h,v $
 * Revision 1.2  1998/06/14 15:19:18  remi
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
#define x8	T15
#define x9	T15
#define x23	T15

#define x3	T14
#define x27	T14
#define x28	T14
#define x29	T14
#define x31	T14
#define x32	T14

/* FREE */
#define x5	T13
#define x6	T13
/* FREE */
#define x25	T13
#define x26	T13
#define x38	T13
#define x39	T13
#define x40	T13
#define x41	T13

/* FREE */
#define x7	T12
#define x33	T12
/* FREE */
#define Preload3_7 T12

#define x4	T11
#define x24	T11

#define x1	T10
#define x2	T10
/* FREE */
#define x15	T10
#define x44	T10
#define x45	T10
#define x46	T10

#define x30	a4	/* GROSS, but ran out of registers */
/* FREE */
#define t_a2	a4	/* could be a4, but slows down? */

/* FREE */
#define t_key2	T8
/* FREE but hold */
#define t_k2	T8

/* FREE */
#define x20	T7
#define x42	T7
#define x43	T7
#define x47	T7

/* FREE */
#define x34	T6
#define x35	T6
#define x36	T6
#define x37	T6
/* FREE */
#define t_a4	T6

/* FREE */
#define t_key4	T5
/* FREE but hold */
#define t_k4	T5

#define x10	T4
#define x11	T4
/* FREE */
#define x21	T4
#define x22	T4
#define x48	T4
#define x49	T4
#define x50	T4
#define x51	T4

#define key6	T3
/* FREE but hold */
#define k6	T3
/* FREE */
#define Preload2_7 T3
/* FREE */
#define Preload4_7 T3

/* FREE */
#define x12	T2
#define x13	T2
#define x14	T2
/* FREE */
#define Preload1_7 T2
/* FREE */
#define t_key5	T2
/* FREE but hold */
#define t_k5	T2

/* FREE */
#define key1	T1
/* FREE but hold */
#define k1	T1
/* FREE */
#define x16	T1
#define x17	T1
#define x18	T1
#define x19	T1
/* FREE */
#define t_key3	T1
/* FREE but hold */
#define t_k3	T1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_SLICE x1, x2, x3, x4, x5, x6, x7, x8;
    INNER_LOOP_SLICE x9, x10, x11, x12, x13, x14, x15, x16;
    INNER_LOOP_SLICE x17, x18, x19, x20, x21, x22, x23, x24;
    INNER_LOOP_SLICE x25, x26, x27, x28, x29, x30, x31, x32;
    INNER_LOOP_SLICE x33, x34, x35, x36, x37, x38, x39, x40;
    INNER_LOOP_SLICE x41, x42, x43, x44, x45, x46, x47, x48;
    INNER_LOOP_SLICE x49, x50, x51;
    INNER_LOOP_SLICE Preload1_7, Preload2_7, Preload3_7, Preload4_7;

    INNER_LOOP_SLICE *key1, *key6;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a2, a4, a5, a3 valid */

	ASM_A_LOAD (key6, offsets->Key_Ptrs[OFFSET7 + 6]);

	ASM_AND (x1, a2, a4);			/*             M  */
	ASM_OR (x10, a2, a4);			/*               O */
	ASM_D_LOAD (a6, in[S76]);

	ASM_XOR (x2, x1, a5);			/*             M  */
	ASM_OR (x11, x10, a5);			/*               O */
	ASM_D_LOAD (k6, key6[0]);

	ASM_AND (x3, a4, x2);			/*             M  */
	ASM_AND_NOT (x12, a5, a2);		/*              N */
	ASM_A_LOAD (key1, offsets->Key_Ptrs[OFFSET7 + 1]);

	ASM_XOR_NOT (x4, x3, a2);		/*             M  */
	ASM_OR (x13, a3, x12);			/*              N */
	ASM_D_LOAD (a1, in[S71]);

	ASM_AND (x5, a3, x4);			/*             M  */
	ASM_XOR (x14, x11, x13);		/*              N */
	ASM_D_LOAD (k1, key1[0]);	

	ASM_XOR_NOT (x7, a3, x5);		/*             M  */
	ASM_XOR (x6, x2, x5);			/*               O */
/* FREE MEMORY REF */

	ASM_XOR (a6, a6, k6);
	ASM_XOR (a1, a1, k1);
/* FREE MEMORY REF */

	ASM_AND (x8, a6, x7);			/*             M  */
	ASM_XOR (x15, x3, x6);			/*            L   */
/* FREE MEMORY REF */

	ASM_XOR (x9, x6, x8);			/*             M  */
	ASM_OR (x16, a6, x15);			/*            L   */
/* FREE MEMORY REF */

	ASM_XOR (x17, x14, x16);		/*            L N */
	ASM_AND_NOT (x20, a4, a3);		/*          J     */
/* FREE MEMORY REF */

	ASM_AND (x18, a1, x17);			/*            L   */
	ASM_AND_NOT (x21, a2, x20);		/*          J     */
	ASM_D_LOAD (Preload1_7, merge[D71]);	/*                */

	ASM_XOR (x19, x9, x18);			/*            LM  */
	ASM_AND (x22, a6, x21);			/*          J     */
/* FREE MEMORY REF */

	ASM_XOR (Preload1_7, Preload1_7, x19);	/*            L   */
	ASM_D_STORE (out[D71], Preload1_7);	/*            L   */
	ASM_XOR_NOT (x24, a4, x4);		/*           K    */

	ASM_OR (x25, a3, x3);			/*           K    */
	ASM_XOR (x27, a3, x3);			/*         I      */
/* FREE MEMORY REF */

	ASM_XOR (x26, x24, x25);		/*           K    */
	ASM_AND (x28, x27, a2);			/*         I      */
/* FREE MEMORY REF */

	ASM_XOR_NOT (x23, x9, x22);		/*          J     */
	ASM_AND_NOT (x29, a6, x28);		/*         I      */
/* FREE MEMORY REF */

/**/	ASM_XOR (x30, x26, x29);		/*         I K    */
	ASM_OR (x34, a2, x24);			/*      F         */
/* FREE MEMORY REF */

	ASM_OR (x31, a1, x30);			/*         I      */
	ASM_XOR_NOT (x33, x7, x30);		/*        H       */
	ASM_D_LOAD (Preload2_7, merge[D72]);	/*                */

	ASM_XOR (x32, x23, x31);		/*         IJ     */
	ASM_XOR_NOT (x35, x34, x19);		/*      F         */
/* FREE MEMORY REF */

	ASM_XOR (Preload2_7, Preload2_7, x32);	/*         I      */
	ASM_D_STORE (out[D72], Preload2_7);	/*         I      */
	ASM_OR (x36, a6, x35);			/*      F         */

	ASM_XOR (x37, x33, x36);		/*      F H       */
	ASM_AND_NOT (x38, x26, a3);		/*     E G        */
/* FREE MEMORY REF */

	ASM_OR (x39, x38, x30);			/*     E          */
	ASM_OR (x42, a5, x20);			/*    D           */
/* FREE MEMORY REF */

	ASM_OR_NOT (x40, a1, x39);		/*     E          */
	ASM_XOR (x43, x42, x33);		/*    D           */
	ASM_D_LOAD (Preload3_7, merge[D73]);	/*                */

	ASM_XOR (x41, x37, x40);		/*     EF         */
	ASM_XOR (x44, a2, x15);			/* A              */
/* FREE MEMORY REF */

	ASM_XOR (Preload3_7, Preload3_7, x41);	/*     E          */
	ASM_D_STORE (out[D73], Preload3_7);	/*     E          */
	ASM_AND_NOT (x45, x24, x44);		/* A              */

	ASM_AND (x46, a6, x45);			/* A              */
	ASM_AND (x48, a3, x22);			/*   C       J    */
/* FREE MEMORY REF */

/* FREE ALU OP */
	ASM_XOR (x49, x48, x46);		/* A              */
/* FREE MEMORY REF */

	ASM_XOR (x47, x43, x46);		/*  B D           */
	ASM_OR (x50, a1, x49);			/* A C            */
	ASM_D_LOAD (Preload4_7, merge[D74]);	/*                */

/* FREE ALU OP */
	ASM_XOR (x51, x47, x50);		/* AB             */
/* FREE MEMORY REF */

/* FREE ALU OP */
	ASM_XOR (Preload4_7, Preload4_7, x51);	/* A              */
	ASM_D_STORE (out[D74], Preload4_7);	/* A              */     

    ASM_COMMENT_END_INCLUDE (end_of_s7);

#include "s0.h"
/* end of s7.h */
