/* s5.h v3.0 */

/* $Log: s5.h,v $
/* Revision 1.1.1.1  1998/06/14 14:23:50  remi
/* Initial integration.
/* */


#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_SLICE T1, T2, T3, T4, T5, T6, T7, T8;
    register INNER_LOOP_SLICE T9, T10, T11, T12, T13, T14, T15;

#define x29	a1	/* GROSS, but ran out of registers */
#define x30	a1

/* FREE */
#define Preload4_5 a6
/* FREE */
#define x28	a6	/* GROSS, but ran out of registers */

/* FREE */
#define x18	T15
#define x19	T15

#define x15	T14
#define x16	T14
#define x37	T14

/* FREE */
#define Preload2_5 T13
/* FREE */
#define x39	T13
#define x40	T13
#define x41	T13
#define x42	T13

/* FREE */
#define x24	T12
/* FREE */
#define Preload3_5 T12

/* FREE */
#define x8	T11
#define x33	T11
#define x34	T11

#define x3	T10
#define x14	T10
#define x17	T10
#define x31	T10
#define x38	T10
#define x49	T10
#define x50	T10

#define x5	T9
/* FREE */
#define x35	T9
#define x36	T9
/* FREE */
#define t_a3	T9

#define key5	T8
/* FREE but hold */
#define k5	T8
/* FREE */
#define x10	T8
#define x11	T8
/* FREE */
#define t_key3	T8
/* FREE but hold */
#define t_k3	T8

#define x2	T7
#define x25	T7
#define x26	T7
#define x27	T7
#define x45	T7
#define x46	T7
#define x47	T7
#define x48	T7

/* FREE */
#define x20	T6
#define x21	T6
#define x51	T6
#define x52	T6
#define x53	T6
#define x54	T6
#define x55	T6
#define x56	T6

#define x1	T5
#define x7	T5
#define x32	T5
#define x43	T5
#define x44	T5
/* FREE */
#define Preload1_5 T5

#define x6	T4
/* FREE */
#define t_a4	T4

#define x4	T3
#define x22	T3
#define x23	T3
/* FREE */
#define t_key4	T3
/* FREE but hold */
#define t_k4	T3

#define key2	T2
/* FREE but hold */
#define k2	T2
/* FREE */
#define x9	T2
/* FREE */
#define t_key1	T2
/* FREE but hold */
#define t_k1	T2

#define x12	T1
#define x13	T1
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
    INNER_LOOP_SLICE Preload1_5, Preload2_5, Preload3_5, Preload4_5;

    INNER_LOOP_SLICE *key2, *key5;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a3, a4, a1, a6 valid */

	ASM_A_LOAD (key5, offsets->Key_Ptrs[OFFSET5 + 5]);

	ASM_AND_NOT (x1, a3, a4);		/*               */
	ASM_AND_NOT (x3, a1, a3);		/*               */
	ASM_D_LOAD (a5, in[S55]);

	ASM_XOR (x2, x1, a1);			/*               */
	ASM_XOR (x6, a4, a1);			/*              N */
	ASM_D_LOAD (k5, ((INNER_LOOP_SLICE *)key5)[0]);

	ASM_OR (x4, a6, x3);			/*               */
	ASM_OR (x7, x6, x1);			/*              N */
	ASM_A_LOAD (key2, offsets->Key_Ptrs[OFFSET5 + 2]);

	ASM_XOR (x5, x2, x4);			/*               */
	ASM_AND (x12, a3, x7);			/*              N */
	ASM_D_LOAD (a2, in[S52]);

	ASM_XOR (x13, x12, a4);			/*              N */
	ASM_XOR (x15, a4, x3);			/*          J     */
	ASM_D_LOAD (k2, ((INNER_LOOP_SLICE *)key2)[0]);

	ASM_AND_NOT (x14, x13, x3);		/*              N */
	ASM_OR (x16, a6, x15);			/*          J     */
/* FREE MEMORY REF */

	ASM_XOR (a5, a5, k5);
	ASM_XOR (a2, a2, k2);
/* FREE MEMORY REF */

	ASM_AND_NOT (x8, x7, a6);		/*               */
	ASM_XOR (x17, x14, x16);		/*          J   N */
/* FREE MEMORY REF */

	ASM_XOR (x9, a3, x8);			/*               */
	ASM_OR (x18, a5, x17);			/*          J     */
/* FREE MEMORY REF */

	ASM_OR (x10, a5, x9);			/*               */
	ASM_XOR (x19, x13, x18);		/*          J     */
/* FREE MEMORY REF */

	ASM_XOR (x11, x5, x10);			/*              N */
	ASM_AND_NOT (x20, x19, a2);		/*          J     */
	ASM_D_LOAD (Preload4_5, merge[D54]);	/*                */

	ASM_XOR (x21, x11, x20);		/*          J     */
	ASM_AND (x22, a4, x4);			/*             M  */
/* FREE MEMORY REF */

	ASM_XOR (Preload4_5, Preload4_5, x21);	/*          J     */
	ASM_D_STORE (out[D54], Preload4_5);	/*          J     */
/**/	ASM_XOR (x24, a1, x9);			/*            L   */

	ASM_XOR (x23, x22, x17);		/*          J  M  */
	ASM_AND (x25, x2, x24);			/*            L   */
/* FREE MEMORY REF */

	ASM_AND_NOT (x26, a5, x25);		/*            L   */
	ASM_OR (x28, a4, x24);			/*           K    */
/* FREE MEMORY REF */

	ASM_XOR (x27, x23, x26);		/*            LM  */
/**/	ASM_AND_NOT (x29, x28, a2);		/*           K    */
	ASM_D_LOAD (Preload2_5, merge[D52]);	/*                */

	ASM_XOR (x30, x27, x29);		/*           KL   */
	ASM_AND_NOT (x33, x8, a4);		/*       G        */
/* FREE MEMORY REF */

	ASM_XOR (Preload2_5, Preload2_5, x30);	/*           K    */
	ASM_D_STORE (out[D52], Preload2_5);	/*           K    */
	ASM_XOR (x34, x33, a3);			/*       G        */

	ASM_AND (x31, x17, x5);			/*          J     */
	ASM_AND (x35, a5, x34);			/*       G        */
/* FREE MEMORY REF */

	ASM_AND_NOT (x32, x7, x31);		/*          J     */
	ASM_XOR (x38, x9, x31);			/*      F  I      */
/* FREE MEMORY REF */

	ASM_XOR (x36, x32, x35);		/*       G  J     */
	ASM_OR (x37, x13, x16);			/*        H       */
/* FREE MEMORY REF */

	ASM_OR (x39, a5, x38);			/*      F         */
	ASM_AND_NOT (x43, x19, x32);		/*  B             */
/* FREE MEMORY REF */

	ASM_XOR (x40, x37, x39);		/*      F H       */
	ASM_OR (x45, x27, x43);			/*  B             */
/* FREE MEMORY REF */

	ASM_OR (x41, a2, x40);			/*      F         */
	ASM_XOR (x44, x43, x24);		/*     E          */
	ASM_D_LOAD (Preload3_5, merge[D53]);	/*                */

	ASM_XOR_NOT (x42, x36, x41);		/*      FG        */
	ASM_XOR (x46, x45, x6);			/*  B             */
/* FREE MEMORY REF */

	ASM_XOR (Preload3_5, Preload3_5, x42);	/*      F         */
	ASM_D_STORE (out[D53], Preload3_5);	/*      F         */
	ASM_XOR (x51, x21, x38);		/* A  D           */

	ASM_AND (x49, x6, x38);			/*   CD           */
	ASM_AND_NOT (x52, x28, x51);		/* A              */
/* FREE MEMORY REF */

	ASM_XOR (x50, x49, x34);		/*   C            */
	ASM_AND (x53, a5, x52);			/* A              */
/* FREE MEMORY REF */

	ASM_AND_NOT (x47, a5, x46);		/*  B             */
	ASM_XOR (x54, x50, x53);		/* A C            */
/* FREE MEMORY REF */

	ASM_XOR (x48, x44, x47);		/*  B  E          */
	ASM_OR (x55, a2, x54);			/* A              */
	ASM_D_LOAD (Preload1_5, merge[D51]);	/*                */

/* FREE ALU OP */
	ASM_XOR (x56, x48, x55);		/* AB             */
/* FREE MEMORY REF */

/* FREE ALU OP */
	ASM_XOR (Preload1_5, Preload1_5, x56);	/* A              */
	ASM_D_STORE (out[D51], Preload1_5);	/* A              */

    ASM_COMMENT_END_INCLUDE (end_of_s5);

#include "s0.h"
/* end of s5.h */
