/* s4_4n.h v3.0 */
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
 * $Log: s4_4n.h,v $
 * Revision 1.2  1998/06/14 15:19:05  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:50  remi
 * Initial integration.
 *
 */


#ifdef MANUAL_REGISTER_ALLOCATION

    register INNER_LOOP_SLICE T1, T2, T3, T4, T5, T6, T7, T8;
    register INNER_LOOP_SLICE T9, T10, T11, T12, T13;

/* FREE */
#define x21	a1	/* GROSS, but ran out of registers */
#define x22	a1

/* FREE */
#define Preload1_4 T13

/* FREE */
#define x10	T12
#define x11	T12
#define x24	T12

/* FREE */
#define t_a1	T11

/* FREE */
#define t_key1	T10
/* FREE but hold */
#define t_k1	T10

/* FREE */
#define x27	T9
#define x28	T9
#define x29	T9
#define x30	T9

#define x3	T8
/* FREE */
#define Preload2_4 T8
/* FREE */
#define Preload3_4 T8

#define x13	T7
#define x14	T7
/* FREE */
#define x32	T7
#define x33	T7
/* FREE */
#define x36	T7
#define x37	T7

#define x1	T6
#define x2	T6
#define x12	T6
#define x15	T6
/* FREE */
#define x31	T6
#define x34	T6
#define x35	T6
#define x38	T6
#define x39	T6

#define key4	T5
/* FREE but hold */
#define k4	T5
/* FREE */
#define x18	T5
#define x19	T5
#define x20	T5
#define x23	T5
/* FREE */
#define Preload4_4 T5

#define x6	T4
#define x7	T4
#define x8	T4
#define x9	T4
#define x25	T4
#define x26	T4
/* FREE */
#define t_a3	T4

#define key6	T3
/* FREE but hold */
#define k6	T3
/* FREE */
#define t_key3	T3
/* FREE but hold */
#define t_k3	T3

#define x4	T2
#define x5	T2
/* FREE */
#define t_key5	T2
/* FREE but hold */
#define t_k5	T2

#define x16	T1
#define x17	T1
/* FREE */
#define t_key2	T1
/* FREE but hold */
#define t_k2	T1

#else /* MANUAL_REGISTER_ALLOCATION */

    INNER_LOOP_SLICE x1, x2, x3, x4, x5, x6, x7, x8;
    INNER_LOOP_SLICE x9, x10, x11, x12, x13, x14, x15, x16;
    INNER_LOOP_SLICE x17, x18, x19, x20, x21, x22, x23, x24;
    INNER_LOOP_SLICE x25, x26, x27, x28, x29, x30, x31, x32;
    INNER_LOOP_SLICE x33, x34, x35, x36, x37, x38, x39;
    INNER_LOOP_SLICE Preload1_4, Preload2_4, Preload3_4, Preload4_4;

    INNER_LOOP_SLICE k4, k6;
    INNER_LOOP_SLICE t_a1, t_a3;
    INNER_LOOP_SLICE t_k1, t_k2, t_k3, t_k5;

    INNER_LOOP_SLICE *key4, *key6;
    INNER_LOOP_SLICE *t_key1, *t_key2, *t_key3, *t_key5;

#endif /* MANUAL_REGISTER_ALLOCATION */

/* upon entry here, needs a1, a3, a5, a2 valid */

	ASM_A_LOAD (key4, offsets->Key_Ptrs[OFFSET4 + 4]);

	ASM_OR (x1, a1, a3);			/*            L   */
	ASM_OR (x4, a2, a3);			/*            L   */
	ASM_D_LOAD (a4, in[S44]);

	ASM_AND (x2, a5, x1);			/*            L   */
	ASM_AND_NOT (x6, a3, a1);		/*     E          */
	ASM_D_LOAD (k4, ((INNER_LOOP_SLICE *)key4)[0]);

	ASM_XOR (x3, a1, x2);			/*            L   */
	ASM_XOR (x12, a3, x2);			/*          J     */
	ASM_A_LOAD (key6, offsets->Key_Ptrs[OFFSET4 + 6]);

	ASM_XOR (x5, x3, x4);			/*            L   */
	ASM_OR (x7, x6, x3);			/*     E          */
	ASM_D_LOAD (a6, in[S46]);

	ASM_AND_NOT (x13, a2, x12);		/*          J     */
	ASM_XOR (x16, a3, a5);			/*         I      */
	ASM_D_LOAD (k6, ((INNER_LOOP_SLICE *)key6)[0]);

/* FREE ALU OP */
/* FREE ALU OP */
    ASM_A_LOAD (in, offsets->Next_Source);		/* PIPELINE */

/* FREE ALU OP */
/* FREE ALU OP */
    ASM_A_LOAD (offsets, offsets->Next_Offsets);	/* PIPELINE */

	ASM_XOR (x14, x7, x13);			/*          J     */
	ASM_OR (x15, x12, x3);			/*           K    */
    ASM_D_LOAD (t_a1, in[S41]);				/* PIPELINE */

	ASM_XOR (a4, a4, k4);
	ASM_XOR (a6, a6, k6);
    ASM_A_LOAD (t_key1, offsets->Key_Ptrs[OFFSET4 + 1]);	/* PIPELINE */

	ASM_AND (x8, a2, x7);			/*     E          */
	ASM_AND_NOT (x17, x16, a2);		/*         I      */
    ASM_A_LOAD (t_key3, offsets->Key_Ptrs[OFFSET4 + 3]);	/* PIPELINE */

	ASM_XOR (x27, a3, x8);			/* A              */
	ASM_XOR (x9, a5, x8);			/*     E          */
    ASM_D_LOAD (t_k1, ((INNER_LOOP_SLICE *)t_key1)[0]);		/* PIPELINE */

	ASM_XOR (x18, x15, x17);		/*         I K    */
	ASM_AND (x10, a4, x9);			/*     E          */
    ASM_D_LOAD (t_k3, ((INNER_LOOP_SLICE *)t_key3)[0]);		/* PIPELINE */

	ASM_OR (x19, a4, x18);			/*         I      */
	ASM_XOR (x11, x5, x10);			/*     E      L   */
    ASM_A_LOAD (t_key5, offsets->Key_Ptrs[OFFSET4 + 5]);	/* PIPELINE */

	ASM_XOR (x20, x14, x19);		/*         IJ     */
	ASM_XOR (x28, x27, x17);		/* A     G        */
    ASM_A_LOAD (t_key2, offsets->Key_Ptrs[OFFSET4 + 2]);	/* PIPELINE */

	ASM_OR (x21, a6, x20);			/*         I      */
	ASM_AND (x25, a2, x9);			/*      F         */
    ASM_D_LOAD (t_k5, ((INNER_LOOP_SLICE *)t_key5)[0]);		/* PIPELINE */

	ASM_XOR (x26, x25, x15);		/*      F         */
	ASM_AND_NOT (x29, a4, x28);		/* A              */
    ASM_D_LOAD (t_k2, ((INNER_LOOP_SLICE *)t_key2)[0]);		/* PIPELINE */

	ASM_XOR (x22, x11, x21);		/*     E   I      */
	ASM_XOR (x30, x26, x29);		/* A    F         */
    ASM_D_LOAD (t_a3, in[S43]);					/* PIPELINE */

	ASM_AND (x23, a6, x20);			/*        H       */
	ASM_XOR (x31, x11, x30);		/* A   E          */
	ASM_D_LOAD (Preload1_4, merge[D41]);	/*                */

	ASM_XOR_NOT (x24, x23, x11);		/*     E  H       */
	ASM_AND_NOT (x32, a2, x31);		/* A              */
	ASM_D_LOAD (Preload2_4, merge[D42]);	/*                */

	ASM_XOR (Preload1_4, Preload1_4, x22);	/*         I      */
	ASM_D_STORE (out[D41], Preload1_4);	/*         I      */
	ASM_XOR (x33, x22, x32);		/* A  D           */

	ASM_XOR (Preload2_4, Preload2_4, x24);	/*        H       */
	ASM_D_STORE (out[D42], Preload2_4);	/*        H       */
	ASM_AND_NOT (x34, x31, a4);		/*   C            */

/* FREE ALU OP */
	ASM_XOR (x35, x33, x34);		/* A C            */
	ASM_D_LOAD (Preload3_4, merge[D43]);	/*                */

	ASM_OR (x36, a6, x35);			/* A              */
	ASM_XOR (x38, x23, x35);		/*  B             */
	ASM_D_LOAD (Preload4_4, merge[D44]);	/*                */

    ASM_XOR (a1, t_a1, t_k1);				/* PIPELINE */
	ASM_XOR_NOT (x37, x30, x36);		/* A              */
    ASM_A_LOAD (merge, offsets->Merge);			/* PIPELINE */

	ASM_XOR (Preload3_4, Preload3_4, x37);	/* A              */
	ASM_D_STORE (out[D43], Preload3_4);	/* A              */
	ASM_XOR (x39, x38, x37);		/* AB             */

/* FREE ALU OP */
/* FREE ALU OP */
    ASM_D_LOAD (a5, in[S45]);				/* OUCH PIPELINE */

/* FREE ALU OP */
/* FREE ALU OP */
    ASM_D_LOAD (a2, in[S42]);				/* OUCH PIPELINE */

	ASM_XOR (Preload4_4, Preload4_4, x39);	/* A              */
	ASM_D_STORE (out[D44], Preload4_4);	/* A              */     
    ASM_XOR (a3, t_a3, t_k3);				/* PIPELINE */

    ASM_XOR (a5, a5, t_k5);				/* PIPELINE */
    ASM_XOR (a2, a2, t_k2);				/* PIPELINE */

    ASM_COMMENT_END_INCLUDE (end_of_s4_4n);

#include "s0.h"
/* end of s4_4n.h */
