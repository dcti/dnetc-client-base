/* simple.c v4.0 */

/* $Log: simple.c,v $
/* Revision 1.1.1.1  1998/06/14 14:23:51  remi
/* Initial integration.
/* */


#include "des.h"

static char *id="@(#)$Id: simple.c,v 1.1.1.1 1998/06/14 14:23:51 remi Exp $";

/* Solaris dumps on the high half of the Local and In 64-bit registers.
 * To try to identify when this happens, all routines in this code set
 * high register values in the stack and frame pointers upon entry.
 * Then at routine exit, they return the values that remain there.
 * It is hoped that these will be clobbered when the rest of the
 * registers are dumped.
 */

/* gcc says that these are the number of vars needed for each routine:
 * s1: 24, s2: 16, s3: 32, s4: 0, s5: 16, s6: 16, s7: 16, s8: 32
 * do_all: 72, do_all_fancy: 224
 */

/* As written, these routines seem that they will be operand starved.
 * Each group of 8 functions needs (6 + 6 + 6 + 4 + 4) * 8 = 208 memory
 * references.
 * Each group of 8 functions needs (51 + 6) * 8 = 456 instructions.
 * Assuming that 3 instructions are dispatched each clock, that is 152 clocks.
 * Not near enough time to do 208 instructions.
 *
 * Assume instead that (56 + 6) instructions are done in the floating point
 * pipeline.  Then there are 396 remaining instructions, which would
 * take 198 clocks.  This is much closer to the number of total clocks
 * needed.
 *
 * Assume that there are 2 memory references saved between functions, from
 * reusing "a" values.  If there are 7 functions, that would save 12 loads.
 * That would result in a need for 196 load/stores, which seems to fit well
 * with the number of instructions needed if one long function is done in VIS.
 *
 * It therefore seems that the load/store activity completely dominates
 * the work done here.  Followed closely by running out of integer registers.
 *
 * In order to get the VIS pipe running as early as possible, it seems valuable
 * to do a long function first.  S1 is long.  In order to remove as many
 * instructions as possible from the intevger pipeline, it seems valuable
 * to move a long function upstairs.  S5 is long.  Try things that way.
 */


/* forward references */
extern unsigned long asm_do_all (
		struct INNER_OFFSET_DISTANCES *Offset_List, int Special_Case );
extern unsigned long asm_do_all_fancy (
		struct INNER_OFFSET_DISTANCES *Offset_List, int Special_Case );
extern unsigned long asm_do_s1_s3 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s1 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s2 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s3 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s4 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s5 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s6 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s7 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long asm_do_s8 ( struct INNER_OFFSET_DISTANCES *Offset_List );

/* constants used to do loads and stores in the assembler code.  The assembler
 * code is not aware that data items are unsigned long long.  Instead, it
 * is written so that it thinks things are simply unsigned long
 * This is so gcc does not go berserk when it tries to do 64-bit refs
 */

#include "s_asm.h"

#include "s_pipeline.h"

unsigned long
asm_do_s1 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a3, t_a5;
	INNER_LOOP_SLICE t_k3, t_k4, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key3, *t_key4, *t_key5, *t_key6;

	ASM_PIPELINE_S1 (in, offsets);
    }

/* upon entry here, needs a3, a5, a4, a6 valid */
    do {

#include "s1_1n.h"
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s1_s3 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a3, t_a5;
	INNER_LOOP_SLICE t_k3, t_k4, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key3, *t_key4, *t_key5, *t_key6;

	ASM_PIPELINE_S1 (in, offsets);
    }

/* upon entry here, needs a3, a5, a4, a6 valid */
    do {

	{
	#include "s1_3.h"
	}

	{
	#include "s3_1n.h"
	}
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s2 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a1, t_a6;
	INNER_LOOP_SLICE t_k1, t_k2, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key2, *t_key5, *t_key6;

	ASM_PIPELINE_S2 (in, offsets);
    }

/* upon entry here, needs a1, a6, a5, a2 valid */
    do {

#include "s2_2n.h"
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s3 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a2, t_a3;
	INNER_LOOP_SLICE t_k2, t_k3, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key2, *t_key3, *t_key5, *t_key6;

	ASM_PIPELINE_S3 (in, offsets);
    }

/* upon entry here, needs a2, a3, a6, a5 valid */
    do {

#include "s3_3n.h"
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s4 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a1, t_a3;
	INNER_LOOP_SLICE t_k1, t_k2, t_k3, t_k5;
	INNER_LOOP_SLICE *t_key1, *t_key2, *t_key3, *t_key5;

	ASM_PIPELINE_S4 (in, offsets);
    }

/* upon entry here, needs a1, a3, a5, a2 valid */
    do {

#include "s4_4n.h"
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s5 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a3, t_a4;
	INNER_LOOP_SLICE t_k1, t_k3, t_k4, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key3, *t_key4, *t_key6;

	ASM_PIPELINE_S5 (in, offsets);
    }

/* upon entry here, needs a3, a4, a1, a6 valid */
    do {

#include "s5_5n.h"			/* TOADS NOT TESTED */
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s6 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a1, t_a5;
	INNER_LOOP_SLICE t_k1, t_k4, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key4, *t_key5, *t_key6;

	ASM_PIPELINE_S6 (in, offsets);
    }

/* upon entry here, needs a5, a1, a6, a4 valid */
    do {

#include "s6_6n.h"			/* TOADS NOT TESTED */
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s6f ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

#include "s6_f_regs.h"

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    ASM_PIPELINE_F_S6_all (in, offsets);

/* upon entry here, needs a5, a1, a6, a4 valid */
    {
	#include "s6_f.h"	/* debug s6_f */
    }
#include "sf0.h"

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s7f ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

#include "s6_f_regs.h"
#include "s7_f_regs.h"

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    ASM_PIPELINE_F_S7_all (in, offsets);

/* upon entry here, needs a5, a1, a6, a4 valid */
    {
	#include "s7_f.h"	/* debug s6_f */
    }
#include "sf0.h"

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s7 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a2, t_a4;
	INNER_LOOP_SLICE t_k2, t_k3, t_k4, t_k5;
	INNER_LOOP_SLICE *t_key2, *t_key3, *t_key4, *t_key5;

	ASM_PIPELINE_S7 (in, offsets);
    }

/* upon entry here, needs a2, a4, a5, a3 valid */
    do {

#include "s7_7n.h"
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

unsigned long
asm_do_s8 ( register struct INNER_OFFSET_DISTANCES *offsets
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a1, t_a3;
	INNER_LOOP_SLICE t_k1, t_k3, t_k4, t_k5;
	INNER_LOOP_SLICE *t_key1, *t_key3, *t_key4, *t_key5;

	ASM_PIPELINE_S8 (in, offsets);
    }

/* upon entry here, needs a3, a1, a4, a5 valid */
    do {

#include "s8_8n.h"			/* TOADS NOT TESTED */
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}

#ifdef DO_FLOAT_PIPE
unsigned long
asm_do_all ( register struct INNER_OFFSET_DISTANCES *offsets, int Special_Case
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

#ifdef USE_IDENTICAL_FLOAT_REGISTERS
#define USE_SAME_FLOAT_REGISTERS 1
#endif

#include "s6_f_regs.h"
#include "s7_f_regs.h"

/* order does not matter, except for PIPELINE */

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    {	INNER_LOOP_SLICE t_a1, t_a6;
	INNER_LOOP_SLICE t_k1, t_k2, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key2, *t_key5, *t_key6;
	ASM_PIPELINE_S2 (in, offsets);
    }

    do {
	{
	#include "s2_3.h"	/* function 2, pipelining into function 3 */
	}

	ASM_PIPELINE_F_S6 (in, offsets); /* functions 6 and 7 are in the VIS pipeline */

	{
	#include "s3_4.h"	/* function 3, pipelining into function 4 */
	}

	{
	#include "s4_5.h"	/* function 4, pipelining into function 6 */
	}

/* function 6 is in the VIS pipeline */

	ASM_PIPELINE_F_S7 (in, offsets); /* functions 6 and 7 are in the VIS pipeline */

	{			/* floating work happens in other routines */
	#include "s5_8_w6.h"	/* function 5, pipelining into function 8 */
	}

/* functions 6 and 7 are in the VIS pipeline */

	{
	#include "s8_1.h"	/* function 8, pipelining into function 1 */
	}

/* link forward to next pass, dovetailed into next SBOX */
	{			/* modify to write back results from VIS s7 */
	#include "s1_2n_w7.h"	/* function 1, pipelining into function 2 next pass */
	}
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
#include "sf0.h"
#undef USE_SAME_FLOAT_REGISTERS
}

/* This routine serves many purposes.  It is most useful when it is
 * doing all 8 sboxes in a row, for encryption pass after pass.
 * However, there are times when it is advantageous to do a few sboxes,
 * followed by a batch of 8-sbox passes.
 * Special Case 0 does only groups of 8.
 * Special Case 1 means do SBOX 1, link to the next pass, do sboxes 2, 3,
 *  4, 5, 6, 8, link to the next pass, fall into groups of 8.
 * Special Case 1 means do SBOX 7, link to the next pass, do sboxes 1, 2,
 *  3, 4, 6, 8, link to the next pass, fall into groups of 8.
 *
 * The LONGEST routines are S1 and S5.  If these are first, there will
 * be lots of opportunities to move data to the floating point pipeline
 * The SHORTEST routine is S4.  It MAY have the fewest live variables.
 * The routine with the fewest live variables should be at the end.
 */

unsigned long
asm_do_all_fancy ( register struct INNER_OFFSET_DISTANCES *offsets, int Special_Case
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

#ifdef USE_IDENTICAL_FLOAT_REGISTERS
#define USE_SAME_FLOAT_REGISTERS 1
#endif

#include "s6_f_regs.h"
#include "s7_f_regs.h"

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    if (Special_Case > 0)
    {	if (Special_Case == 1)	/* 12/16 of the time, do 1, next pass 2,3,4,5,6,8 */
	{   {	INNER_LOOP_SLICE t_a3, t_a5;
		INNER_LOOP_SLICE t_k3, t_k4, t_k5, t_k6;
		INNER_LOOP_SLICE *t_key3, *t_key4, *t_key5, *t_key6;
		ASM_PIPELINE_S1 (in, offsets);
	    }

/* link forward to next pass, dovetailed into next SBOX */
	    {
	    #include "s1_2n.h"	/* function 1, pipelining into function 2 next pass */
	    }
	    out = offsets->Dest;
	}
	else /* must be case 2, 3/16 of the time.  do 7, next pass 1,2,3,4,6,8 */
	{   {	INNER_LOOP_SLICE t_a2, t_a4;
		INNER_LOOP_SLICE t_k2, t_k3, t_k4, t_k5;
		INNER_LOOP_SLICE *t_key2, *t_key3, *t_key4, *t_key5;
		ASM_PIPELINE_S7 (in, offsets);
	    }

/* link forward to next pass, dovetailed into next SBOX */
	    {
	    #include "s7_1n.h"	/* function 7, pipelining into function 1 next pass */
	    }
	    out = offsets->Dest;

	    {
	    #include "s1_2.h"	/* function 1, pipelining into function 2 next pass */
	    }
	}

/* do 2,3,4,5,6, and possibly 8 hoisted up into floating point
 * This results in extra work.  S5 is not needed in case 2 above
 * But I have to worry about optimizing fewer special cases
 */
	{
	#include "s2_3.h"	/* function 2, pipelining into function 3 */
	}

	ASM_PIPELINE_F_S6 (in, offsets); /* function 6 is in the VIS pipeline */

	{
	#include "s3_4.h"	/* function 3, pipelining into function 4 */
	}

	{
	#include "s4_5.h"	/* function 4, pipelining into function 6 */
	}

/* function 6 is in the VIS pipeline */

	{			/* floating work happens in other routines */
	#include "s5_8_w6.h"	/* function 5, pipelining into function 8 */
	}

/* link forward to next pass, dovetailed into next SBOX */
	{			/* modify to write back results from VIS s6 */
	#include "s8_2n.h"	/* function 6, pipelining into function 8 */
	}
	out = offsets->Dest;

    }
    else
    {	INNER_LOOP_SLICE t_a1, t_a6;
	INNER_LOOP_SLICE t_k1, t_k2, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key2, *t_key5, *t_key6;
	ASM_PIPELINE_S2 (in, offsets);
    }

/* in next part, do 2,3,4,5,6,7, and possibly 8, 1 hoisted into floating point */
    do {
	{
	#include "s2_3.h"	/* function 2, pipelining into function 3 */
	}

	ASM_PIPELINE_F_S6 (in, offsets); /* functions 6 and 7 are in the VIS pipeline */

	{
	#include "s3_4.h"	/* function 3, pipelining into function 4 */
	}

	{
	#include "s4_5.h"	/* function 4, pipelining into function 6 */
	}

/* function 6 is in the VIS pipeline */

	ASM_PIPELINE_F_S7 (in, offsets); /* functions 6 and 7 are in the VIS pipeline */

	{
	#include "s5_8_w6.h"	/* function 5, pipelining into function 8 */
	}

/* functions 6 and 7 are in the VIS pipeline */

	{
	#include "s8_1.h"	/* function 8, pipelining into function 1 */
	}

/* link forward to next pass, dovetailed into next SBOX */
	{			/* modify to write back results from VIS s7 */
	#include "s1_2n_w7.h"	/* function 1, pipelining into function 2 next pass */
	}

	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
#include "sf0.h"
#undef USE_SAME_FLOAT_REGISTERS
}

#else /* DO_FLOAT_PIPE */
unsigned long
asm_do_all ( register struct INNER_OFFSET_DISTANCES *offsets, int Special_Case
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

#ifdef USE_IDENTICAL_FLOAT_REGISTERS
#define USE_SAME_FLOAT_REGISTERS 1
#endif

#include "s6_f_regs.h"
#include "s7_f_regs.h"

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

/* order does not matter, except for PIPELINE */

    {	INNER_LOOP_SLICE t_a1, t_a6;
	INNER_LOOP_SLICE t_k1, t_k2, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key2, *t_key5, *t_key6;
	ASM_PIPELINE_S2 (in, offsets);
    }

    do {
	{
	#include "s2_3.h"	/* function 2, pipelining into function 3 */
	}

	{
	#include "s3_4.h"	/* function 3, pipelining into function 4 */
	}

	{
	#include "s4_5.h"	/* function 4, pipelining into function 6 */
	}

	{
	#include "s5_6.h"	/* function 5, pipelining into function 6 */
	}

	{
	#include "s6_7.h"	/* function 6, pipelining into function 7 */
	}

	{
	#include "s7_8.h"	/* function 7, pipelining into function 8 */
	}

	{
	#include "s8_1.h"	/* function 8, pipelining into function 1 */
	}

/* link forward to next pass, dovetailed into next SBOX */
	{
	#include "s1_2n.h"	/* function 1, pipelining into function 2 next pass */
	}
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
#include "sf0.h"
#undef USE_SAME_FLOAT_REGISTERS
}

/* This routine serves many purposes.  It is most useful when it is
 * doing all 8 sboxes in a row, for encryption pass after pass.
 * However, there are times when it is advantageous to do a few sboxes,
 * followed by a batch of 8-sbox passes.
 * Special Case 0 does only groups of 8.
 * Special Case 1 means do SBOX 1, link to the next pass, do sboxes 2, 3,
 *  4, 5, 6, 8, link to the next pass, fall into groups of 8.
 * Special Case 1 means do SBOX 7, link to the next pass, do sboxes 1, 2,
 *  3, 4, 6, 8, link to the next pass, fall into groups of 8.
 *
 * The LONGEST routines are S1 and S5.  If these are first, there will
 * be lots of opportunities to move data to the floating point pipeline
 * The SHORTEST routine is S4.  It MAY have the fewest live variables.
 * The routine with the fewest live variables should be at the end.
 */

unsigned long
asm_do_all_fancy ( register struct INNER_OFFSET_DISTANCES *offsets, int Special_Case
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

#ifdef USE_IDENTICAL_FLOAT_REGISTERS
#define USE_SAME_FLOAT_REGISTERS 1
#endif

#include "s6_f_regs.h"
#include "s7_f_regs.h"

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

#if 0
    if ((offsets->Operand_Stride != REQUIRED_STRIDE)
	|| (offsets->Operand_Stride != (sizeof (INNER_LOOP_SLICE) * SIZE_FUDGE_FACTOR)))
    {	printf ("do_all_fancy called with inconsistent stride %x %x %x\n",
		offsets->Operand_Stride, REQUIRED_STRIDE,
		sizeof (INNER_LOOP_SLICE) * SIZE_FUDGE_FACTOR);
    }

    if (offsets->Operand_Size > (sizeof (INNER_LOOP_SLICE) * SIZE_FUDGE_FACTOR))
    {	printf ("do_all_fancy not calculating enough data %x %x\n",
	    offsets->Operand_Size, sizeof (INNER_LOOP_SLICE) * SIZE_FUDGE_FACTOR);
    }
#endif

    if (Special_Case > 0)
    {	if (Special_Case == 1)	/* 12/16 of the time, do 1, next pass 2,3,4,5,6,8 */
	{   {	INNER_LOOP_SLICE t_a3, t_a5;
		INNER_LOOP_SLICE t_k3, t_k4, t_k5, t_k6;
		INNER_LOOP_SLICE *t_key3, *t_key4, *t_key5, *t_key6;
		ASM_PIPELINE_S1 (in, offsets);
	    }

/* link forward to next pass, dovetailed into next SBOX */
	    {
	    #include "s1_2n.h"	/* function 1, pipelining into function 2 next pass */
	    }
	    out = offsets->Dest;
	}
	else /* must be case 2, 3/16 of the time.  do 7, next pass 1,2,3,4,6,8 */
	{   {	INNER_LOOP_SLICE t_a2, t_a4;
		INNER_LOOP_SLICE t_k2, t_k3, t_k4, t_k5;
		INNER_LOOP_SLICE *t_key2, *t_key3, *t_key4, *t_key5;
		ASM_PIPELINE_S7 (in, offsets);
	    }

/* link forward to next pass, dovetailed into next SBOX */
	    {
	    #include "s7_1n.h"	/* function 7, pipelining into function 1 next pass */
	    }
	    out = offsets->Dest;

	    {
	    #include "s1_2.h"	/* function 1, pipelining into function 2 next pass */
	    }
	}

/* do 2,3,4,5,6, and possibly 8 hoisted up into floating point
 * This results in extra work.  S5 is not needed in case 2 above
 * But I have to worry about optimizing fewer special cases
 */
	{
	#include "s2_3.h"	/* function 2, pipelining into function 3 */
	}

	{
	#include "s3_4.h"	/* function 3, pipelining into function 4 */
	}

	{
	#include "s4_5.h"	/* function 4, pipelining into function 6 */
	}

	{
	#include "s5_6.h"	/* function 5, pipelining into function 6 */
	}

	{
	#include "s6_8.h"	/* function 6, pipelining into function 8 */
	}

/* link forward to next pass, dovetailed into next SBOX */
	{
	#include "s8_2n.h"	/* function 6, pipelining into function 8 */
	}
	out = offsets->Dest;

    }
    else
    {	INNER_LOOP_SLICE t_a1, t_a6;
	INNER_LOOP_SLICE t_k1, t_k2, t_k5, t_k6;
	INNER_LOOP_SLICE *t_key1, *t_key2, *t_key5, *t_key6;
	ASM_PIPELINE_S2 (in, offsets);
    }

/* in next part, do 2,3,4,5,6,7, and possibly 8, 1 hoisted into floating point */
    do {
	{
	#include "s2_3.h"	/* function 2, pipelining into function 3 */
	}

	{
	#include "s3_4.h"	/* function 3, pipelining into function 4 */
	}

	{
	#include "s4_5.h"	/* function 4, pipelining into function 6 */
	}

	{
	#include "s5_6.h"	/* function 5, pipelining into function 6 */
	}

	{
	#include "s6_7.h"	/* function 6, pipelining into function 7 */
	}

	{
	#include "s7_8.h"	/* function 7, pipelining into function 8 */
	}

	{
	#include "s8_1.h"	/* function 8, pipelining into function 1 */
	}

/* link forward to next pass, dovetailed into next SBOX */
	{
	#include "s1_2n.h"	/* function 1, pipelining into function 2 next pass */
	}
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* check sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
#include "sf0.h"
#undef USE_SAME_FLOAT_REGISTERS
}

#endif /* DO_FLOAT_PIPE */

/* end of simple.c */
