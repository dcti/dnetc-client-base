/* do_all_fancy.c v4.0 */

/* $Log: do_all_fancy.c,v $
/* Revision 1.1.1.1  1998/06/14 14:23:49  remi
/* Initial integration.
/* */

static char *id="@(#)$Id: do_all_fancy.c,v 1.1.1.1 1998/06/14 14:23:49 remi Exp $";

#include "des.h"

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
do_all_fancy ( register struct INNER_OFFSET_DISTANCES *offsets, int Special_Case
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

    if (Special_Case > 0)
    {	if (Special_Case == 1)	/* 12/16 of the time, do 1, next pass 2,3,4,5,6,8 */
	{   PIPELINE_S1 (in, offsets);

	    wrap_s1 (in, merge, out, offsets, a3, a5, a4, a6);

/* link forward to next pass, dovetailed into next SBOX */
	    in = offsets->Next_Source;
	    offsets = offsets->Next_Offsets;

	    PIPELINE_S2 (in, offsets);

	    merge = offsets->Merge;
	    out = offsets->Dest;
	}
	else /* must be case 2, 3/16 of the time.  do 7, next pass 1,2,3,4,6,8 */
	{   PIPELINE_S7 (in, offsets);

	    wrap_s7 (in, merge, out, offsets, a2, a4, a5, a3);

/* link forward to next pass, dovetailed into next SBOX */
	    in = offsets->Next_Source;
	    offsets = offsets->Next_Offsets;

	    PIPELINE_S1 (in, offsets);

	    merge = offsets->Merge;
	    out = offsets->Dest;

	    wrap_s1 (in, merge, out, offsets, a3, a5, a4, a6);
	    PIPELINE_S2 (in, offsets);
	}

/* do 2,3,4,5,6, and possibly 8 hoisted up into floating point
 * This results in extra work.  S5 is not needed in case 2 above
 * But I have to worry about optimizing fewer special cases
 */
	wrap_s2 (in, merge, out, offsets, a1, a6, a5, a2);
	PIPELINE_S3 (in, offsets);

	wrap_s3 (in, merge, out, offsets, a2, a3, a6, a5);
	PIPELINE_S4 (in, offsets);

	wrap_s4 (in, merge, out, offsets, a1, a3, a5, a2);
	PIPELINE_S5 (in, offsets);

	wrap_s5 (in, merge, out, offsets, a3, a4, a1, a6);
	PIPELINE_S6 (in, offsets);

	wrap_s6 (in, merge, out, offsets, a5, a1, a6, a4);
	PIPELINE_S8 (in, offsets);

	wrap_s8 (in, merge, out, offsets, a3, a1, a4, a5);

/* link forward to next pass, dovetailed into next SBOX */
	in = offsets->Next_Source;
	offsets = offsets->Next_Offsets;

	PIPELINE_S2 (in, offsets);

	merge = offsets->Merge;
	out = offsets->Dest;
    }
    else
    {
	PIPELINE_S2 (in, offsets);
    }

/* might substitute a call to do_all (offsets, 0); here, instead of the below */
/* in next part, do 2,3,4,5,6,7, and possibly 8, 1 hoisted into floating point */
    do {
	wrap_s2 (in, merge, out, offsets, a1, a6, a5, a2);
	PIPELINE_S3 (in, offsets);

	wrap_s3 (in, merge, out, offsets, a2, a3, a6, a5);
	PIPELINE_S4 (in, offsets);

	wrap_s4 (in, merge, out, offsets, a1, a3, a5, a2);
	PIPELINE_S5 (in, offsets);

	wrap_s5 (in, merge, out, offsets, a3, a4, a1, a6);
	PIPELINE_S6 (in, offsets);

	wrap_s6 (in, merge, out, offsets, a5, a1, a6, a4);
	PIPELINE_S7 (in, offsets);

	wrap_s7 (in, merge, out, offsets, a2, a4, a5, a3);
	PIPELINE_S8 (in, offsets);

	wrap_s8 (in, merge, out, offsets, a3, a1, a4, a5);
	PIPELINE_S1 (in, offsets);

	wrap_s1 (in, merge, out, offsets, a3, a5, a4, a6);

/* link forward to next pass, dovetailed into next SBOX */
	in = offsets->Next_Source;
	offsets = offsets->Next_Offsets;

	PIPELINE_S2 (in, offsets);

	merge = offsets->Merge;
	out = offsets->Dest;

    } while (merge != (INNER_LOOP_SLICE *)0);

/* return sentinel value in high half of frame pointer */
    RETURN_64_BIT_SENTINEL;
}
