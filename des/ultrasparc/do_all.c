/* do_all.c v4.0 */

/* $Log: do_all.c,v $
/* Revision 1.1  1998/06/14 14:23:49  remi
/* Initial revision
/* */

static char *id="@(#)$Id: do_all.c,v 1.1 1998/06/14 14:23:49 remi Exp $";

#include "des.h"

unsigned long
do_all ( register struct INNER_OFFSET_DISTANCES *offsets, int Special_Case
) {
    register INNER_LOOP_SLICE *in = offsets->Source;
    register INNER_LOOP_SLICE *merge = offsets->Merge;
    register INNER_LOOP_SLICE *out = offsets->Dest;

    register INNER_LOOP_SLICE a1, a2, a3, a4, a5, a6;

/* set sentinel value into high half of frame pointer */
    SET_64_BIT_SENTINEL;

/* order does not matter, except for PIPELINE */

	PIPELINE_S2 (in, offsets);

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
