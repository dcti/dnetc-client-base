/* des.h v4.0 */

/*
 * $Log: des.h,v $
 * Revision 1.3  1998/06/16 06:27:36  remi
 * - Integrated some patches in the UltraSparc DES code.
 * - Cleaned-up C++ style comments in the UltraSparc DES code.
 * - Replaced "rm `find ..`" by "find . -name ..." in superclean.
 *
 * Revision 1.2  1998/06/14 15:18:30  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:48  remi
 * Initial integration.
 *
 */

#define MAGIC_SENTINEL_VAL 0x5a3cf012ul

#define SET_64_BIT_SENTINEL /* put MAGIC_SENTINEL_VAL in high half of register */
#define RETURN_64_BIT_SENTINEL			return (MAGIC_SENTINEL_VAL);

/* defines are set in configure & Makefile when compiling DCTI client */
#ifndef IN_DCTI_CLIENT
#include "s_paramaters.h"
#endif

/* comment this out to use 64-bit operands but calculate only 32 valid bits at once */
    typedef unsigned long INNER_LOOP_SLICE; /* assuming 64-bit work done here */
    #define SIZE_FUDGE_FACTOR 2		/* 2 if outer loop is 64-bit */
    #define LOAD_STORE_64_BIT_INTS	/* define if ldx/stx instructions wanted */
    #define REQUIRED_SIZE 4
    #define REQUIRED_STRIDE 8

#ifdef DO_FLOAT_PIPE
#ifdef ASM
    typedef double INNER_LOOP_FSLICE; /* assuming 64-bit work done here */
#else
    typedef unsigned long INNER_LOOP_FSLICE; /* assuming 64-bit work done here */
#endif
#else
    typedef unsigned long INNER_LOOP_FSLICE; /* assuming 64-bit work done here */
#endif

struct INNER_OFFSET_DISTANCES
    {	long Operand_Size;
	long Operand_Stride;
	INNER_LOOP_SLICE * Key_Ptrs[48]; /* pointers to pick up 48 keys */
	INNER_LOOP_SLICE * Source;
	INNER_LOOP_SLICE * Merge;
	INNER_LOOP_SLICE * Dest;
	struct INNER_OFFSET_DISTANCES * Next_Offsets;
	INNER_LOOP_SLICE * Next_Source;
    };

/* external routine declarations */
extern unsigned long do_s1 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s2 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s3 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s4 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s5 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s6 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s7 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s8 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_s1_s3 ( struct INNER_OFFSET_DISTANCES *Offset_List );
extern unsigned long do_all (
		struct INNER_OFFSET_DISTANCES *Offset_List, int Special_Case );
extern unsigned long do_all_fancy (
		struct INNER_OFFSET_DISTANCES *Offset_List, int Special_Case );

/* constants used to do loads and stores in the assembler code.  The assembler
 * code is not aware that data items are unsigned long long.  Instead, it
 * is written so that it thinks things are simply unsigned long
 * This is so gcc does not go berserk when it tries to do 64-bit refs
 */

#define OFFSET1         (0 - 1)
#define S11             (SIZE_FUDGE_FACTOR * 31)
#define S12             (SIZE_FUDGE_FACTOR * 0)
#define S13             (SIZE_FUDGE_FACTOR * 1)
#define S14             (SIZE_FUDGE_FACTOR * 2)
#define S15             (SIZE_FUDGE_FACTOR * 3)
#define S16             (SIZE_FUDGE_FACTOR * 4)
#define D11             (SIZE_FUDGE_FACTOR * 8)
#define D12             (SIZE_FUDGE_FACTOR * 16)
#define D13             (SIZE_FUDGE_FACTOR * 22)
#define D14             (SIZE_FUDGE_FACTOR * 30)

#define OFFSET2		(6 - 1)
#define S21		(SIZE_FUDGE_FACTOR * 3)
#define S22		(SIZE_FUDGE_FACTOR * 4)
#define S23		(SIZE_FUDGE_FACTOR * 5)
#define S24		(SIZE_FUDGE_FACTOR * 6)
#define S25		(SIZE_FUDGE_FACTOR * 7)
#define S26		(SIZE_FUDGE_FACTOR * 8)
#define D21		(SIZE_FUDGE_FACTOR * 12)
#define D22		(SIZE_FUDGE_FACTOR * 27)
#define D23		(SIZE_FUDGE_FACTOR * 1)
#define D24		(SIZE_FUDGE_FACTOR * 17)

#define OFFSET3		(12 - 1)
#define S31		(SIZE_FUDGE_FACTOR * 7)
#define S32		(SIZE_FUDGE_FACTOR * 8)
#define S33		(SIZE_FUDGE_FACTOR * 9)
#define S34		(SIZE_FUDGE_FACTOR * 10)
#define S35		(SIZE_FUDGE_FACTOR * 11)
#define S36		(SIZE_FUDGE_FACTOR * 12)
#define D31		(SIZE_FUDGE_FACTOR * 23)
#define D32		(SIZE_FUDGE_FACTOR * 15)
#define D33		(SIZE_FUDGE_FACTOR * 29)
#define D34		(SIZE_FUDGE_FACTOR * 5)

#define OFFSET4		(18 - 1)
#define S41		(SIZE_FUDGE_FACTOR * 11)
#define S42		(SIZE_FUDGE_FACTOR * 12)
#define S43		(SIZE_FUDGE_FACTOR * 13)
#define S44		(SIZE_FUDGE_FACTOR * 14)
#define S45		(SIZE_FUDGE_FACTOR * 15)
#define S46		(SIZE_FUDGE_FACTOR * 16)
#define D41		(SIZE_FUDGE_FACTOR * 25)
#define D42		(SIZE_FUDGE_FACTOR * 19)
#define D43		(SIZE_FUDGE_FACTOR * 9)
#define D44		(SIZE_FUDGE_FACTOR * 0)

#define OFFSET5		(24 - 1)
#define S51		(SIZE_FUDGE_FACTOR * 15)
#define S52		(SIZE_FUDGE_FACTOR * 16)
#define S53		(SIZE_FUDGE_FACTOR * 17)
#define S54		(SIZE_FUDGE_FACTOR * 18)
#define S55		(SIZE_FUDGE_FACTOR * 19)
#define S56		(SIZE_FUDGE_FACTOR * 20)
#define D51		(SIZE_FUDGE_FACTOR * 7)
#define D52		(SIZE_FUDGE_FACTOR * 13)
#define D53		(SIZE_FUDGE_FACTOR * 24)
#define D54		(SIZE_FUDGE_FACTOR * 2)
#define FS51		(15)
#define FS52		(16)
#define FS53		(17)
#define FS54		(18)
#define FS55		(19)
#define FS56		(20)
#define FD51		(7)
#define FD52		(13)
#define FD53		(24)
#define FD54		(2)

#define OFFSET6		(30 - 1)
#define S61		(SIZE_FUDGE_FACTOR * 19)
#define S62		(SIZE_FUDGE_FACTOR * 20)
#define S63		(SIZE_FUDGE_FACTOR * 21)
#define S64		(SIZE_FUDGE_FACTOR * 22)
#define S65		(SIZE_FUDGE_FACTOR * 23)
#define S66		(SIZE_FUDGE_FACTOR * 24)
#define D61		(SIZE_FUDGE_FACTOR * 3)
#define D62		(SIZE_FUDGE_FACTOR * 28)
#define D63		(SIZE_FUDGE_FACTOR * 10)
#define D64		(SIZE_FUDGE_FACTOR * 18)

#define OFFSET7		(36 - 1)
#define S71		(SIZE_FUDGE_FACTOR * 23)
#define S72		(SIZE_FUDGE_FACTOR * 24)
#define S73		(SIZE_FUDGE_FACTOR * 25)
#define S74		(SIZE_FUDGE_FACTOR * 26)
#define S75		(SIZE_FUDGE_FACTOR * 27)
#define S76		(SIZE_FUDGE_FACTOR * 28)
#define D71		(SIZE_FUDGE_FACTOR * 31)
#define D72		(SIZE_FUDGE_FACTOR * 11)
#define D73		(SIZE_FUDGE_FACTOR * 21)
#define D74		(SIZE_FUDGE_FACTOR * 6)

#define OFFSET8		(42 - 1)
#define S81		(SIZE_FUDGE_FACTOR * 27)
#define S82		(SIZE_FUDGE_FACTOR * 28)
#define S83		(SIZE_FUDGE_FACTOR * 29)
#define S84		(SIZE_FUDGE_FACTOR * 30)
#define S85		(SIZE_FUDGE_FACTOR * 31)
#define S86		(SIZE_FUDGE_FACTOR * 0)
#define D81		(SIZE_FUDGE_FACTOR * 4)
#define D82		(SIZE_FUDGE_FACTOR * 26)
#define D83		(SIZE_FUDGE_FACTOR * 14)
#define D84		(SIZE_FUDGE_FACTOR * 20)

extern void s1 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S1(in, offsets) {				\
    INNER_LOOP_SLICE t_a3, t_a5;				\
    INNER_LOOP_SLICE t_k3, t_k4, t_k5, t_k6;			\
    INNER_LOOP_SLICE *t_key3, *t_key4, *t_key5, *t_key6;	\
								\
    t_key3 = offsets->Key_Ptrs[OFFSET1 + 3];	/* PIPELINE */ \
    t_a3 = in[S13];				/* PIPELINE */ \
    t_k3 = t_key3[0];				/* PIPELINE */ \
								\
    t_key5 = offsets->Key_Ptrs[OFFSET1 + 5];	/* PIPELINE */ \
    t_a5 = in[S15];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    t_key4 = offsets->Key_Ptrs[OFFSET1 + 4];	/* PIPELINE */ \
    a4 = in[S14];				/* PIPELINE */ \
    t_k4 = t_key4[0];				/* PIPELINE */ \
								\
    t_key6 = offsets->Key_Ptrs[OFFSET1 + 6];	/* PIPELINE */ \
    a6 = in[S16];				/* PIPELINE */ \
    t_k6 = t_key6[0];				/* PIPELINE */ \
								\
    a3 = t_a3 ^ t_k3;				/* PIPELINE */ \
    a5 = t_a5 ^ t_k5;				/* PIPELINE */ \
								\
    a4 = a4 ^ t_k4;				/* PIPELINE */ \
    a6 = a6 ^ t_k6;				/* PIPELINE */ \
    }

INLINE static void
wrap_s1 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a3, INNER_LOOP_SLICE a5,
    INNER_LOOP_SLICE a4, INNER_LOOP_SLICE a6
) {
    INNER_LOOP_SLICE a1, a2;
    INNER_LOOP_SLICE k1, k2;
    INNER_LOOP_SLICE *key1, *key2;

	key2 = offsets->Key_Ptrs[OFFSET1 + 2];
	a2 = in[S12];
	k2 = key2[0];	

	key1 = offsets->Key_Ptrs[OFFSET1 + 1];
	a1 = in[S11];
	k1 = key1[0];

	a2 = a2 ^ k2;
	a1 = a1 ^ k1;

    s1 ( a1, a2, a3, a4, a5, a6,
	&merge[D11], &merge[D12], &merge[D13], &merge[D14],
	&out[D11], &out[D12], &out[D13], &out[D14]);
/* NOTE could pass a3 and a4 to next function */
}

extern void s2 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S2(in, offsets) {				\
    INNER_LOOP_SLICE t_a1, t_a6;				\
    INNER_LOOP_SLICE t_k1, t_k2, t_k5, t_k6;			\
    INNER_LOOP_SLICE *t_key1, *t_key2, *t_key5, *t_key6;	\
								\
    t_key1 = offsets->Key_Ptrs[OFFSET2 + 1];	/* PIPELINE */ \
    t_a1 = in[S21];				/* PIPELINE */ \
    t_k1 = t_key1[0];				/* PIPELINE */ \
								\
    t_key6 = offsets->Key_Ptrs[OFFSET2 + 6];	/* PIPELINE */ \
    t_a6 = in[S26];				/* PIPELINE */ \
    t_k6 = t_key6[0];				/* PIPELINE */ \
								\
    t_key5 = offsets->Key_Ptrs[OFFSET2 + 5];	/* PIPELINE */ \
    a5 = in[S25];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    t_key2 = offsets->Key_Ptrs[OFFSET2 + 2];	/* PIPELINE */ \
    a2 = in[S22];				/* PIPELINE */ \
    t_k2 = t_key2[0];				/* PIPELINE */ \
								\
    a1 = t_a1 ^ t_k1;				/* PIPELINE */ \
    a6 = t_a6 ^ t_k6;				/* PIPELINE */ \
								\
    a5 = a5 ^ t_k5;				/* PIPELINE */ \
    a2 = a2 ^ t_k2;				/* PIPELINE */ \
    }

INLINE static void
wrap_s2 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a1, INNER_LOOP_SLICE a6,
    INNER_LOOP_SLICE a5, INNER_LOOP_SLICE a2
) {
    INNER_LOOP_SLICE a3, a4;
    INNER_LOOP_SLICE k3, k4;
    INNER_LOOP_SLICE *key3, *key4;

	key3 = offsets->Key_Ptrs[OFFSET2 + 3];
	a3 = in[S23];
	k3 = key3[0];	

	key4 = offsets->Key_Ptrs[OFFSET2 + 4];
	a4 = in[S24];
	k4 = key4[0];

	a3 = a3 ^ k3;
	a4 = a4 ^ k4;

    s2 ( a1, a2, a3, a4, a5, a6,
	&merge[D21], &merge[D22], &merge[D23], &merge[D24],
	&out[D21], &out[D22], &out[D23], &out[D24]);
/* NOTE could pass a7 and a8 to next function */
}

extern void s3 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S3(in, offsets) {				\
    INNER_LOOP_SLICE t_a2, t_a3;				\
    INNER_LOOP_SLICE t_k2, t_k3, t_k5, t_k6;			\
    INNER_LOOP_SLICE *t_key2, *t_key3, *t_key5, *t_key6;	\
								\
    t_key2 = offsets->Key_Ptrs[OFFSET3 + 2];	/* PIPELINE */ \
    t_a2 = in[S32];				/* PIPELINE */ \
    t_k2 = t_key2[0];				/* PIPELINE */ \
								\
    t_key3 = offsets->Key_Ptrs[OFFSET3 + 3];	/* PIPELINE */ \
    t_a3 = in[S33];				/* PIPELINE */ \
    t_k3 = t_key3[0];				/* PIPELINE */ \
								\
    t_key6 = offsets->Key_Ptrs[OFFSET3 + 6];	/* PIPELINE */ \
    a6 = in[S36];				/* PIPELINE */ \
    t_k6 = t_key6[0];				/* PIPELINE */ \
								\
    t_key5 = offsets->Key_Ptrs[OFFSET3 + 5];	/* PIPELINE */ \
    a5 = in[S35];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    a2 = t_a2 ^ t_k2;				/* PIPELINE */ \
    a3 = t_a3 ^ t_k3;				/* PIPELINE */ \
								\
    a6 = a6 ^ t_k6;				/* PIPELINE */ \
    a5 = a5 ^ t_k5;				/* PIPELINE */ \
    }

INLINE static void
wrap_s3 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a2, INNER_LOOP_SLICE a3,
    INNER_LOOP_SLICE a6, INNER_LOOP_SLICE a5
) {
    INNER_LOOP_SLICE a1, a4;
    INNER_LOOP_SLICE k1, k4;
    INNER_LOOP_SLICE *key1, *key4;

	key1 = offsets->Key_Ptrs[OFFSET3 + 1];
	a1 = in[S31];
	k1 = key1[0];	

	key4 = offsets->Key_Ptrs[OFFSET3 + 4];
	a4 = in[S34];
	k4 = key4[0];

	a1 = a1 ^ k1;
	a4 = a4 ^ k4;

    s3 ( a1, a2, a3, a4, a5, a6,
	&merge[D31], &merge[D32], &merge[D33], &merge[D34],
	&out[D31], &out[D32], &out[D33], &out[D34]);
/* NOTE could pass a11 and a12 to next function */
}

extern void s4 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S4(in, offsets) {				\
    INNER_LOOP_SLICE t_a1, t_a3;				\
    INNER_LOOP_SLICE t_k1, t_k2, t_k3, t_k5;			\
    INNER_LOOP_SLICE *t_key1, *t_key2, *t_key3, *t_key5;	\
								\
    t_key1 = offsets->Key_Ptrs[OFFSET4 + 1];	/* PIPELINE */ \
    t_a1 = in[S41];				/* PIPELINE */ \
    t_k1 = t_key1[0];				/* PIPELINE */ \
								\
    t_key3 = offsets->Key_Ptrs[OFFSET4 + 3];	/* PIPELINE */ \
    t_a3 = in[S43];				/* PIPELINE */ \
    t_k3 = t_key3[0];				/* PIPELINE */ \
								\
    t_key5 = offsets->Key_Ptrs[OFFSET4 + 5];	/* PIPELINE */ \
    a5 = in[S45];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    t_key2 = offsets->Key_Ptrs[OFFSET4 + 2];	/* PIPELINE */ \
    a2 = in[S42];				/* PIPELINE */ \
    t_k2 = t_key2[0];				/* PIPELINE */ \
								\
    a1 = t_a1 ^ t_k1;				/* PIPELINE */ \
    a3 = t_a3 ^ t_k3;				/* PIPELINE */ \
								\
    a5 = a5 ^ t_k5;				/* PIPELINE */ \
    a2 = a2 ^ t_k2;				/* PIPELINE */ \
    }

INLINE static void
wrap_s4 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a1, INNER_LOOP_SLICE a3,
    INNER_LOOP_SLICE a5, INNER_LOOP_SLICE a2
) {
    INNER_LOOP_SLICE a4, a6;
    INNER_LOOP_SLICE k4, k6;
    INNER_LOOP_SLICE *key4, *key6;

	key4 = offsets->Key_Ptrs[OFFSET4 + 4];
	a4 = in[S44];
	k4 = key4[0];	

	key6 = offsets->Key_Ptrs[OFFSET4 + 6];
	a6 = in[S46];
	k6 = key6[0];

	a4 = a4 ^ k4;
	a6 = a6 ^ k6;

    s4 ( a1, a2, a3, a4, a5, a6,
	&merge[D41], &merge[D42], &merge[D43], &merge[D44],
	&out[D41], &out[D42], &out[D43], &out[D44]);
/* NOTE could pass a15 and a16 to next function */
}

extern void s5 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S5(in, offsets) {				\
    INNER_LOOP_SLICE t_a3, t_a4;				\
    INNER_LOOP_SLICE t_k1, t_k3, t_k4, t_k6;			\
    INNER_LOOP_SLICE *t_key1, *t_key3, *t_key4, *t_key6;	\
								\
    t_key3 = offsets->Key_Ptrs[OFFSET5 + 3];	/* PIPELINE */ \
    t_a3 = in[S53];				/* PIPELINE */ \
    t_k3 = t_key3[0];				/* PIPELINE */ \
								\
    t_key4 = offsets->Key_Ptrs[OFFSET5 + 4];	/* PIPELINE */ \
    t_a4 = in[S54];				/* PIPELINE */ \
    t_k4 = t_key4[0];				/* PIPELINE */ \
								\
    t_key1 = offsets->Key_Ptrs[OFFSET5 + 1];	/* PIPELINE */ \
    a1 = in[S51];				/* PIPELINE */ \
    t_k1 = t_key1[0];				/* PIPELINE */ \
								\
    t_key6 = offsets->Key_Ptrs[OFFSET5 + 6];	/* PIPELINE */ \
    a6 = in[S56];				/* PIPELINE */ \
    t_k6 = t_key6[0];				/* PIPELINE */ \
								\
    a3 = t_a3 ^ t_k3;				/* PIPELINE */ \
    a4 = t_a4 ^ t_k4;				/* PIPELINE */ \
								\
    a1 = a1 ^ t_k1;				/* PIPELINE */ \
    a6 = a6 ^ t_k6;				/* PIPELINE */ \
    }

INLINE static void
wrap_s5 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a3, INNER_LOOP_SLICE a4,
    INNER_LOOP_SLICE a1, INNER_LOOP_SLICE a6
) {
    INNER_LOOP_SLICE a2, a5;
    INNER_LOOP_SLICE k2, k5;
    INNER_LOOP_SLICE *key2, *key5;

	key5 = offsets->Key_Ptrs[OFFSET5 + 5];
	a5 = in[S55];
	k5 = key5[0];	

	key2 = offsets->Key_Ptrs[OFFSET5 + 2];
	a2 = in[S52];
	k2 = key2[0];

	a5 = a5 ^ k5;
	a2 = a2 ^ k2;

    s5 ( a1, a2, a3, a4, a5, a6,
	&merge[D51], &merge[D52], &merge[D53], &merge[D54],
	&out[D51], &out[D52], &out[D53], &out[D54]);
/* NOTE could pass a19 and a20 to next function */
}

extern void s6 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S6(in, offsets) {				\
    INNER_LOOP_SLICE t_a1, t_a5;				\
    INNER_LOOP_SLICE t_k1, t_k4, t_k5, t_k6;			\
    INNER_LOOP_SLICE *t_key1, *t_key4, *t_key5, *t_key6;	\
								\
    t_key5 = offsets->Key_Ptrs[OFFSET6 + 5];	/* PIPELINE */ \
    t_a5 = in[S65];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    t_key1 = offsets->Key_Ptrs[OFFSET6 + 1];	/* PIPELINE */ \
    t_a1 = in[S61];				/* PIPELINE */ \
    t_k1 = t_key1[0];				/* PIPELINE */ \
								\
    t_key6 = offsets->Key_Ptrs[OFFSET6 + 6];	/* PIPELINE */ \
    a6 = in[S66];				/* PIPELINE */ \
    t_k6 = t_key6[0];				/* PIPELINE */ \
								\
    t_key4 = offsets->Key_Ptrs[OFFSET6 + 4];	/* PIPELINE */ \
    a4 = in[S64];				/* PIPELINE */ \
    t_k4 = t_key4[0];				/* PIPELINE */ \
								\
    a5 = t_a5 ^ t_k5;				/* PIPELINE */ \
    a1 = t_a1 ^ t_k1;				/* PIPELINE */ \
								\
    a6 = a6 ^ t_k6;				/* PIPELINE */ \
    a4 = a4 ^ t_k4;				/* PIPELINE */ \
    }

INLINE static void
wrap_s6 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a5, INNER_LOOP_SLICE a1,
    INNER_LOOP_SLICE a6, INNER_LOOP_SLICE a4
) {
    INNER_LOOP_SLICE a2, a3;
    INNER_LOOP_SLICE k2, k3;
    INNER_LOOP_SLICE *key2, *key3;

	key2 = offsets->Key_Ptrs[OFFSET6 + 2];
	a2 = in[S62];
	k2 = key2[0];	

	key3 = offsets->Key_Ptrs[OFFSET6 + 3];
	a3 = in[S63];
	k3 = key3[0];

	a2 = a2 ^ k2;
	a3 = a3 ^ k3;

    s6 ( a1, a2, a3, a4, a5, a6,
	&merge[D61], &merge[D62], &merge[D63], &merge[D64],
	&out[D61], &out[D62], &out[D63], &out[D64]);
/* NOTE could pass a23 and a24 to next function */
}

extern void s7 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S7(in, offsets) {				\
    INNER_LOOP_SLICE t_a2, t_a4;				\
    INNER_LOOP_SLICE t_k2, t_k3, t_k4, t_k5;			\
    INNER_LOOP_SLICE *t_key2, *t_key3, *t_key4, *t_key5;	\
								\
    t_key2 = offsets->Key_Ptrs[OFFSET7 + 2];	/* PIPELINE */ \
    t_a2 = in[S72];				/* PIPELINE */ \
    t_k2 = t_key2[0];				/* PIPELINE */ \
								\
    t_key4 = offsets->Key_Ptrs[OFFSET7 + 4];	/* PIPELINE */ \
    t_a4 = in[S74];				/* PIPELINE */ \
    t_k4 = t_key4[0];				/* PIPELINE */ \
								\
    t_key5 = offsets->Key_Ptrs[OFFSET7 + 5];	/* PIPELINE */ \
    a5 = in[S75];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    t_key3 = offsets->Key_Ptrs[OFFSET7 + 3];	/* PIPELINE */ \
    a3 = in[S73];				/* PIPELINE */ \
    t_k3 = t_key3[0];				/* PIPELINE */ \
								\
    a2 = t_a2 ^ t_k2;				/* PIPELINE */ \
    a4 = t_a4 ^ t_k4;				/* PIPELINE */ \
								\
    a5 = a5 ^ t_k5;				/* PIPELINE */ \
    a3 = a3 ^ t_k3;				/* PIPELINE */ \
    }

INLINE static void
wrap_s7 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a2, INNER_LOOP_SLICE a4,
    INNER_LOOP_SLICE a5, INNER_LOOP_SLICE a3
) {
    INNER_LOOP_SLICE a1, a6;
    INNER_LOOP_SLICE k1, k6;
    INNER_LOOP_SLICE *key1, *key6;

	key6 = offsets->Key_Ptrs[OFFSET7 + 6];
	a6 = in[S76];
	k6 = key6[0];	

	key1 = offsets->Key_Ptrs[OFFSET7 + 1];
	a1 = in[S71];
	k1 = key1[0];

	a6 = a6 ^ k6;
	a1 = a1 ^ k1;

    s7 ( a1, a2, a3, a4, a5, a6,
	&merge[D71], &merge[D72], &merge[D73], &merge[D74],
	&out[D71], &out[D72], &out[D73], &out[D74]);
/* NOTE could pass a27 and a28 to next function */
}

extern void s8 ( KWAN_LOOP_SLICE a1, KWAN_LOOP_SLICE a2,
	KWAN_LOOP_SLICE a3, KWAN_LOOP_SLICE a4,
	KWAN_LOOP_SLICE a5, KWAN_LOOP_SLICE a6,
	KWAN_LOOP_SLICE *merge1, KWAN_LOOP_SLICE *merge2,
	KWAN_LOOP_SLICE *merge3, KWAN_LOOP_SLICE *merge4,
	KWAN_LOOP_SLICE *out1, KWAN_LOOP_SLICE *out2,
	KWAN_LOOP_SLICE *out3, KWAN_LOOP_SLICE *out4 );

#define PIPELINE_S8(in, offsets) {				\
    INNER_LOOP_SLICE t_a1, t_a3;				\
    INNER_LOOP_SLICE t_k1, t_k3, t_k4, t_k5;			\
    INNER_LOOP_SLICE *t_key1, *t_key3, *t_key4, *t_key5;	\
								\
    t_key3 = offsets->Key_Ptrs[OFFSET8 + 3];	/* PIPELINE */ \
    t_a3 = in[S83];				/* PIPELINE */ \
    t_k3 = t_key3[0];				/* PIPELINE */ \
								\
    t_key1 = offsets->Key_Ptrs[OFFSET8 + 1];	/* PIPELINE */ \
    t_a1 = in[S81];				/* PIPELINE */ \
    t_k1 = t_key1[0];				/* PIPELINE */ \
								\
    t_key4 = offsets->Key_Ptrs[OFFSET8 + 4];	/* PIPELINE */ \
    a4 = in[S84];				/* PIPELINE */ \
    t_k4 = t_key4[0];				/* PIPELINE */ \
								\
    t_key5 = offsets->Key_Ptrs[OFFSET8 + 5];	/* PIPELINE */ \
    a5 = in[S85];				/* PIPELINE */ \
    t_k5 = t_key5[0];				/* PIPELINE */ \
								\
    a3 = t_a3 ^ t_k3;				/* PIPELINE */ \
    a1 = t_a1 ^ t_k1;				/* PIPELINE */ \
								\
    a4 = a4 ^ t_k4;				/* PIPELINE */ \
    a5 = a5 ^ t_k5;				/* PIPELINE */ \
    }

INLINE static void
wrap_s8 ( INNER_LOOP_SLICE *in, INNER_LOOP_SLICE *merge,
    INNER_LOOP_SLICE *out, struct INNER_OFFSET_DISTANCES *offsets,
    INNER_LOOP_SLICE a3, INNER_LOOP_SLICE a1,
    INNER_LOOP_SLICE a4, INNER_LOOP_SLICE a5
) {
    INNER_LOOP_SLICE a2, a6;
    INNER_LOOP_SLICE k2, k6;
    INNER_LOOP_SLICE *key2, *key6;

	key2 = offsets->Key_Ptrs[OFFSET8 + 2];
	a2 = in[S82];
	k2 = key2[0];	

	key6 = offsets->Key_Ptrs[OFFSET8 + 6];
	a6 = in[S86];
	k6 = key6[0];

	a2 = a2 ^ k2;
	a6 = a6 ^ k6;

    s8 ( a1, a2, a3, a4, a5, a6,
	&merge[D81], &merge[D82], &merge[D83], &merge[D84],
	&out[D81], &out[D82], &out[D83], &out[D84]);
/* NOTE could pass a23 and a24 to next function */
}

/* end of des.h */
