/* des-ultra-crunch.c v4.0 */

/*
 * $Log: des_ultra_crunch.c,v $
 * Revision 1.3  1998/06/14 15:18:32  remi
 * Avoid tons of warnings due to a brain-dead CVS.
 *
 * Revision 1.2  1998/06/14 14:32:23  remi
 * Added the right key to the list of internal test-cases.
 *
 * Revision 1.1.1.1  1998/06/14 14:23:49  remi
 * Initial integration.
 *
 */

static char *id="@(#)$Id: des_ultra_crunch.c,v 1.3 1998/06/14 15:18:32 remi Exp $";

/* defines are set in configure & Makefile when compiling DCTI client */
#ifndef IN_DCTI_CLIENT
#include "s_paramaters.h"
#endif

#ifdef ASM	/* too lazy to double code size.  Just clobber routine calls */
#define DO_S1		asm_do_s1
#define DO_S2		asm_do_s2
#define DO_S3		asm_do_s3
#define DO_S4		asm_do_s4
#define DO_S5		asm_do_s5
#define DO_S6		asm_do_s6
#define DO_S7		asm_do_s7
#define DO_S8		asm_do_s8
#define DO_S1_S3	asm_do_s1_s3
#define DO_ALL		asm_do_all
#define DO_ALL_FANCY	asm_do_all_fancy
#else /* ASM */
#define DO_S1		do_s1
#define DO_S2		do_s2
#define DO_S3		do_s3
#define DO_S4		do_s4
#define DO_S5		do_s5
#define DO_S6		do_s6
#define DO_S7		do_s7
#define DO_S8		do_s8
#define DO_S1_S3	do_s1_s3
#define DO_ALL		do_all
#define DO_ALL_FANCY	do_all
#endif /* ASM */

typedef unsigned long long OUTER_LOOP_SLICE;	/* assuming 64-bit work done here */
#define SLICE_0	 ((OUTER_LOOP_SLICE) 0ull)
#define SLICE_1 (~((OUTER_LOOP_SLICE) 0ull))

/* comment this out to use 64-bit operands but calculate only 32 valid bits at once */

#ifdef FULL_64_BIT_VALID
    #define ULTRA_BIT_64			/* only if all 64 bits are valid */
    #define REQUIRED_SIZE 8
    #define REQUIRED_STRIDE 8
#else /* not FULL_64_BIT_VALID, so only 32 bits in MSB of each 64 are valid */
    #define NO_ULTRA_BIT_64			/* only if all 64 bits are valid */
    #define REQUIRED_SIZE 4
    #define REQUIRED_STRIDE 8
#endif /* FULL_64_BIT_VALID */

#define stat_printf(a)	/* printf (a) */

/* this struct must agree in size with the do_all_fancy routine */
struct OUTER_OFFSET_DISTANCES
    {	long Operand_Size;
	long Operand_Stride;
	OUTER_LOOP_SLICE * Key_Ptrs[48];	/* pointers to pick up 48 keys */
	OUTER_LOOP_SLICE * Source;
	OUTER_LOOP_SLICE * Merge;
	OUTER_LOOP_SLICE * Dest;
	struct OUTER_OFFSET_DISTANCES * Next_Offsets;
	OUTER_LOOP_SLICE * Next_Source;
    };

/* forward declaration */
extern unsigned long DO_S1 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S2 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S3 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S4 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S5 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S6 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S7 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S8 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_S1_S3 ( struct OUTER_OFFSET_DISTANCES *Offset_List );
extern unsigned long DO_ALL (
	struct OUTER_OFFSET_DISTANCES *Offset_List, int Special_Case );
extern unsigned long DO_ALL_FANCY (
	struct OUTER_OFFSET_DISTANCES *Offset_List, int Special_Case );

OUTER_LOOP_SLICE whack16 (OUTER_LOOP_SLICE *plain_base,
	OUTER_LOOP_SLICE *cypher_base, OUTER_LOOP_SLICE *key_base);

/* work routines are all of the form:
 *     in = Offset_List->Source;
 *     merge = Offset_List->Merge;
 *     out = Offset_List->Dest;
 *  {  pntr = Offset_List->Key_Ptrs[blah];
 *     key = pntr[0];	// 6 times
 *     a = in[blah];    // 6 times
 *     a = a ^ key;     // 6 times
 *     lots of stuff producing result
 *     out[blah] = merge[blah] * result;  // 4 times
 *     Offset_List = Offset_List->Next_Offsets;
 *     in = Offset_List->Source;
 *     merge = Offset_List->Merge;
 *     out = Offset_List->Dest;
 *  }  while (merge != (OUTER_OFFSET_DISTANCES *)0);
 */

/* This code is based on work done by Kwan and Meggs
 * Two smart cookies.
 * Also dej@inode.org and rguyom@mail.dotcom.fr worked here
 * http://www/cs/mu/oz/au/~mkwan
 */

#include <stdio.h>
#include <stdlib.h>

static int Round1_Key[] =	/* applies against the Right data */
    {	47, 11, 26,  3, 13, 41,
	27,  6, 54, 48, 39, 19,
	53, 25, 33, 34, 17,  5,
	 4, 55, 24, 32, 40, 20,
	36, 31, 21,  8, 23, 52,
	14, 29, 51,  9, 35, 30,
	 2, 37, 22,  0, 42, 38,
	16, 43, 44,  1,  7, 28,
    };

static int Round2_Key[] =	/* applies against the Left data */
    {	54, 18, 33, 10, 20, 48,
	34, 13,  4, 55, 46, 26,
	 3, 32, 40, 41, 24, 12,
	11,  5,  6, 39, 47, 27,
	43, 38, 28, 15, 30,  0,
	21, 36, 31, 16, 42, 37,
	 9, 44, 29,  7, 49, 45,
	23, 50, 51,  8, 14, 35,
    };

static int Round3_Key[] =	/* applies against the Right data */
    {	11, 32, 47, 24, 34,  5,
	48, 27, 18, 12,  3, 40,
	17, 46, 54, 55, 13, 26,
	25, 19, 20, 53,  4, 41,
	 2, 52, 42, 29, 44, 14,
	35, 50, 45, 30,  1, 51,
	23, 31, 43, 21,  8,  0,
	37,  9, 38, 22, 28, 49,
    };

static int Round4_Key[] =	/* applies against the Left data */
    {	25, 46,  4, 13, 48, 19,
	 5, 41, 32, 26, 17, 54,
	 6,  3, 11, 12, 27, 40,
	39, 33, 34, 10, 18, 55,
	16,  7,  1, 43, 31, 28,
	49,  9,  0, 44, 15, 38,
	37, 45,  2, 35, 22, 14,
	51, 23, 52, 36, 42,  8,
    };

static int Round5_Key[] =	/* applies against the Right data */
    {	39,  3, 18, 27,  5, 33,
	19, 55, 46, 40,  6, 11,
	20, 17, 25, 26, 41, 54,
	53, 47, 48, 24, 32, 12,
	30, 21, 15,  2, 45, 42,
	 8, 23, 14, 31, 29, 52,
	51,  0, 16, 49, 36, 28,
	38, 37,  7, 50,  1, 22,
    };

static int Round6_Key[] =	/* applies against the Left data */
    {	53, 17, 32, 41, 19, 47,
	33, 12,  3, 54, 20, 25,
	34,  6, 39, 40, 55, 11,
	10,  4,  5, 13, 46, 26,
	44, 35, 29, 16,  0,  1,
	22, 37, 28, 45, 43,  7,
	38, 14, 30,  8, 50, 42,
	52, 51, 21,  9, 15, 36,
    };

static int Round7_Key[] =	/* applies against the Right data */
    {	10,  6, 46, 55, 33,  4,
	47, 26, 17, 11, 34, 39,
	48, 20, 53, 54, 12, 25,
	24, 18, 19, 27,  3, 40,
	31, 49, 43, 30, 14, 15,
	36, 51, 42,  0,  2, 21,
	52, 28, 44, 22,  9,  1,
	 7, 38, 35, 23, 29, 50,
    };

static int Round8_Key[] =	/* applies against the Left data */
    {	24, 20,  3, 12, 47, 18,
	 4, 40,  6, 25, 48, 53,
	 5, 34, 10, 11, 26, 39,
	13, 32, 33, 41, 17, 54,
	45,  8,  2, 44, 28, 29,
	50, 38,  1, 14, 16, 35,
	 7, 42, 31, 36, 23, 15,
	21, 52, 49, 37, 43,  9,
    };

static int Round9_Key[] =	/* applies against the Right data */
    {	 6, 27, 10, 19, 54, 25,
	11, 47, 13, 32, 55,  3,
	12, 41, 17, 18, 33, 46,
	20, 39, 40, 48, 24,  4,
	52, 15,  9, 51, 35, 36,
	 2, 45,  8, 21, 23, 42,
	14, 49, 38, 43, 30, 22,
	28,  0,  1, 44, 50, 16,
    };

static int Round10_Key[] =	/* applies against the Left data */
    {	20, 41, 24, 33, 11, 39,
	25,  4, 27, 46, 12, 17,
	26, 55,  6, 32, 47,  3,
	34, 53, 54,  5, 13, 18,
	 7, 29, 23, 38, 49, 50,
	16,  0, 22, 35, 37,  1,
	28,  8, 52,  2, 44, 36,
	42, 14, 15, 31,  9, 30,
    };

static int Round11_Key[] =	/* applies against the Right data */
    {	34, 55, 13, 47, 25, 53,
	39, 18, 41,  3, 26,  6,
	40, 12, 20, 46,  4, 17,
	48, 10, 11, 19, 27, 32,
	21, 43, 37, 52,  8,  9,
	30, 14, 36, 49, 51, 15,
	42, 22,  7, 16, 31, 50,
	 1, 28, 29, 45, 23, 44,
    };

static int Round12_Key[] =	/* applies against the Left data */
    {	48, 12, 27,  4, 39, 10,
	53, 32, 55, 17, 40, 20,
	54, 26, 34,  3, 18,  6,
	 5, 24, 25, 33, 41, 46,
	35,  2, 51,  7, 22, 23,
	44, 28, 50,  8, 38, 29,
	 1, 36, 21, 30, 45,  9,
	15, 42, 43,  0, 37, 31,
    };

static int Round13_Key[] =	/* applies against the Right data */
    {	 5, 26, 41, 18, 53, 24,
	10, 46, 12,  6, 54, 34,
	11, 40, 48, 17, 32, 20,
	19, 13, 39, 47, 55,  3,
	49, 16, 38, 21, 36, 37,
	31, 42,  9, 22, 52, 43,
	15, 50, 35, 44,  0, 23,
	29,  1,  2, 14, 51, 45,
    };

/* independent of key bits [18, 21, 22, 39, 41, 42, 44, 47] */
static int Round14_Key[] =	/* applies against the Left data */
    {	19, 40, 55, 32, 10, 13,
	24,  3, 26, 20, 11, 48, /* 3, 11 */ /* 5 */ /* plan to vary this key pair */
	25, 54,  5,  6, 46, 34, /* 5 */
	33, 27, 53,  4, 12, 17,             /* 15 */
	 8, 30, 52, 35, 50, 51, /* 8 */     /* 15 */
	45,  1, 23, 36,  7,  2,             /* 23 */
	29,  9, 49, 31, 14, 37,             /* 23 */
	43, 15, 16, 28, 38,  0, /* 43 */    /* 29 */
    };				/* keys */  /* round 15 funct 3 */

/* independent of key bits [1, 4, 31, 32, 35, 36, 53, 55] */
static int Round15_Key[] =	/* applies against the Right data */
    {	33, 54, 12, 46, 24, 27,
	13, 17, 40, 34, 25,  5, /* 5 */
	39, 11, 19, 20,  3, 48, /* 3, 11 */ /* plan to vary this key pair */
	47, 41, 10, 18, 26,  6,
	22, 44,  7, 49,  9, 38,
	 0, 15, 37, 50, 21, 16,
	43, 23,  8, 45, 28, 51, /* 8, 43 */
	 2, 29, 30, 42, 52, 14, /* 42 */
    };				/* keys */

/* independent of key bits [3, 5, 8, 11, 42, 43, 38, 39] */
static int Round16_Key[] =	/* applies against the Left data */
    {	40,  4, 19, 53,  6, 34,
	20, 24, 47, 41, 32, 12,
	46, 18, 26, 27, 10, 55,
	54, 48, 17, 25, 33, 13,
	29, 51, 14,  1, 16, 45,
	 7, 22, 44,  2, 28, 23,
	50, 30, 15, 52, 35, 31,
	 9, 36, 37, 49,  0, 21,
    };

/* always read Source argument in this order: */
static int Source_Left_Right[] =
    {	31,  0,  1,  2,  3,  4,
	 3,  4,  5,  6,  7,  8,
	 7,  8,  9, 10, 11, 12,
	11, 12, 13, 14, 15, 16,
	15, 16, 17, 18, 19, 20,
	19, 20, 21, 22, 23, 24,
	23, 24, 25, 26, 27, 28,
	27, 28, 29, 30, 31,  0,
    };

/* always write Destination argument in this order: */
static int Dest_Left_Right[] =
    {	 8, 16, 22, 30,
	12, 27,  1, 17,
	23, 15, 29,  5,
	25, 19,  9,  0,
	 7, 13, 24,  2,
	 3, 28, 10, 18,
	31, 11, 21,  6,
	 4, 26, 14, 20,
    };

static int Left_Input_Permutation[] =
    {	6, 14, 22, 30, 38, 46, 54, 62,
	4, 12, 20, 28, 36, 44, 52, 60,
	2, 10, 18, 26, 34, 42, 50, 58,
	0,  8, 16, 24, 32, 40, 48, 56,
    };

static int Right_Input_Permutation[] =
    {	7, 15, 23, 31, 39, 47, 55, 63,
	5, 13, 21, 29, 37, 45, 53, 61,
	3, 11, 19, 27, 35, 43, 51, 59,
	1,  9, 17, 25, 33, 41, 49, 57,
    };

static int Left_Output_Permutation[] =
    {	 7, 15, 23, 31, 39, 47, 55, 63,
	 5, 13, 21, 29, 37, 45, 53, 61,
	 3, 11, 19, 27, 35, 43, 51, 59,
	 1,  9, 17, 25, 33, 41, 49, 57,
    };

static int Right_Output_Permutation[] =
    {	 6, 14, 22, 30, 38, 46, 54, 62,
	 4, 12, 20, 28, 36, 44, 52, 60,
	 2, 10, 18, 26, 34, 42, 50, 58,
	 0,  8, 16, 24, 32, 40, 48, 56,
    };

struct KEYS
    {	OUTER_LOOP_SLICE Keys[56];
    };

struct SLICE_ARRAY
    {	OUTER_LOOP_SLICE Data[32];
    };

struct CACHED_RESULTS_ARRAY
    {	OUTER_LOOP_SLICE Cached_Data_8;
	OUTER_LOOP_SLICE Cached_Data_16;
	OUTER_LOOP_SLICE Cached_Data_22;
	OUTER_LOOP_SLICE Cached_Data_30;
	OUTER_LOOP_SLICE Cached_Data_5;
	OUTER_LOOP_SLICE Cached_Data_15;
	OUTER_LOOP_SLICE Cached_Data_23;
	OUTER_LOOP_SLICE Cached_Data_29;
    };

/* The all-important data structure initialization routine.
 * Since this code lives or dies based on the contents of it's
 * linked lists of data to transform, it is important to get
 * these lists initialized correctly.
 */
static void
work_record_initialize ( int Pass_Number,
    struct OUTER_OFFSET_DISTANCES *Offset_Element,
    OUTER_LOOP_SLICE *keys, OUTER_LOOP_SLICE *in,
    OUTER_LOOP_SLICE *merge, OUTER_LOOP_SLICE *out
) {
    register int i;
    register int *Key_Index;
/* put in the in, merge, out pointers */
    Offset_Element->Operand_Size = REQUIRED_SIZE;
    Offset_Element->Operand_Stride = REQUIRED_STRIDE;
    Offset_Element->Source = in;
    Offset_Element->Merge = merge;
    Offset_Element->Dest = out;
    Offset_Element->Next_Offsets = Offset_Element;	/* self-link */
    Offset_Element->Next_Source = in;

/* fill in the key entries based on the Pass Number */
    switch (Pass_Number)
    {	case 1:		Key_Index = &Round1_Key[0];	break;
	case 2:		Key_Index = &Round2_Key[0];	break;
	case 3:		Key_Index = &Round3_Key[0];	break;
	case 4:		Key_Index = &Round4_Key[0];	break;
	case 5:		Key_Index = &Round5_Key[0];	break;
	case 6:		Key_Index = &Round6_Key[0];	break;
	case 7:		Key_Index = &Round7_Key[0];	break;
	case 8:		Key_Index = &Round8_Key[0];	break;
	case 9:		Key_Index = &Round9_Key[0];	break;
	case 10:	Key_Index = &Round10_Key[0];	break;
	case 11:	Key_Index = &Round11_Key[0];	break;
	case 12:	Key_Index = &Round12_Key[0];	break;
	case 13:	Key_Index = &Round13_Key[0];	break;
	case 14:	Key_Index = &Round14_Key[0];	break;
	case 15:	Key_Index = &Round15_Key[0];	break;
	case 16:	Key_Index = &Round16_Key[0];	break;
    }
    for (i = 0; i < 48; i++)
    {	Offset_Element->Key_Ptrs[i] = &keys[Key_Index[i]];
    }
}

void
get_permuted_data (struct KEYS * keys,
	struct SLICE_ARRAY * left, struct SLICE_ARRAY * right,
	OUTER_LOOP_SLICE *plain_base, OUTER_LOOP_SLICE *key_base
){  register int i;
    for (i = 0; i < 32; i++)
	left->Data[i] = plain_base[Left_Input_Permutation[i]];

    for (i = 0; i < 32; i++)
	right->Data[i] = plain_base[Right_Input_Permutation[i]];

    for (i = 0; i < 56; i++)
	keys->Keys[i] = key_base[i];
}

void
put_permuted_data (struct SLICE_ARRAY * left, struct SLICE_ARRAY * right,
	OUTER_LOOP_SLICE *cypher_base
){  register int i;
    for (i = 0; i < 32; i++)
	cypher_base[Left_Output_Permutation[i]] = left->Data[i];

    for (i = 0; i < 32; i++)
	cypher_base[Right_Output_Permutation[i]] = right->Data[i];
}

struct Benchmark_Work
    {	struct KEYS Keys;
	struct SLICE_ARRAY Left0;
	struct SLICE_ARRAY Right0;
	struct SLICE_ARRAY Right1;
	struct SLICE_ARRAY Right2;
	struct SLICE_ARRAY Right3;
	struct SLICE_ARRAY Right4;
	struct SLICE_ARRAY Right5;
	struct SLICE_ARRAY Right6;
	struct OUTER_OFFSET_DISTANCES pass_work[16];
	struct OUTER_OFFSET_DISTANCES work_sentinel;
    };

/* this models the data activity of the next routine, although it would
 * be possible to reuse storage here to some advantage
 */

unsigned long
test_benchmark (OUTER_LOOP_SLICE *key_base, OUTER_LOOP_SLICE *plain_base,
					    OUTER_LOOP_SLICE *cypher_base
) {
    register int i;
    struct Benchmark_Work work_struct;
    register struct Benchmark_Work * work = &work_struct;
    unsigned long return_val;

/* malloc a single structure, consisting of key data, working data, and
 * then a bunch of work elements.  By putting these into a struct, we
 * can be sure that they are side-by-side in memory.
 */

/* create some linked-lists */
    work = (struct Benchmark_Work *)
	memalign (64, sizeof (struct Benchmark_Work));

/* make sure sentinel pointer indicates end of list */
    work_record_initialize (1, &work->work_sentinel, &work->Keys.Keys[0],
	&work->Right6.Data[0], (OUTER_LOOP_SLICE *)0, &work->Right1.Data[0]);

/* initialize offset pointers, to be ready to encrypt */
/* pass 1 Benchmark results will be constant as keys are incremented 256 times */
    work_record_initialize (1, &work->pass_work[0], &work->Keys.Keys[0],
	&work->Right0.Data[0], &work->Left0.Data[0], &work->Right1.Data[0]);

/* pass 2 Benchmark results will change 1 sbox every key increment */
    work_record_initialize (2, &work->pass_work[1], &work->Keys.Keys[0],
	&work->Right1.Data[0], &work->Right0.Data[0], &work->Right2.Data[0]);

/* pass 3 Benchmark results will change 6 sboxes every key increment */
    work_record_initialize (3, &work->pass_work[2], &work->Keys.Keys[0],
	&work->Right2.Data[0], &work->Right1.Data[0], &work->Right3.Data[0]);

/* pass 4 Benchmark results are totally new each pass */
    work_record_initialize (4, &work->pass_work[3], &work->Keys.Keys[0],
	&work->Right3.Data[0], &work->Right2.Data[0], &work->Right4.Data[0]);

/* pass 5 Benchmark results are totally new each pass */
    work_record_initialize (5, &work->pass_work[4], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right3.Data[0], &work->Right5.Data[0]);

/* pass 6 Benchmark results are totally new each pass */
    work_record_initialize (6, &work->pass_work[5], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 7 Benchmark results are totally new each pass */
    work_record_initialize (7, &work->pass_work[6], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 8 Benchmark results are totally new each pass */
    work_record_initialize (8, &work->pass_work[7], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 9 Benchmark results are totally new each pass */
    work_record_initialize (9, &work->pass_work[8], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 10 Benchmark results are totally new each pass */
    work_record_initialize (10, &work->pass_work[9], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 11 Benchmark results are totally new each pass */
    work_record_initialize (11, &work->pass_work[10], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 12 Benchmark results are totally new each pass */
    work_record_initialize (12, &work->pass_work[11], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 13 Benchmark results are totally new each pass */
    work_record_initialize (13, &work->pass_work[12], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 14 Benchmark results are totally new each pass */
    work_record_initialize (14, &work->pass_work[13], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 15 Benchmark results are totally new each pass */
    work_record_initialize (15, &work->pass_work[14], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 16 Benchmark results are totally new each pass */
    work_record_initialize (16, &work->pass_work[15], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* link the lists */
    for (i = 0; i < 15; i++)
    {	work->pass_work[i].Next_Offsets = &work->pass_work[i + 1];
	work->pass_work[i].Next_Source = work->pass_work[i + 1].Source;
    }

    work->pass_work[15].Next_Offsets = &work->work_sentinel;
    work->pass_work[15].Next_Source = work->pass_work[15].Source;

/* pick up plain_text data and the keys */
    get_permuted_data (&work->Keys, &work->Left0, &work->Right0,
						plain_base, key_base);
#ifdef TEST_BENCHMARK_RESULTS
#else
/* Meggs implies that the benchmark does 11 passes on average.  Emulate */
    work->pass_work[10].Next_Offsets = &work->work_sentinel;
#endif

    return_val = DO_ALL (&work->pass_work[0], 0);

/* put back cypher_text data */
    put_permuted_data (&work->Right6, &work->Right4, cypher_base);

    free ((void *)work);
    return (return_val);
}

unsigned long
test_unbenchmark ( OUTER_LOOP_SLICE *key_base, OUTER_LOOP_SLICE *plain_base,
					       OUTER_LOOP_SLICE *cypher_base
) {
    register int i;
    struct Benchmark_Work work_struct;
    register struct Benchmark_Work * work = &work_struct;
    unsigned long return_val;

/* malloc a single structure, consisting of key data, working data, and
 * then a bunch of work elements.  By putting these into a struct, we
 * can be sure that they are side-by-side in memory.
 */

/* create some linked-lists */
    work = (struct Benchmark_Work *)
	memalign (64, sizeof (struct Benchmark_Work));

/* make sure sentinel pointer indicates end of list */
    work_record_initialize (1, &work->work_sentinel, &work->Keys.Keys[0],
	&work->Right6.Data[0], (OUTER_LOOP_SLICE *)0, &work->Right1.Data[0]);

/* initialize offset pointers, to be ready to encrypt */
/* pass 1 Benchmark results will be constant as keys are incremented 256 times */
    work_record_initialize (16, &work->pass_work[0], &work->Keys.Keys[0],
	&work->Left0.Data[0], &work->Right0.Data[0], &work->Right1.Data[0]);

/* pass 2 Benchmark results will change 1 sbox every key increment */
    work_record_initialize (15, &work->pass_work[1], &work->Keys.Keys[0],
	&work->Right1.Data[0], &work->Left0.Data[0], &work->Right2.Data[0]);

/* pass 3 Benchmark results will change 6 sboxes every key increment */
    work_record_initialize (14, &work->pass_work[2], &work->Keys.Keys[0],
	&work->Right2.Data[0], &work->Right1.Data[0], &work->Right3.Data[0]);

/* pass 4 Benchmark results are totally new each pass */
    work_record_initialize (13, &work->pass_work[3], &work->Keys.Keys[0],
	&work->Right3.Data[0], &work->Right2.Data[0], &work->Right4.Data[0]);

/* pass 5 Benchmark results are totally new each pass */
    work_record_initialize (12, &work->pass_work[4], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right3.Data[0], &work->Right5.Data[0]);

/* pass 6 Benchmark results are totally new each pass */
    work_record_initialize (11, &work->pass_work[5], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 7 Benchmark results are totally new each pass */
    work_record_initialize (10, &work->pass_work[6], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 8 Benchmark results are totally new each pass */
    work_record_initialize (9, &work->pass_work[7], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 9 Benchmark results are totally new each pass */
    work_record_initialize (8, &work->pass_work[8], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 10 Benchmark results are totally new each pass */
    work_record_initialize (7, &work->pass_work[9], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 11 Benchmark results are totally new each pass */
    work_record_initialize (6, &work->pass_work[10], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 12 Benchmark results are totally new each pass */
    work_record_initialize (5, &work->pass_work[11], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 13 Benchmark results are totally new each pass */
    work_record_initialize (4, &work->pass_work[12], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 14 Benchmark results are totally new each pass */
    work_record_initialize (3, &work->pass_work[13], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 15 Benchmark results are totally new each pass */
    work_record_initialize (2, &work->pass_work[14], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 16 Benchmark results are totally new each pass */
    work_record_initialize (1, &work->pass_work[15], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* link the lists */
    for (i = 0; i < 15; i++)
    {	work->pass_work[i].Next_Offsets = &work->pass_work[i + 1];
	work->pass_work[i].Next_Source = work->pass_work[i + 1].Source;
    }

    work->pass_work[15].Next_Offsets = &work->work_sentinel;
    work->pass_work[15].Next_Source = work->pass_work[15].Source;

/* pick up plain_text data and the keys */
    get_permuted_data (&work->Keys, &work->Right0, &work->Left0,
						plain_base, key_base);
#ifdef TEST_BENCHMARK_RESULTS
#else
/* Meggs implies that the benchmark does 11 passes on average.  Emulate */
    work->pass_work[10].Next_Offsets = &work->work_sentinel;
#endif

    return_val = DO_ALL (&work->pass_work[0], 0);

/* put back cypher_text data */
    put_permuted_data (&work->Right6, &work->Right4, cypher_base);

    free ((void *)work);
    return (return_val);
}

#ifdef TEST_REMI_KEYS
/* these constant data are grabbed from Remi.  They are in order
 * key_low, key_high, iv_low, iv_high, plain_low, plain_hi, code_low, code_high
 */
unsigned long des_test_cases[][8] = {
  {0x54159B85,0x316B02CE,0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x1AD9EA4A,0x15527592,0x805A8EF3,0x21F68C4C,0xC4B9980E,0xD5C3BD8A,0x5AFD5D57,0x335981AD},
  {0x5B199D6E,0x40C40E6D,0xF5756816,0x36088043,0x4EF1DF24,0x6BF462EC,0xC657515F,0xABB6EBB0},
  {0xAD8C0DEC,0x68385DF1,0x19FB0D4F,0x288D2CD6,0x03FA0F6F,0x038E92F8,0x2FA04E4C,0xBFAB830A},
  {0xC475C22C,0xDFE3B67A,0x5614A37E,0xD70F8E2D,0xCA620ACE,0xA1CF54BB,0xB5BF73A1,0xB2BB55BD},
  {0x2FABC40D,0xE03B8CE6,0xF825C0CF,0x47BDC4A9,0x639F0904,0x354EFC8B,0xC745E11C,0x698BF15F},
  {0x80940E61,0xDCBC7F73,0xA30685EA,0x67CDA3FE,0x6E538AA3,0xC34993BB,0xF6DBDCE9,0x6FCE1832},
  {0x4A701329,0x450D5D0B,0x93D406FA,0x96C9CD56,0xAF7D2E73,0xA1A9F844,0x9428CB49,0x1F93460F},
  {0x2A73B06E,0x8C855D6B,0x3FC6F9D5,0x3F07BC65,0x9A311C3B,0x8FC62B22,0x0E71ECD9,0x003B4F0B},
  {0x255DFBB0,0xB5290115,0xE4663D24,0x702B8D86,0xC082814F,0x6DFA89AC,0xB76E630D,0xF54F4D24},
  {0xBA1A3B6E,0x9158E3C4,0x4C3E8CBC,0xA19D4133,0x7F8072EC,0x6A19424E,0xE09F06DA,0x6508CD88},
  {0xFB32138A,0xF4F73175,0x87C55A28,0xC5FAA7A2,0xDAE86B44,0x629B5CAE,0xAEC607BC,0x9DD8816D},
  {0x5B0BDA4F,0x025B2502,0x1F6A43E5,0x0006E17E,0xB0E11412,0x64EB87EB,0x2AED8682,0x6A8BC154},
  {0xB05837B3,0xFBE083AE,0x3248AE33,0xD92DDA07,0xFAF4525F,0x4E90B16D,0x6725DBD5,0x746A4432},
  {0x76BC4570,0xBFB5941F,0x8F2A8068,0xCE638F26,0xA21EBDF0,0x585A6F8A,0x65A3766E,0x95B6663A},
  {0xC7610E85,0x5DDCBC51,0xB0747E7F,0x8A52D246,0x3825CE90,0xD70EA566,0x50BC63A5,0xDF9DD8FA},
  {0xB9B02615,0x017C3745,0x21BAECAC,0x4771B2AA,0x32703B09,0x0CBEF2BC,0x69907E24,0x0B3928A6},
  {0x0D7C8F0D,0xFDC2DF6E,0x3BBCE282,0x7C62A9D8,0x4E18FA5A,0x2D942C4E,0x5BF53685,0x23E40E20},
  {0xBAA426B6,0xAED92F13,0xC0DAC03C,0x3382923A,0x25F6F952,0x3C691477,0x49B7862A,0x6520E509},
  {0x7C37682A,0x164A43B3,0x9D31C0D1,0x884B1EE5,0x2DCBB169,0xB4530B74,0x3C93D6C3,0x9A9CE765},
  {0x79B55B8F,0x6B8AC2B5,0xE9545371,0x004E203E,0xA3170E57,0x9F71563D,0xF5DE356F,0xBD0191DF},
  {0xC8F80132,0xD532972F,0xBC2145BC,0x42E174FE,0xBA4DCA59,0x6F65FA58,0xB276ADD5,0xA0A9F7B1},
  {0x6E497043,0x7C402CC2,0x0039BB42,0xBD8438A2,0x508592BF,0x1A2F40D6,0x0F1EB5BC,0x6B0C42E7},
  {0xB3C4FD31,0xD619314A,0x39B2DBF7,0x0295F93A,0x4D547967,0x36149936,0x44B02FEE,0xEECC0B2D},
  {0x7FA12954,0x08737CA8,0x8ECDCE90,0x5DACCF36,0x7AA693B0,0x62C8CA9C,0x948CB25E,0xF4781028},
  {0x01BFDC08,0x7558CD0E,0x7D6D82DA,0x19ACD958,0x1EDF3781,0x195110A7,0x021EB315,0xE2EA34C9},
  {0x5161A2C4,0x4F043B43,0x17D76130,0xDCB7695C,0xA70ADBC0,0x843A8801,0xAEE16715,0xE1AF0F07},
  {0x943DF4E3,0xB6D6CEF2,0xC763AAA3,0xA0179248,0xEB61626F,0x1B130032,0x5630226F,0x1C9DBFB2},
  {0xE997049E,0x37D5E085,0x07C372A8,0x3669C801,0x689B4583,0xDA05F0A2,0xFA70DACD,0x3F031F6C},
  {0x4C2F1083,0x5D8A6B32,0xC38544FA,0x017883F5,0xD06D9EAA,0xEE0DFBF6,0xB1A728B7,0x12C311C4},
  {0x5225BCB0,0xE51C98B6,0x2B7ABF2D,0xD714717E,0xC867B0B7,0xF24322B6,0x0A6BF211,0xB0B7C1CA},
/*This next key is the *complement* of the good key */
/*{0xCE6823E9,0x16A8A476,0xCDC4DBA4,0xD93B6603,0xC6E231B9,0xD84C2204,0xDB623F7C,0x3477E4B2}, */
  {0x3197DC16,0xE9575B89,0xCDC4DBA4,0xD93B6603,0xC6E231B9,0xD84C2204,0xDB623F7C,0x3477E4B2},
  {0x11111111,0x11111111,0x00000000,0x00000000,0x00000000,0x00000000,0xb4624df5,0x82e13665},
  {0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0xc1b123a7,0x8ca64de9},
#ifdef TEST_TONS_OF_KEYS
#include "benchmark-test.txt"
#endif /* TEST_TONS_OF_KEYS */
  
#ifdef TEST_KEY_BITS
/* same as above, but with one bit at a time xor'd to see that all are tested */
/* count 3, 5, 8, 10, 11, 12, 15, 18, 40, 41, 42, 43, 45, 46, 49, 50, 0, 1, 2, 4 */
  {0x54159B85 ^ (0x00000001ul << 3+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 5+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 8+2), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 10+2), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 11+2), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 12+2), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 15+3), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 18+3), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (40+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (41+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (42+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (43+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (45+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (46+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (49+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (50+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 0+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 1+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 2+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 4+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
/* count 3, 5, 8, 10, 11, 12, 15, 18, 40, 41, 42, 43, 45, 46, 49, 50, 0, 1, 2, 4 */
/* test opposite! 6, 7, 9, 13, 14, 16, 17, 19, ..., 39, 44, 47, 48, 51, 52, 53, 54, 55 */
/* horrible.  At this point, the LSB of every byte is a parity bit.  Skip */
  {0x54159B85 ^ (0x00000001ul << 6+1), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << 7+2), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (9+2)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (13+2)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (14+3)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (16+3)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x97CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (17+3)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x97CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (19+3)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (20+3)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (21+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (22+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (23+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (24+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (25+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (26+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ (0x00000001ul << (27+4)), 0x316B02CE ^ 0x00000000,
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (28+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (29+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (30+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (31+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (32+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (33+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (34+1 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (35+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (36+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (37+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (38+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (39+2 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (44+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (47+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (48+3 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (51+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (52+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (53+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (54+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
  {0x54159B85 ^ 0x00000000, 0x316B02CE ^ (0x00000001ul << (55+4 - 28)),
	0xB401FC0E,0x737B34C9,0x96CB8518,0x4A6649F9,0x34A74BC8,0x47EE6EF2},
#endif /* TEST_KEY_BITS */
};
#endif /* TEST_REMI_KEYS */

#ifdef DEBUG_MAIN
main()
{   OUTER_LOOP_SLICE plain_text[64];
    OUTER_LOOP_SLICE cypher_text[64];
    OUTER_LOOP_SLICE starting_text[64];
    OUTER_LOOP_SLICE ending_code[64];
    OUTER_LOOP_SLICE key_data[56];
    int i, j;
    unsigned long key_low, key_hi, iv_low, iv_hi, plain_low, plain_hi, code_low, code_hi;
    unsigned long no_par_low, no_par_hi, result_low, result_hi;

    unsigned long long result;
    unsigned long temp_result1, temp_result2;
    unsigned long return_val;
    register unsigned long required_return_value = 0x5a3cf012ul;

#ifdef FULL_64_BIT_VALID
	OUTER_LOOP_SLICE result_mask = SLICE_1;
	OUTER_LOOP_SLICE result_test = 0x01;
#else
#ifdef HIGH_WORD_VALID
	OUTER_LOOP_SLICE result_mask = 0xffffffff00000000ull;
	OUTER_LOOP_SLICE result_test = 0x0100000000ull;
#else
	OUTER_LOOP_SLICE result_mask = 0x00000000ffffffffull;
	OUTER_LOOP_SLICE result_test = 0x01ull;
#endif
#endif

#ifdef TIME_WHACK16
    printf ("Testing 100,000,000 keys\n");
    key_data[6] ^= SLICE_1;		/* 6 is never varied in this test */

    for (i = 0; i < ((REQUIRED_SIZE == 4) ? 200 : 100); i++)
	result = whack16 (&starting_text[0], &ending_code[0], &key_data[0]);
    return(1);
#endif /* TIME_WHACK16 */

#ifdef TEST_TONS_OF_KEYS
    #define TEST_REMI_KEYS
#endif

#ifdef TEST_REMI_KEYS
for (j = 0; j < (sizeof (des_test_cases)/sizeof (des_test_cases[0])); j++)
{    i = j;

    key_low = des_test_cases[i][0];
    key_hi = des_test_cases[i][1];
    iv_low = des_test_cases[i][2];
    iv_hi = des_test_cases[i][3];
    plain_low = des_test_cases[i][4];
    plain_hi = des_test_cases[i][5];
    code_low = des_test_cases[i][6];
    code_hi = des_test_cases[i][7];

/* convert key to slice mode, punting parity bits */
    {	long kk = key_low;
	long mask = 0x00000001;
	for (i = 0; i < 56; i++)
	{   if ((i % 7) == 0) mask <<= 1;
	    key_data[i] = ((kk & mask) != 0) ? SLICE_1 : SLICE_0;
	    if ((mask <<= 1) == 0)
	    {	kk = key_hi;
		mask = 0x00000001;
	    }
	}
    }

    {	long kk = key_low;
	long mask = 0x00000001;
	long or_bit = 0x00000001;
	no_par_low = no_par_hi = 0;
	for (i = 0; i < 28; i++)
	{   if ((i % 7) == 0) mask <<= 1;
	    no_par_low |= ((kk & mask) != 0) ? or_bit : 0;
	    mask <<= 1;
	    or_bit <<= 1;
	}

	kk = key_hi;
	mask = 0x00000001;
	for (i = 0; i < 4; i++)
	{   if ((i % 7) == 0) mask <<= 1;
	    no_par_low |= ((kk & mask) != 0) ? or_bit : 0;
	    mask <<= 1;
	    or_bit <<= 1;
	}

	or_bit = 0x00000001;
	for (i = 4; i < 28; i++)
	{   if ((i % 7) == 0) mask <<= 1;
	    no_par_hi |= ((kk & mask) != 0) ? or_bit : 0;
	    mask <<= 1;
	    or_bit <<= 1;
	}
    }

/* convert plaintext and cyphertext to slice mode */
    {	long pp = plain_low ^ iv_low;
	long cc = code_low;
	long mask = 0x00000001;
	for (i = 0; i < 64; i++)
	{   starting_text[i] = ((pp & mask) != 0) ? SLICE_1 : SLICE_0;
	    ending_code[i] = ((cc & mask) != 0) ? SLICE_1 : SLICE_0;
	    if ((mask <<= 1) == 0)
	    {	pp = plain_hi ^ iv_hi;
		cc = code_hi;
		mask = 0x00000001;
	    }
	}
    }

    result = whack16 (&starting_text[0], &ending_code[0], &key_data[0]);

/* TOADS why does this only trigger when NO_HIGH_WORD_VALID */

#ifdef FULL_64_BIT_VALID
#else /* FULL_64_BIT_VALID */
    temp_result1 = result & 0xffffffff;
    temp_result2 = (result >> 32);
/*    if (temp_result1 != temp_result2)
	printf ("No Banana Results not self-consistent.  %x %x.  Context swap?\n",
					temp_result2, temp_result1);
 */
#endif /* FULL_64_BIT_VALID */

    if ((result & result_mask) != SLICE_0)
    {	result_low = result_hi = 0;
	for (i = 0; i < 32; i++)
	{   result_low |= ((key_data[i] & result & result_mask) != 0x0ull)
						? (1 << i) : 0;
	}
	for (i = 0; i < 24; i++)
	{   result_hi |= ((key_data[i + 32] & result & result_mask) != 0x0ull)
						? (1 << i) : 0;
	}
	printf ("\nGot it %x %x\n",
	    (int)((result & result_mask) >> 32), (int)(result & result_mask));
	if ((no_par_hi != result_hi) || (no_par_low != result_low))
	    printf ("No Banana Bogus Key found is wrong\n");
	printf ("want %x %x got %x %x\n", no_par_hi, no_par_low, result_hi, result_low);
    }
    else
    {	result_low = result_hi = 0;
	for (i = 0; i < 32; i++)
	{   result_low |= ((key_data[i] & result & result_mask) != 0x0ull)
						? (1 << i) : 0;
	}
	for (i = 0; i < 24; i++)
	{   result_hi |= ((key_data[i + 32] & result & result_mask) != 0x0ull)
						? (1 << i) : 0;
	}
	printf ("\nNo Banana\n");
	printf ("want %x %x got %x %x\n", no_par_hi, no_par_low, result_hi, result_low);
    }
}
    return(1);
#endif /* TEST_REMI_KEYS */

/* if not testing whack16, test the benchmark routine */

    printf ("Doing 11 * 1,000,000 function calls\n");
printf ("Code is intentionally corrupted to prevent it from doing an undesired function\n");
    for (j = 0; j < 1000000; j++)
    {	for (i = 0; i < 64; i++) cypher_text[i] = 0;
	return_val = test_benchmark (&key_data[0], &starting_text[0], &cypher_text[0]);

#ifdef TEST_BENCHMARK_RESULTS

/* convert slice mode to data */
	{   result_low = result_hi = 0;
	    for (i = 0; i < 32; i++)
	    {	result_low |= ((cypher_text[i] & result_test) != 0x0ull) ? (1 << i) : 0;
		result_hi |= ((cypher_text[i + 32] & result_test) != 0x0ull) ? (1 << i) : 0;
	    }
	}

	if ((result_low != code_low) || (result_hi != code_hi))
	{   if (return_val != 0)
		printf ("\nbenchmark result failed %x %x, %x %x\n",
		    result_hi, result_low, code_hi, code_low);
	    if (return_val != 0)
		printf ("\nreturn val non-zero %x\n", return_val);
	}
	else
	{   if (return_val != required_return_value)
		if (return_val == 0)
		{   printf (".");
		    fflush (stdout);
		}
		else
		    printf ("\nval %x\n", return_val);
	}

/*	for (i = 0; i < 64; i++)
	{   if ((cypher_text[i] & result_mask) != (ending_code[i] & result_mask))
		printf ("bit %d cypher text %x != expected code %x\n",
		    i, (cypher_text[i] & result_mask), (ending_code[i] & result_mask));
	}
 */
#endif /* TEST_BENCHMARK_RESULTS */
    }

    for (i = 0; i < 64; i++) plain_text[i] = 0;
    test_unbenchmark (&key_data[0], &ending_code[0], &plain_text[0]);

#ifdef TEST_BENCHMARK_RESULTS

/* convert slice mode to data */
    {	{   result_low = result_hi = 0;
	    for (i = 0; i < 32; i++)
	    {	result_low |= ((plain_text[i] & result_test) != 0x0ull) ? (1 << i) : 0;
		result_hi |= ((plain_text[i + 32] & result_test) != 0x0ull) ? (1 << i) : 0;
	    }
	}

	if ((result_low != (plain_low ^ iv_low)) || (result_hi != (plain_hi ^ iv_hi)))
	    printf ("unbenchmark result failed %x %x, %x %x\n",
		result_hi, result_low, plain_hi, plain_low);
	else printf ("unbenchmark succeeded\n");

	for (i = 0; i < 64; i++)
	{   if ((plain_text[i] & result_mask) != (starting_text[i] & result_mask))
		printf ("bit %d plain text %x != starting text %x\n",
		    i, (plain_text[i] & result_mask), (starting_text[i] & result_mask));
	}
    }
#endif /* TEST_BENCHMARK_RESULTS */
    return(1);
}
#endif /* DEBUG_MAIN */

/* Meggs is super sharp.  In fact, smarter than I am.  I will do the
 * obvious part of the taking advantage of constant parts of SBOXes.
 * Meggs knows some tricks that could get a few more percent out of this.
 */

/* trick 1: count using bits that are don't care for encryption pass 1
 * trick 2: count using key bits 10 and 18, which results in only SBOX 1
 *          being re-evaluated in pass 2, and SBOXES 2, 3, 4, 5, 6, 8
 *          in pass 3;
 * trick 3: count using key bits 45 and 49, which results in only SBOX 7
 *          being re-evaluated in pass 2, and SBOXES 1, 2, 3, 4, 6, 8
 *          and all SBOXES being re-evaluated in pass 3;
 * trick 4: every 16 counts as above, simply re-evaluate all sboxes.  This
 *          would need 3 + 8 sboxes if I were tricky, instead of 8+8, but
 *          the savings are only 1/3 SBOX per key test.
 * trick 5: always evaluate passes 4, 5, 6, 7, 8, 9, 10, and 11.  Meggs
 *          short-circuits this, which saves 4/5 sbox evaluation.
 * trick 6: Meggs's great idea:  Meggs does incremental calculations to
 *          discard data early.  It turns out that there are 2 pass 14
 *          inverse sboxes that simply do not depend on the the 8 bits
 *          that are varying as don't cares to Pass 1.  These have been
 *          precalculated.  They can be compared to partial results out
 *          of pass 12, and result in early discards of the keys.
 * trick 7: Meggs does incremental calculations thereafter.  I am too lazy.
 *          The user should consider doing a couple of individual sbox
 *          calculations, and then just fall back to doing 8 at a time.
 *
 * The main evaluate routine can evaluate 1 followed by 2, 3, 4, 5, 6, 8,
 * or it can evaluate 7 followed by 1, 2, 3, 4, 6, 8, or it can do all 8.
 */

/* In des-slice-meggs.cpp
 * Upon entry to this code, key 40, key 41, key 0, key 1, key 2, and
 * if 64-bit activity, key 4 are set to incrementing values
 * Meggs expects bits 3, 5, 8, 10, 11, 12, 15, 18, 42, 43, 45, 46, 49, and 50
 * to be incremented inside his routine, which will to a total of 20 bits
 * of work (or 19 bits if 32-bit code).
 *
 * In des-slice.cpp
 * Upon entry to this code, key 3, key 5, key 8, key 10, key 11, and
 * if 64-bit activity, key 12 are set to incrementing values.
 * That code expects bits 15, 18, 40, 41, 42, 43, 45, 46, 49, 50, 0, 1, 2, 4
 * to be incremented inside this routine, which will to a total of 20 bits
 * of work (or 19 bits if 32-bit code).
 *
 * How nice! The same bits!
 *
 * It turns out that bits 10, 12, 15, 18, 45, 46, 49, and 50 are Dont Care
 * bits for round 1 calculations.
 * It turns out that 3, 5, 8, 11, 42, 43, and 38 and 39 are Dont Care bits
 * for round 16 calculations.
 *
 * This code should feel free to overstomp any of the above bits in the
 * input keys, if it can to advantage.  BUT NO OTHER BITS.  They are handled
 * by the contest people.
 */

/* this struct is here to force malloc to allocate contiguous data */
struct Contest_Work
    {	struct KEYS Keys;
	struct KEYS Inverse_Keys;
	struct SLICE_ARRAY Left0;	/* initial */
	struct SLICE_ARRAY Right0;	/* initial */
	struct SLICE_ARRAY Right1;	/* updated once every 256 */
	struct SLICE_ARRAY Right2;	/* updated 1 sbox every pass */
	struct SLICE_ARRAY Right3;	/* updated 6 sboxes every pass */
	struct SLICE_ARRAY Right4;	/* running storage during looping */
	struct SLICE_ARRAY Right5;	/* running storage during looping */
	struct SLICE_ARRAY Right6;	/* running storage during looping */
	struct SLICE_ARRAY Right7;	/* running storage during checking */
	struct SLICE_ARRAY Right8;	/* running storage during checking */
	struct SLICE_ARRAY Right9;	/* running storage during checking */
	struct OUTER_OFFSET_DISTANCES pass_1_work;
	struct OUTER_OFFSET_DISTANCES pass_2_11_work[10];
	struct OUTER_OFFSET_DISTANCES pass_12_work;
	struct OUTER_OFFSET_DISTANCES pass_12_forward_14_back_work[2];
	struct OUTER_OFFSET_DISTANCES work_sentinel;
	struct OUTER_OFFSET_DISTANCES pass_12_13_work[2];
	struct OUTER_OFFSET_DISTANCES pass_12_15_work[4];
	struct OUTER_OFFSET_DISTANCES pass_14_work;
	struct OUTER_OFFSET_DISTANCES pass_15_work;
	struct OUTER_OFFSET_DISTANCES pass_16_work;
	struct OUTER_OFFSET_DISTANCES inverse_pass_14_work;
	struct CACHED_RESULTS_ARRAY Cached_Results[256];
	struct OUTER_OFFSET_DISTANCES inverse_pass_14_work_list[16];
	struct OUTER_OFFSET_DISTANCES inverse_pass_15_work_list[16];
	struct OUTER_OFFSET_DISTANCES inverse_pass_16_work_list[16];
	struct SLICE_ARRAY Right13[256];	/* TOADS could be 16 long. */
	struct SLICE_ARRAY Right14[256];
	struct SLICE_ARRAY Right15[256];
	struct SLICE_ARRAY Right16;
	struct SLICE_ARRAY Left16;
    };

/* These routines apply a list of 16 work items to a set of 256 array items
 * This is supposed to save memory, which is said to decrease cache thrashing.
 * In the case of pass 16, a single source is used for all operations.
 * For other cases, different sources are used for each destination
 * As this code applies the 16 work items 16 times, it must increment
 * the key vectors.  The keys are incremented in the order [12, 15, 46, 50].
 * (The key bits [10, 18, 45, 49] are pre-incremented in the saved data.)
 * The other key bits  [3, 11, 8, 43, 5, 42] come from the running key set
 */

void
do_inverse_list_fancy (register struct Contest_Work *work, int Pass, int Funct
){  register int i, j;
    unsigned long return_val;
    register unsigned long required_return_value = 0x5a3cf012ul;
    struct SLICE_ARRAY *Cached_Data_Src;
    struct CACHED_RESULTS_ARRAY *Cached_Data_Dest;

    for (j = 0; j < 16; j++) /* 16 groups of 16 calculations are 256 */
    {
/* increment the keys */
	work->Inverse_Keys.Keys[12] = ((j & 0x01) != 0) ? SLICE_1 : SLICE_0;
	work->Inverse_Keys.Keys[15] = ((j & 0x02) != 0) ? SLICE_1 : SLICE_0;
	work->Inverse_Keys.Keys[46] = ((j & 0x04) != 0) ? SLICE_1 : SLICE_0;
	work->Inverse_Keys.Keys[50] = ((j & 0x08) != 0) ? SLICE_1 : SLICE_0;

	if (Pass == 15) /* pass 15 different sources to different destinations */
	{	/* make the source point to the appropriate place */
		/* make the destination point to the appropriate place */
	    for (i = 0; i < 16; i++)
	    {	work->inverse_pass_15_work_list[i].Source =
				&work->Right15[(16 * j) + i].Data[0];
		work->inverse_pass_15_work_list[i].Dest =
				&work->Right14[(16 * j) + i].Data[0];
	    }

/* make the next source point to the appropriate place */
	    for (i = 0; i < 15; i++)
		work->inverse_pass_15_work_list[i].Next_Source =
				work->inverse_pass_15_work_list[i + 1].Source;
		work->inverse_pass_15_work_list[15].Next_Source =
				work->inverse_pass_15_work_list[15].Source;

/* apply the work record, which does 16 8-box calculations */
	    if (Funct == 3)
	    {	do {
		    return_val = DO_S3 (&work->inverse_pass_15_work_list[0]);
		} while (return_val != required_return_value);
	    }
	    else if (Funct == -1)	/* -1 means 3 followed by 7 */
	    {	do {
		    return_val = DO_S3 (&work->inverse_pass_15_work_list[0]);
		} while (return_val != required_return_value);
		do {
		    return_val = DO_S7 (&work->inverse_pass_15_work_list[0]);
		} while (return_val != required_return_value);
	    }
	    else
	    {	do {
		    return_val = DO_ALL_FANCY (&work->inverse_pass_15_work_list[0], 0);
		} while (return_val != required_return_value);
	    }
	}
	else if (Pass == 14) /* pass 14 different sources to different destinations */
	{	/* make the source point to the appropriate place */
		/* make the destination point to the appropriate place */
	    for (i = 0; i < 16; i++)
	    {	work->inverse_pass_14_work_list[i].Source =
				&work->Right14[(16 * j) + i].Data[0];
		work->inverse_pass_14_work_list[i].Merge =
				&work->Right15[(16 * j) + i].Data[0];
		work->inverse_pass_14_work_list[i].Dest =
				&work->Right13[(16 * j) + i].Data[0];
	    }

/* make the next source point to the appropriate place */
	    for (i = 0; i < 15; i++)
		work->inverse_pass_14_work_list[i].Next_Source =
				work->inverse_pass_14_work_list[i + 1].Source;
		work->inverse_pass_14_work_list[15].Next_Source =
				work->inverse_pass_14_work_list[15].Source;

/* apply the work record, which does 16 8-box calculations */
	    if (Funct == -1)		/* -1 means 1 followed by 3 */
	    {	do {
		    return_val = DO_S1 (&work->inverse_pass_14_work_list[0]);
		} while (return_val != required_return_value);
		do {
		    return_val = DO_S3 (&work->inverse_pass_14_work_list[0]);
		} while (return_val != required_return_value);
	    }
	    else
	    {	do {
		    return_val = DO_ALL_FANCY (&work->inverse_pass_14_work_list[0], 0);
		} while (return_val != required_return_value);
	    }

/* collect the results into the result array to speed the key test */
	    for (i = 0; i < 16; i++)
	    {	Cached_Data_Src = &work->Right13[(16 * j) + i];
		Cached_Data_Dest = &work->Cached_Results[(16 * j) + i];
		Cached_Data_Dest->Cached_Data_5 = Cached_Data_Src->Data[5];
		Cached_Data_Dest->Cached_Data_8 = Cached_Data_Src->Data[8];
		Cached_Data_Dest->Cached_Data_15 = Cached_Data_Src->Data[15];
		Cached_Data_Dest->Cached_Data_16 = Cached_Data_Src->Data[16];
		Cached_Data_Dest->Cached_Data_22 = Cached_Data_Src->Data[22];
		Cached_Data_Dest->Cached_Data_23 = Cached_Data_Src->Data[23];
		Cached_Data_Dest->Cached_Data_29 = Cached_Data_Src->Data[29];
		Cached_Data_Dest->Cached_Data_30 = Cached_Data_Src->Data[30];
	    }
	}
	else if (Pass == 16) /* pass 16 constant source to different destinations */
	{	/* make the destination point to the appropriate place */
	    for (i = 0; i < 16; i++)
	    {	work->inverse_pass_16_work_list[i].Dest =
				&work->Right15[(16 * j) + i].Data[0];
	    }

/* apply the work record, which does 16 8-box calculations */
	    do {
		return_val = DO_ALL_FANCY (&work->inverse_pass_16_work_list[0], 0);
	    } while (return_val != required_return_value);
	}
    }
}

/* need to hold several pieces of data:
 * 1) initial Plaintext data.  This will be used as input to Round 1
 * 2) data output from Round 1.  This will be used as input to Round 2.
 * 3) running data.  This is output data from Round 3, 4, 5, 6, 7, 8, 9, 10, 11
 * 4) Initial Code data.  This will be used as input to Inverse Round 16.
 *
 * all forward calculations are initialized here.  Inverse operations are
 * initialized in the next routine.
 */

void
initialize_contest_work_structure (register struct Contest_Work *work
){  register int i;
/* initialize offset pointers, to be ready to encrypt */
/* pass 1 Benchmark results will be constant as keys are incremented 256 times */
    work_record_initialize (1, &work->pass_1_work, &work->Keys.Keys[0],
	&work->Right0.Data[0], &work->Left0.Data[0], &work->Right1.Data[0]);

/* pass 2 Benchmark results will change 1 sbox every key increment */
    work_record_initialize (2, &work->pass_2_11_work[0], &work->Keys.Keys[0],
	&work->Right1.Data[0], &work->Right0.Data[0], &work->Right2.Data[0]);

/* pass 3 Benchmark results will change 6 sboxes every key increment */
    work_record_initialize (3, &work->pass_2_11_work[1], &work->Keys.Keys[0],
	&work->Right2.Data[0], &work->Right1.Data[0], &work->Right3.Data[0]);

/* pass 4 Benchmark results are totally new each pass */
    work_record_initialize (4, &work->pass_2_11_work[2], &work->Keys.Keys[0],
	&work->Right3.Data[0], &work->Right2.Data[0], &work->Right4.Data[0]);

/* pass 5 Benchmark results are totally new each pass */
    work_record_initialize (5, &work->pass_2_11_work[3], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right3.Data[0], &work->Right5.Data[0]);

/* pass 6 Benchmark results are totally new each pass */
    work_record_initialize (6, &work->pass_2_11_work[4], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 7 Benchmark results are totally new each pass */
    work_record_initialize (7, &work->pass_2_11_work[5], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 8 Benchmark results are totally new each pass */
    work_record_initialize (8, &work->pass_2_11_work[6], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 9 Benchmark results are totally new each pass */
    work_record_initialize (9, &work->pass_2_11_work[7], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 10 Benchmark results are totally new each pass */
    work_record_initialize (10, &work->pass_2_11_work[8], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right4.Data[0]);

/* pass 11 Benchmark results are totally new each pass */
    work_record_initialize (11, &work->pass_2_11_work[9], &work->Keys.Keys[0],
	&work->Right4.Data[0], &work->Right6.Data[0], &work->Right5.Data[0]);

/* pass 12 Benchmark results are totally new each pass */
    work_record_initialize (12, &work->pass_12_work, &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 12 Benchmark results are totally new each pass */
    work_record_initialize (12, &work->pass_12_15_work[0], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 13 Benchmark results are totally new each pass */
    work_record_initialize (13, &work->pass_12_15_work[1], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right7.Data[0]);

/* pass 14 Benchmark results are totally new each pass */
    work_record_initialize (14, &work->pass_12_15_work[2], &work->Keys.Keys[0],
	&work->Right7.Data[0], &work->Right6.Data[0], &work->Right8.Data[0]);

/* pass 15 Benchmark results are totally new each pass */
    work_record_initialize (15, &work->pass_12_15_work[3], &work->Keys.Keys[0],
	&work->Right8.Data[0], &work->Right7.Data[0], &work->Right9.Data[0]);

/* pass 12 Benchmark results are totally new each pass */
    work_record_initialize (12, &work->pass_12_13_work[0], &work->Keys.Keys[0],
	&work->Right5.Data[0], &work->Right4.Data[0], &work->Right6.Data[0]);

/* pass 13 Benchmark results are totally new each pass */
    work_record_initialize (13, &work->pass_12_13_work[1], &work->Keys.Keys[0],
	&work->Right6.Data[0], &work->Right5.Data[0], &work->Right7.Data[0]);

/* pass 14 Benchmark results are totally new each pass */
    work_record_initialize (14, &work->pass_14_work, &work->Keys.Keys[0],
	&work->Right7.Data[0], &work->Right6.Data[0], &work->Right8.Data[0]);

/* pass 15 Benchmark results are totally new each pass */
    work_record_initialize (15, &work->pass_15_work, &work->Keys.Keys[0],
	&work->Right8.Data[0], &work->Right7.Data[0], &work->Right9.Data[0]);

/* pass 16 Benchmark results are totally new each pass */
    work_record_initialize (16, &work->pass_16_work, &work->Keys.Keys[0],
	&work->Right9.Data[0], &work->Right8.Data[0], &work->Right4.Data[0]);

/* link the lists */
/* pass 1 always falls through to pass 2 */
    work->pass_1_work.Next_Offsets = &work->pass_2_11_work[0];
    work->pass_1_work.Next_Source = work->pass_2_11_work[0].Source;

/* going to do passes 2 through 11, and then stop */
    for (i = 0; i < 9; i++)
    {	work->pass_2_11_work[i].Next_Offsets = &work->pass_2_11_work[i + 1];
	work->pass_2_11_work[i].Next_Source = work->pass_2_11_work[i + 1].Source;
    }

    work->pass_2_11_work[9].Next_Offsets = &work->work_sentinel;
    work->pass_2_11_work[9].Next_Source = work->pass_2_11_work[9].Source;

/* going to do passes 12 through 15, and then stop */
    for (i = 0; i < 3; i++)
    {	work->pass_12_15_work[i].Next_Offsets = &work->pass_12_15_work[i + 1];
	work->pass_12_15_work[i].Next_Source = work->pass_12_15_work[i + 1].Source;
    }

    work->pass_12_15_work[3].Next_Offsets = &work->work_sentinel;
    work->pass_12_15_work[3].Next_Source = work->pass_12_15_work[3].Source;

/* do pass 12 followed by pass 13 */
    work->pass_12_13_work[0].Next_Offsets = &work->pass_12_13_work[1];
    work->pass_12_13_work[0].Next_Source = work->pass_12_13_work[1].Source;

    work->pass_12_13_work[1].Next_Offsets = &work->work_sentinel;
    work->pass_12_13_work[1].Next_Source = work->pass_12_13_work[1].Source;

/* going to do pass 12 all by itself */
    work->pass_12_work.Next_Offsets = &work->work_sentinel;
    work->pass_12_work.Next_Source = work->pass_12_work.Source;

/* going to do pass 14 all by itself */
    work->pass_14_work.Next_Offsets = &work->work_sentinel;
    work->pass_14_work.Next_Source = work->pass_14_work.Source;

/* going to do pass 15 all by itself */
    work->pass_15_work.Next_Offsets = &work->work_sentinel;
    work->pass_15_work.Next_Source = work->pass_15_work.Source;

/* going to do pass 16 all by itself */
    work->pass_16_work.Next_Offsets = &work->work_sentinel;
    work->pass_16_work.Next_Source = work->pass_16_work.Source;

/* this work element is shared between forward and backward.  It must
 * be initialized forward first, then backawrds
 */
    work_record_initialize (12, &work->pass_12_forward_14_back_work[1],
	&work->Keys.Keys[0], &work->Right5.Data[0],
	&work->Right4.Data[0], &work->Right6.Data[0]);

    work->pass_12_forward_14_back_work[1].Next_Offsets = &work->work_sentinel;
    work->pass_12_forward_14_back_work[1].Next_Source =
	work->pass_12_forward_14_back_work[1].Source;
}

/* 5) 256 copies of Inverse Round 16 output.  These are used to hold the results
 *    of pre-calculations, which are later used to compare with round 13 outputs.
 * 6) An array of Inverse Round 15 output data.  This data is used to compare
 *    with 2 sboxes of Round 12 outputs.
 */

/* There will be 16 work items that are used to calculate 256 cached sets
 * of inverse data.  Each of these 16 work items will apply different
 * keys to the data they operate on.
 * The clever hack here is to use the normal key vector for all the
 * bits that are fixed across this iteration.  The 8 bits that vary
 * across this calculation are separated into 4 bits that are pre-computed
 * for each of the 16 work list items, and 4 bits that vary.
 * Put these keys where they will be useful for inverse calculations
 * These keys will be used to produce 256 cached results that are
 * used to cause early rejection of keys.
 *
 * For the purposes of this calculation keys are counted in the order
 * [10, 18, 45, 49], [12, 15, 46, 50], [3, 11, 8, 43, 5, 42].
 */

void
initialize_contest_inverse_work_structure (register struct Contest_Work *work
){  register int i, j;
    OUTER_LOOP_SLICE *Key_Ptrs10[16], *Key_Ptrs18[16], *Key_Ptrs45[16], *Key_Ptrs49[16];

/* need a 0 and a 0xff to be used for counting keys across work list elements */
    work->Inverse_Keys.Keys[0] = 0;
    work->Inverse_Keys.Keys[1] = ~((OUTER_LOOP_SLICE) SLICE_0);
    for (j = 0; j < 16; j++)
    {	Key_Ptrs10[j] = Key_Ptrs18[j] = Key_Ptrs45[j] = Key_Ptrs49[j] =
	    &work->Inverse_Keys.Keys[0];	/* 0 */
    }

/* spell out the bits, because these are done in grey code order */
    for (j = 0; j < 16; j++)
    {	switch (j)
	{   case 0x01: Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1]; /* ~0 */
		break;
	    case 0x02: Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x03: Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x04: Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x05: Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x06: Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x07: Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x08: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x09: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x0a: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x0b: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs45[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x0c: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x0d: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs18[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x0e: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		       Key_Ptrs10[j] = &work->Inverse_Keys.Keys[1];
		break;
	    case 0x0f: Key_Ptrs49[j] = &work->Inverse_Keys.Keys[1];
		break;
	    default:
		break;
	}
    }

/* Meggs cleverly caches data when it does not change */
/* To do this, we need even more linked lists */

/* inverse pass 16 results will be calculated at the beginning */
    for (i = 0; i < 16; i++)
    {	work_record_initialize (16, &work->inverse_pass_16_work_list[i],
	    &work->Keys.Keys[0], &work->Left16.Data[0],
	    &work->Right16.Data[0], &work->Right15[i].Data[0]);
    }

/* customize the keys to point to 16 incrementing key sets */
    if (   (Round16_Key[16] != 10) || (Round16_Key[13] != 18)
	|| (Round16_Key[29] != 45) || (Round16_Key[45] != 49)
	|| (Round16_Key[11] != 12) || (Round16_Key[38] != 15)
	|| (Round16_Key[12] != 46) || (Round16_Key[36] != 50))
	printf ("Putting inverse round 16 keys in wrong inverse address\n");
    for (i = 0; i < 16; i++)
    {	work->inverse_pass_16_work_list[i].Key_Ptrs[16] = Key_Ptrs10[i];
	work->inverse_pass_16_work_list[i].Key_Ptrs[13] = Key_Ptrs18[i];
	work->inverse_pass_16_work_list[i].Key_Ptrs[29] = Key_Ptrs45[i];
	work->inverse_pass_16_work_list[i].Key_Ptrs[45] = Key_Ptrs49[i];
	work->inverse_pass_16_work_list[i].Key_Ptrs[11] =
					&work->Inverse_Keys.Keys[12];
	work->inverse_pass_16_work_list[i].Key_Ptrs[38] =
					&work->Inverse_Keys.Keys[15];
	work->inverse_pass_16_work_list[i].Key_Ptrs[12] =
					&work->Inverse_Keys.Keys[46];
	work->inverse_pass_16_work_list[i].Key_Ptrs[36] =
					&work->Inverse_Keys.Keys[50];
    }

/* link the lists */
    for (i = 0; i < 15; i++)
    {	work->inverse_pass_16_work_list[i].Next_Offsets =
	    &work->inverse_pass_16_work_list[i + 1];
	work->inverse_pass_16_work_list[i].Next_Source =
	    work->inverse_pass_16_work_list[i + 1].Source;
    }

    work->inverse_pass_16_work_list[15].Next_Offsets = &work->work_sentinel;
    work->inverse_pass_16_work_list[15].Next_Source =
	work->inverse_pass_16_work_list[15].Source;

/* inverse pass 15 results will be calculated every 4096 passes */
    for (i = 0; i < 16; i++)
    {	work_record_initialize (15, &work->inverse_pass_15_work_list[i],
	    &work->Keys.Keys[0], &work->Right15[i].Data[0],
	    &work->Left16.Data[0], &work->Right14[i].Data[0]);
    }

/* customize the keys to point to 16 incrementing key sets */
    if (   (Round15_Key[20] != 10) || (Round15_Key[21] != 18)
	|| (Round15_Key[39] != 45) || (Round15_Key[27] != 49)
	|| (Round15_Key[2] != 12)  || (Round15_Key[31] != 15)
	|| (Round15_Key[3] != 46)  || (Round15_Key[33] != 50))
	printf ("Putting inverse round 15 keys in wrong inverse address\n");
    for (i = 0; i < 16; i++)
    {	work->inverse_pass_15_work_list[i].Key_Ptrs[20] = Key_Ptrs10[i];
	work->inverse_pass_15_work_list[i].Key_Ptrs[21] = Key_Ptrs18[i];
	work->inverse_pass_15_work_list[i].Key_Ptrs[39] = Key_Ptrs45[i];
	work->inverse_pass_15_work_list[i].Key_Ptrs[27] = Key_Ptrs49[i];
	work->inverse_pass_15_work_list[i].Key_Ptrs[2] =
					&work->Inverse_Keys.Keys[12];
	work->inverse_pass_15_work_list[i].Key_Ptrs[31] =
					&work->Inverse_Keys.Keys[15];
	work->inverse_pass_15_work_list[i].Key_Ptrs[3] =
					&work->Inverse_Keys.Keys[46];
	work->inverse_pass_15_work_list[i].Key_Ptrs[33] =
					&work->Inverse_Keys.Keys[50];
    }

/* link the lists */
    for (i = 0; i < 15; i++)
    {	work->inverse_pass_15_work_list[i].Next_Offsets =
	    &work->inverse_pass_15_work_list[i + 1];
	work->inverse_pass_15_work_list[i].Next_Source =
	    work->inverse_pass_15_work_list[i + 1].Source;
    }

    work->inverse_pass_15_work_list[15].Next_Offsets = &work->work_sentinel;
    work->inverse_pass_15_work_list[15].Next_Source =
	work->inverse_pass_15_work_list[15].Source;

/* inverse pass 14 results will be calculated every 1024 passes */
    for (i = 0; i < 16; i++)
    {	work_record_initialize (14, &work->inverse_pass_14_work_list[i],
	    &work->Keys.Keys[0], &work->Right14[i].Data[0],
	    &work->Right15[i].Data[0], &work->Right13[i].Data[0]);
    }

/* customize the keys to point to 16 incrementing key sets */
    if (   (Round14_Key[4] != 10) /* || (Round14_Key[] != 18)  NO 18 in pass 14*/
	|| (Round14_Key[30] != 45) || (Round14_Key[38] != 49)
	|| (Round14_Key[22] != 12) || (Round14_Key[43] != 15)
	|| (Round14_Key[16] != 46) || (Round14_Key[28] != 50))
	printf ("Putting inverse round 14 keys in wrong inverse address\n");
    for (i = 0; i < 16; i++)
    {	work->inverse_pass_14_work_list[i].Key_Ptrs[4] = Key_Ptrs10[i];
/*	work->inverse_pass_14_work_list[i].Key_Ptrs[18] = Key_Ptrs18[i]; */
	work->inverse_pass_14_work_list[i].Key_Ptrs[30] = Key_Ptrs45[i];
	work->inverse_pass_14_work_list[i].Key_Ptrs[38] = Key_Ptrs49[i];
	work->inverse_pass_14_work_list[i].Key_Ptrs[22] =
					&work->Inverse_Keys.Keys[12];
	work->inverse_pass_14_work_list[i].Key_Ptrs[43] =
					&work->Inverse_Keys.Keys[15];
	work->inverse_pass_14_work_list[i].Key_Ptrs[16] =
					&work->Inverse_Keys.Keys[46];
	work->inverse_pass_14_work_list[i].Key_Ptrs[28] =
					&work->Inverse_Keys.Keys[50];
    }

/* link the lists */
    for (i = 0; i < 15; i++)
    {	work->inverse_pass_14_work_list[i].Next_Offsets =
	    &work->inverse_pass_14_work_list[i + 1];
	work->inverse_pass_14_work_list[i].Next_Source =
	    work->inverse_pass_14_work_list[i + 1].Source;
    }

    work->inverse_pass_14_work_list[15].Next_Offsets = &work->work_sentinel;
    work->inverse_pass_14_work_list[15].Next_Source =
	work->inverse_pass_14_work_list[15].Source;

/* individual inverse pass 14 results will be calculated each pass */
    work_record_initialize (14, &work->inverse_pass_14_work,
	&work->Keys.Keys[0], &work->Right14[0].Data[0],
	&work->Right15[0].Data[0], &work->Right13[0].Data[0]);

    work->inverse_pass_14_work.Next_Offsets = &work->work_sentinel;
    work->inverse_pass_14_work.Next_Source = work->inverse_pass_14_work.Source;

/* Finally, make a list consisting of an inverse pass followed by a forward
 * pass.  This reduces the overhead of early discards
 */

/* this work element is shared between forward and backward.  It must
 * be initialized forward first, then backawrds
 */
    work_record_initialize (14, &work->pass_12_forward_14_back_work[0],
	&work->Keys.Keys[0], &work->Right14[0].Data[0],
	&work->Right15[0].Data[0], &work->Right13[0].Data[0]);

    work->pass_12_forward_14_back_work[0].Next_Offsets =
	&work->pass_12_forward_14_back_work[1];
    work->pass_12_forward_14_back_work[0].Next_Source =
	work->pass_12_forward_14_back_work[1].Source;
}

OUTER_LOOP_SLICE
whack16 (OUTER_LOOP_SLICE *plain_base,
		OUTER_LOOP_SLICE *cypher_base, OUTER_LOOP_SLICE *key_base
){
    register int i, j, k;
    struct Contest_Work work_struct;
    register struct Contest_Work * work = &work_struct;
    register OUTER_LOOP_SLICE *calculated_inverse_data;
    struct CACHED_RESULTS_ARRAY *cached_inverse_data;
    register OUTER_LOOP_SLICE result;
    unsigned long return_val;
    register unsigned long required_return_value = 0x5a3cf012ul;

/* malloc a single structure, consisting of key data, working data, and
 * then a bunch of work elements.  By putting these into a struct, we
 * can be sure that they are side-by-side in memory.
 */

/* create some linked-lists */
    work = (struct Contest_Work *)
	memalign (64, sizeof (struct Contest_Work));

/* make sure sentinel pointer indicates end of list */
    work_record_initialize (1, &work->work_sentinel, &work->Keys.Keys[0],
	&work->Right6.Data[0], (OUTER_LOOP_SLICE *)0, &work->Right1.Data[0]);

/* initialize offset pointers, to be ready to encrypt */
    initialize_contest_work_structure (work);

/* initialize offset pointers, to be ready to encrypt */
    initialize_contest_inverse_work_structure (work);

/* This code counts bits in order
 * [ [10, 18], [45, 49], [12, 15, 46, 50], [3, 11], [8, 43], [5, 42] ]
 *
 * LSB grey-code counting of [10, 18, 45, 49] lets the code do partial
 * updates of round 2 and 3 data, saving 8 sbox calculations per key
 *
 * MSB countings of [3,11] first keeps the results of sboxes 1 and 3
 * in inverse pass 14 constant.  These bits can therefore be compared
 * with results from pass 12, resulting in early discards.
 *
 * MSB countings of [8, 43] next results in the need to do recalculations
 * of inverse pass 15 SBOX 7, followed by recalculation of inverse pass 14
 * sboxes 1 and 3.
 *
 * Meggs is super sharp.  In fact, smarter than I am.  I will do the
 * obvious part of the taking advantage of constant parts of SBOXes.
 * Meggs knows some tricks that could get a few more percent out of this.
 *
 * Clever Meggs counts bit 42 before bits [8, 43] because this requires
 * only that inverse pass 14 sbox 1 be recalculated, saving work.  However,
 * I am worried about code explosion, and want fewer special cases.  So
 * I use the patterns that need the least work, and do extra work in all
 * other cases.
 */

/* pick up plain_text data and the keys */
    get_permuted_data (&work->Keys, &work->Left0, &work->Right0,
					plain_base, key_base);

/* pick up cypher_text data and the keys */
    get_permuted_data (&work->Keys, &work->Right16, &work->Left16,
					cypher_base, key_base);

/* bits not used by Meggs core */
    work->Keys.Keys[40] = 0xAAAAAAAAAAAAAAAAull;
    work->Keys.Keys[41] = 0xCCCCCCCCCCCCCCCCull;
    work->Keys.Keys[ 0] = 0xF0F0F0F0F0F0F0F0ull;
    work->Keys.Keys[ 1] = 0xFF00FF00FF00FF00ull;
    work->Keys.Keys[ 2] = 0xFFFF0000FFFF0000ull;
#ifdef ULTRA_BIT_64
    work->Keys.Keys[ 4] = 0xFFFFFFFF00000000ull;
#endif

/* Zero out all the bits that are to be varied */
    work->Keys.Keys[10] = work->Keys.Keys[18] =
    work->Keys.Keys[45] = work->Keys.Keys[49] =
    work->Keys.Keys[12] = work->Keys.Keys[15] =
    work->Keys.Keys[46] = work->Keys.Keys[50] =
    work->Keys.Keys[ 3] = work->Keys.Keys[11] =
    work->Keys.Keys[ 8] = work->Keys.Keys[43] =
    work->Keys.Keys[ 5] = work->Keys.Keys[42] = SLICE_0;

/* at startup, recalculate all 8 inverse round 16 sboxes into 256 separate areas */
    do_inverse_list_fancy (work, 16, 0);

/* loop over all keys to test, and search for a passing key.  This does
 * 16,000 passes * 64 because of varying keys per block, or 1 million keys
 */
    for (j = 0; j < (256 * 64); j++)
    {
/* grey-code results in (15 * (1 + 6) + (2 * 8)) / 16 == 7.6 sboxes per pass */
	if ((j & 0x01) != 0)
	{   if ((j & 0x02) != 0)
		work->Keys.Keys[10] = SLICE_0;
	    else
		work->Keys.Keys[10] = SLICE_1;
	}
	else
	{   if ((j & 0x02) != 0)
	    {	if ((j & 0x04) != 0)
		    work->Keys.Keys[18] = SLICE_0;
		else
		    work->Keys.Keys[18] = SLICE_1;
	    }
	    else
	    {	if ((j & 0x04) != 0)
		{   if ((j & 0x08) != 0)
			work->Keys.Keys[45] = SLICE_0;
		    else
			work->Keys.Keys[45] = SLICE_1;
		}
		else
		{   if ((j & 0x08) == 0)
		    {	work->Keys.Keys[49] = SLICE_0;
			work->Keys.Keys[12] = ((j & 0x10) != 0) ? SLICE_1 : SLICE_0;
			work->Keys.Keys[15] = ((j & 0x20) != 0) ? SLICE_1 : SLICE_0;
			work->Keys.Keys[46] = ((j & 0x40) != 0) ? SLICE_1 : SLICE_0;
			work->Keys.Keys[50] = ((j & 0x80) != 0) ? SLICE_1 : SLICE_0;
		    }
		    else
			work->Keys.Keys[49] = SLICE_1;
		}
	    }
	}

/* do extra work based on MSB of counter, resulting in inverse data being cached */
	if ((j & 0x00ff) == 0)
	{   work->Keys.Keys[3] = ((j & 0x100) != 0) ? SLICE_1 : SLICE_0;
	    work->Keys.Keys[11] = ((j & 0x200) != 0) ? SLICE_1 : SLICE_0;

/* 1/1024 early result discard calculations based on cached results take
 * 1/1024 * ((3/4 * 256 * 3) + (1/4 * 256 * 10)) = 1.2 sboxes per iteration
 */
	    if ((j & 0x03ff) == 0)
	    {	work->Keys.Keys[8] = ((j & 0x400) != 0) ? SLICE_1 : SLICE_0;
		work->Keys.Keys[43] = ((j & 0x800) != 0) ? SLICE_1 : SLICE_0;

		if ((j & 0x0fff) == 0)
		{   work->Keys.Keys[5] = ((j & 0x1000) != 0) ? SLICE_1 : SLICE_0;
		    work->Keys.Keys[42] = ((j & 0x2000) != 0) ? SLICE_1 : SLICE_0;
/* 1/4096 recalculate all 8 inverse round 15 sboxes into 256 separate areas */
		    do_inverse_list_fancy (work, 15, 0);
		}
		else
		{
/* 3/4096 recalculate all of inverse 15 sbox 3 and 7 into 256 separate areas */
		    do_inverse_list_fancy (work, 15, -1); /* -1 means 3 then 7 */
		}
/* recalculate all of inverse 14 sbox 1 and sbox 3 values into 256 separate areas */
		do_inverse_list_fancy (work, 14, -1); /* -1 means 1 then 3 */
	    }
	    else
	    {
/* 1/256 early result discard calculations take 1 sbox per iteration */
		do_inverse_list_fancy (work, 15, 3);
	    }
	}

/* forward calculations do ((15 * 7) + 16)/16 = 7 sboxes for first 3 passes */
	do {
	    if ((j & 0x03) != 0)	/* 1 means do sbox 1, sboxes 2,3,4,5,6,8 */
		return_val = DO_ALL_FANCY (&work->pass_2_11_work[0], 1);
	    else
	    {	if ((j & 0x0f) != 0)	/* 2 means do sbox 7, sboxes 1,2,3,4,6,8 */
		    return_val = DO_ALL_FANCY (&work->pass_2_11_work[0], 2);
		else			/* 0 means do sbox 1,2,3,4,5,6,7,8, all 8 sboxes */
		{   if ((j & 0xff) != 0)
			return_val = DO_ALL_FANCY (&work->pass_2_11_work[0], 0);
		    else
			return_val = DO_ALL_FANCY (&work->pass_1_work, 0);
		}
	    }
	} while (return_val != required_return_value);

/* TOADS make this -1 to test that both halves calculate the same result */
#ifdef FULL_64_BIT_VALID
	result = SLICE_1;
#else
#ifdef HIGH_WORD_VALID
	result = 0xffffffff00000000ull;
#else
	result = 0x00000000ffffffffull;
#endif
#endif /* FULL_64_BIT_VALID */

	k = j & 0x00ff;	/* index to saved data */

/* calculate 2 sboxes of pass 12, and compare them with precalculated data */
/* if that succeeds (47%), go on to next key.  first fails 5%, second fails 48% */
	cached_inverse_data = &work->Cached_Results[k];

#if 0
	do {
	    return_val = DO_S1 (&work->pass_12_work);
	} while (return_val != required_return_value);
	do {
	    return_val = DO_S3 (&work->pass_12_work);
	} while (return_val != required_return_value);
#else
	do {
	    return_val = DO_S1_S3 (&work->pass_12_work);
	} while (return_val != required_return_value);
#endif

	result &= ((work->Right6.Data[8]  ^ ~cached_inverse_data->Cached_Data_8)
		 & (work->Right6.Data[16] ^ ~cached_inverse_data->Cached_Data_16)
		 & (work->Right6.Data[22] ^ ~cached_inverse_data->Cached_Data_22)
		 & (work->Right6.Data[30] ^ ~cached_inverse_data->Cached_Data_30));
	if (result == SLICE_0)
	{   stat_printf ("early 1\n");
	    continue;
	}

	result &= ((work->Right6.Data[5]  ^ ~cached_inverse_data->Cached_Data_5)
		 & (work->Right6.Data[15] ^ ~cached_inverse_data->Cached_Data_15)
		 & (work->Right6.Data[23] ^ ~cached_inverse_data->Cached_Data_23)
		 & (work->Right6.Data[29] ^ ~cached_inverse_data->Cached_Data_29));
	if (result == SLICE_0)
	{   stat_printf ("early 2\n");
	    continue;
	}

/* if that succeeds, calculate 1 forward, 1 backward, compare */
/* if that succeeds (3%), go on to next key.  fails 44% */
 	calculated_inverse_data = &work->Right13[k].Data[0];

/* Set up operand pointers for the inverse sbox calculation */
	work->pass_12_forward_14_back_work[0].Source = &work->Right14[k].Data[0];
	work->pass_12_forward_14_back_work[0].Merge = &work->Right15[k].Data[0];
	work->pass_12_forward_14_back_work[0].Dest = &work->Right13[k].Data[0];

	do {
	    return_val = DO_S4 (&work->pass_12_forward_14_back_work[0]);
	} while (return_val != required_return_value);
	result &= ((work->Right6.Data[0]  ^ ~calculated_inverse_data[0])
		 & (work->Right6.Data[9]  ^ ~calculated_inverse_data[9])
		 & (work->Right6.Data[19] ^ ~calculated_inverse_data[19])
		 & (work->Right6.Data[25] ^ ~calculated_inverse_data[25]));

	if (result == SLICE_0)
	{   stat_printf ("early 3\n");
	    continue;
	}

/* if that succeeds, calculate 1 forward, 1 backward, compare */
/* if that succeeds (0.3%), go on to next key.  fails 2.7% */
	do {
	    return_val = DO_S2 (&work->pass_12_forward_14_back_work[0]);
	} while (return_val != required_return_value);
	result &= ((work->Right6.Data[1]  ^ ~calculated_inverse_data[1])
		 & (work->Right6.Data[12] ^ ~calculated_inverse_data[12])
		 & (work->Right6.Data[17] ^ ~calculated_inverse_data[17])
		 & (work->Right6.Data[27] ^ ~calculated_inverse_data[27]));

	if (result == SLICE_0)
	{   stat_printf ("early 4\n");
	    continue;
	}

/* if that succeeds, calculate 12, 13 forward, compare to saved data */
/* The saved data is the results of inverse pass 15 */
/* if that fails, go on to next key */
	do {
	    return_val = DO_ALL_FANCY (&work->pass_12_13_work[0], 0);
	} while (return_val != required_return_value);

	calculated_inverse_data = &work->Right14[k].Data[0];

	for (i = 0; i < 32; i++)
	    result &= (work->Right7.Data[i] ^ ~calculated_inverse_data[i]);
	if (result == SLICE_0)
	{   stat_printf ("early 5\n");
	    continue;
	}

/* if that succeeds, calculate 14 forward, compare to saved data */
/* The saved data is the results of inverse pass 16 */
/* if that fails, go on to next key */
	do {
	    return_val = DO_ALL_FANCY (&work->pass_14_work, 0);
	} while (return_val != required_return_value);

	calculated_inverse_data = &work->Right15[k].Data[0];

	for (i = 0; i < 32; i++)
	    result &= (work->Right8.Data[i] ^ ~calculated_inverse_data[i]);
	if (result == SLICE_0)
	{   stat_printf ("early 6\n");
	    continue;
	}

/* success, so break */
	break;

#ifdef NO_FANCY_TEST
/* extra code to let above stuff be commented out to find bugs */

/* if that succeeds, calculate 12, 13 forward, compare to saved data */
/* The saved data is the results of inverse pass 15 */
/* if that fails, go on to next key */
	do {
	    return_val = DO_ALL_FANCY (&work->pass_12_15_work[0], 0);
	} while (return_val != required_return_value);

	for (i = 0; i < 32; i++)
	    result &= (work->Right9.Data[i] ^ ~work->Left16.Data[i]);
	if (result == SLICE_0)
	{   stat_printf ("early 5\n");
	    continue;
	}

/* if that succeeds, calculate 14 forward, compare to saved data */
/* The saved data is the results of inverse pass 16 */
/* if that fails, go on to next key */
	do {
	    return_val = DO_ALL_FANCY (&work->pass_16_work, 0);
	} while (return_val != required_return_value);

	for (i = 0; i < 32; i++)
	    result &= (work->Right4.Data[i] ^ ~work->Right16.Data[i]);
	if (result == SLICE_0)
	{   stat_printf ("early 6\n");
	    continue;
	}

/* success, so break */
	break;
#endif /* NO_FANCY_TEST */
    }

/* look at loop index to decide what happened */
    if (j == (256 * 64))
    {	free ((void *)work);
	return (SLICE_0);	/* nothing found */
    }

/* update the key in the caller's data to indicate successful key found */
    for (i = 0; i < 56; i++)
	key_base[i] = work->Keys.Keys[i];

    free ((void *)work);
    return (result); /* bit mask indicating which key is the good one */
}
/* end of des-ultra-crunch.c */
