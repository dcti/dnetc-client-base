// Bitslice driver copyright (C) 1998 Andrew Meggs / casa de la cabeza explosiva
// All rights reserved. A non-transferrable, royalty free license to this code
// is granted to distributed.net for use exclusively in the DES Challenge II,
// but ownership remains with the author.
//
//
// MMX implementation by :
//
// Bruce Ford <b.ford@qut.edu.au>
// Rémi Guyomarch <rguyom@mail.dotcom.fr>
//
//
//		BrydDES		old MMX 	  new MMX
// PII-400:   1910 kkeys/s    3150 kkeys/s      3200 kkeys/s
// P5-200 :    995 kkeys/s    1478 kkeys/s      1650 kkeys/s
// K6-200 :    960 kkeys/s    1118 kkeys/s      1230 kkeys/s
// 6x86MX :    562 kkeys/s     652 kkeys/s	  ??

//
// $Log: deseval-meggs3-mmx.cpp,v $
// Revision 1.1.2.1  2001/01/21 16:57:16  cyp
// reorg
//
// Revision 1.9  1999/01/23 21:39:05  patrick
//
// OS2 also needs call functions with a leading '_'
//
// Revision 1.8  1998/11/08 12:57:19  silby
// Changes to make client buildable on freebsd 2.x boxes as well as 3.x boxes.
//
// Revision 1.7  1998/11/08 07:41:40  dbaker
// Changes to make MMX_BITSLICE client buildable on freebsd
//
// Revision 1.6  1998/07/14 10:43:36  remi
// Added support for a minimum timeslice value of 16 instead of 20 when
// using BIT_64, which is needed by MMX_BITSLICER. Will help some platforms
// like Netware or Win16. I added support in deseval-meggs3.cpp, but it's just
// for completness, Alphas don't need this patch.
//
// Important note : this patch **WON'T** work with deseval-meggs2.cpp, but
// according to the configure script it isn't used anymore. If you compile
// des-slice-meggs.cpp and deseval-meggs2.cpp with BIT_64 and
// BITSLICER_WITH_LESS_BITS, the DES self-test will fail.
//
// Revision 1.5  1998/07/13 00:37:29  silby
// Changes to make MMX_BITSLICE client buildable on freebsd
//
// Revision 1.4  1998/07/12 05:29:14  fordbr
// Replaced sboxes 1, 2 and 7 with Kwan versions
// Now 1876 kkeys/s on a P5-200MMX
//
// Revision 1.3  1998/07/09 21:01:07  remi
// Fixed sboxes names in inline assembly for Linux-aout.
//
// Revision 1.2  1998/07/08 23:37:35  remi
// Added support for aout targets (.align).
// Tweaked $Id: deseval-meggs3-mmx.cpp,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $.
//
// Revision 1.1  1998/07/08 15:49:36  remi
// MMX bitslicer integration.
//
//

#if (!defined(lint) && defined(__showids__))
const char *deseval_meggs3_mmx_cpp(void) {
return "@(#)$Id: deseval-meggs3-mmx.cpp,v 1.1.2.1 2001/01/21 16:57:16 cyp Exp $"; }
#endif

#include <stdlib.h>
#include "cputypes.h"
#include "sboxes-mmx.h"

#if defined(__ELF__)
	#define ALIGN4  ".align 4"
	#define ALIGN16 ".align 16"
#else
	#define ALIGN4  ".align 2"
	#define ALIGN16 ".align 4"
#endif

#define load_params(_a1,_a2,_a3,_a4,_a5,_a6,_i0,_o0,_i1,_o1,_i2,_o2,_i3,_o3) \
	mmxParams->older.a1 = _a1;		\
	mmxParams->older.a2 = _a2;		\
	mmxParams->older.a3 = _a3;		\
	mmxParams->older.a4 = _a4;		\
	mmxParams->older.a5 = _a5;		\
	mmxParams->older.a6 = _a6;		\
 	mmxParams->older.i0 = _i0;		\
	mmxParams->older.i1 = _i1;		\
	mmxParams->older.i2 = _i2;		\
	mmxParams->older.i3 = _i3;		\
 	mmxParams->older.o0 = &(_o0);		\
	mmxParams->older.o1 = &(_o1);		\
	mmxParams->older.o2 = &(_o2);		\
	mmxParams->older.o3 = &(_o3)

#define load_kwan_params(_a1,_a2,_a3,_a4,_a5,_a6,_i0,_i1,_i2,_i3) \
	asm ("movq %0, %%mm0" : :"m"(_a1));	\
 	mmxParams->newer.i0 = _i0;		\
	asm ("movq %0, %%mm1" : :"m"(_a2));	\
	mmxParams->newer.i1 = _i1;		\
	asm ("movq %0, %%mm2" : :"m"(_a3));	\
	mmxParams->newer.i2 = _i2;		\
	asm ("movq %0, %%mm3" : :"m"(_a4));	\
	mmxParams->newer.i3 = _i3;		\
	asm ("movq %0, %%mm4" : :"m"(_a5));	\
	asm ("movq %0, %%mm5" : :"m"(_a6));

#define s1(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs1(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s2(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs2(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s3(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs3(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s4(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs4(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s5(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs5(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s6(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs6(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s7(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs7(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)
#define s8(i0,i1,i2,i3,i4,i5,out0,out1,out2,out3) xs8(i0,i1,i2,i3,i4,i5,out0,out0,out1,out1,out2,out2,out3,out3)

#define xs1(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs1_kwan (mmxParams); \
    asm ("movq %%mm5, %0" : "=m"(o0)); \
    asm ("movq %%mm7, %0" : "=m"(o1)); \
    asm ("movq %%mm2, %0" : "=m"(o2)); \
    asm ("movq %%mm0, %0" : "=m"(o3)); \
} while (0)

#define xs2(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs2_kwan (mmxParams); \
    asm ("movq %%mm1, %0" : "=m"(o0)); \
    asm ("movq %%mm5, %0" : "=m"(o1)); \
    asm ("movq %%mm7, %0" : "=m"(o2)); \
    asm ("movq %%mm2, %0" : "=m"(o3)); \
} while (0)

#define xs3(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs3_kwan (mmxParams); \
    asm ("movq %%mm2, %0" : "=m"(o0)); \
    asm ("movq %%mm6, %0" : "=m"(o1)); \
    asm ("movq %%mm3, %0" : "=m"(o2)); \
    asm ("movq %%mm7, %0" : "=m"(o3)); \
} while (0)

#define xs4(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs4_kwan (mmxParams); \
    asm ("movq %%mm1, %0" : "=m"(o0)); \
    asm ("movq %%mm0, %0" : "=m"(o1)); \
    asm ("movq %%mm6, %0" : "=m"(o2)); \
    asm ("movq %%mm5, %0" : "=m"(o3)); \
} while (0)

#define xs5(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs5_kwan (mmxParams); \
    asm ("movq %%mm5, %0" : "=m"(o0)); \
    asm ("movq %%mm7, %0" : "=m"(o1)); \
    asm ("movq %%mm6, %0" : "=m"(o2)); \
    asm ("movq %%mm4, %0" : "=m"(o3)); \
} while (0)

#define xs6(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs6_kwan (mmxParams); \
    asm ("movq %%mm0, %0" : "=m"(o0)); \
    asm ("movq %%mm1, %0" : "=m"(o1)); \
    asm ("movq %%mm2, %0" : "=m"(o2)); \
    asm ("movq %%mm4, %0" : "=m"(o3)); \
} while (0)

#define xs7(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs7_kwan (mmxParams); \
    asm ("movq %%mm7, %0" : "=m"(o0)); \
    asm ("movq %%mm1, %0" : "=m"(o1)); \
    asm ("movq %%mm3, %0" : "=m"(o2)); \
    asm ("movq %%mm0, %0" : "=m"(o3)); \
} while (0)

#define xs8(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) \
do { \
    load_kwan_params (a1,a2,a3,a4,a5,a6,i0,i1,i2,i3); \
    mmxs8_kwan (mmxParams); \
    asm ("movq %%mm6, %0" : "=m"(o0)); \
    asm ("movq %%mm2, %0" : "=m"(o1)); \
    asm ("movq %%mm5, %0" : "=m"(o2)); \
    asm ("movq %%mm1, %0" : "=m"(o3)); \
} while (0)

//#define xs1(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs1 (mmxParams); } while (0)
//#define xs2(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs2 (mmxParams); } while (0)
//#define xs3(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs3 (mmxParams); } while (0)
//#define xs4(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs4 (mmxParams); } while (0)
//#define xs5(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs5 (mmxParams); } while (0)
//#define xs6(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs6 (mmxParams); } while (0)
//#define xs7(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs7 (mmxParams); } while (0)
//#define xs8(a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3) do { load_params (a1,a2,a3,a4,a5,a6,i0,o0,i1,o1,i2,o2,i3,o3); mmxs8 (mmxParams); } while (0)

// CYGWIN32 because it's my debugging & benchmarking platform
#if ((CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_DOSWIN16) || \
     (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_OS2) || \
    (((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD)) && !defined(__ELF__)) || defined(__CYGWIN32__))
#define CALL(sfunc) "call _"#sfunc
#else
#define CALL(sfunc) "call "#sfunc
#endif

//-------------------------------------------------------------------
//-------------------------------------------------------------------
static const unsigned int keybits[ 13 /*rounds*/ * 8 /*boxes*/ * 6 /*bits*/ ] = {
	// round 1
		/* s1 */   47, 11, 26,  3, 13, 41,
		/* s2 */   27,  6, 54, 48, 39, 19,
		/* s3 */   53, 25, 33, 34, 17,  5,
		/* s4 */    4, 55, 24, 32, 40, 20,
		/* s5 */   36, 31, 21,  8, 23, 52,
		/* s6 */   14, 29, 51,  9, 35, 30,
		/* s7 */    2, 37, 22,  0, 42, 38,
		/* s8 */   16, 43, 44,  1,  7, 28,
	// round 2
		/* s1 */   54, 18, 33, 10, 20, 48,
		/* s2 */   34, 13,  4, 55, 46, 26,
		/* s3 */    3, 32, 40, 41, 24, 12,
		/* s4 */   11,  5,  6, 39, 47, 27,
		/* s5 */   43, 38, 28, 15, 30,  0,
		/* s6 */   21, 36, 31, 16, 42, 37,
		/* s7 */    9, 44, 29,  7, 49, 45,
		/* s8 */   23, 50, 51,  8, 14, 35,
	// round 3
		/* s1 */   11, 32, 47, 24, 34,  5,
		/* s2 */   48, 27, 18, 12,  3, 40,
		/* s3 */   17, 46, 54, 55, 13, 26,
		/* s4 */   25, 19, 20, 53,  4, 41,
		/* s5 */    2, 52, 42, 29, 44, 14,
		/* s6 */   35, 50, 45, 30,  1, 51,
		/* s7 */   23, 31, 43, 21,  8,  0,
		/* s8 */   37,  9, 38, 22, 28, 49,
	// round 4
		/* s1 */   25, 46,  4, 13, 48, 19,
		/* s2 */    5, 41, 32, 26, 17, 54,
		/* s3 */    6,  3, 11, 12, 27, 40,
		/* s4 */   39, 33, 34, 10, 18, 55,
		/* s5 */   16,  7,  1, 43, 31, 28,
		/* s6 */   49,  9,  0, 44, 15, 38,
		/* s7 */   37, 45,  2, 35, 22, 14,
		/* s8 */   51, 23, 52, 36, 42,  8,
	// round 5
		/* s1 */   39,  3, 18, 27,  5, 33,
		/* s2 */   19, 55, 46, 40,  6, 11,
		/* s3 */   20, 17, 25, 26, 41, 54,
		/* s4 */   53, 47, 48, 24, 32, 12,
		/* s5 */   30, 21, 15,  2, 45, 42,
		/* s6 */    8, 23, 14, 31, 29, 52,
		/* s7 */   51,  0, 16, 49, 36, 28,
		/* s8 */   38, 37,  7, 50,  1, 22,
	// round 6
		/* s1 */   53, 17, 32, 41, 19, 47,
		/* s2 */   33, 12,  3, 54, 20, 25,
		/* s3 */   34,  6, 39, 40, 55, 11,
		/* s4 */   10,  4,  5, 13, 46, 26,
		/* s5 */   44, 35, 29, 16,  0,  1,
		/* s6 */   22, 37, 28, 45, 43,  7,
		/* s7 */   38, 14, 30,  8, 50, 42,
		/* s8 */   52, 51, 21,  9, 15, 36,
	// round 7
		/* s1 */   10,  6, 46, 55, 33,  4,
		/* s2 */   47, 26, 17, 11, 34, 39,
		/* s3 */   48, 20, 53, 54, 12, 25,
		/* s4 */   24, 18, 19, 27,  3, 40,
		/* s5 */   31, 49, 43, 30, 14, 15,
		/* s6 */   36, 51, 42,  0,  2, 21,
		/* s7 */   52, 28, 44, 22,  9,  1,
		/* s8 */    7, 38, 35, 23, 29, 50,
	// round 8
		/* s1 */   24, 20,  3, 12, 47, 18,
		/* s2 */    4, 40,  6, 25, 48, 53,
		/* s3 */    5, 34, 10, 11, 26, 39,
		/* s4 */   13, 32, 33, 41, 17, 54,
		/* s5 */   45,  8,  2, 44, 28, 29,
		/* s6 */   50, 38,  1, 14, 16, 35,
		/* s7 */    7, 42, 31, 36, 23, 15,
		/* s8 */   21, 52, 49, 37, 43,  9,
	// round 9
		/* s1 */    6, 27, 10, 19, 54, 25,
		/* s2 */   11, 47, 13, 32, 55,  3,
		/* s3 */   12, 41, 17, 18, 33, 46,
		/* s4 */   20, 39, 40, 48, 24,  4,
		/* s5 */   52, 15,  9, 51, 35, 36,
		/* s6 */    2, 45,  8, 21, 23, 42,
		/* s7 */   14, 49, 38, 43, 30, 22,
		/* s8 */   28,  0,  1, 44, 50, 16,
	// round 10
		/* s1 */   20, 41, 24, 33, 11, 39,
		/* s2 */   25,  4, 27, 46, 12, 17,
		/* s3 */   26, 55,  6, 32, 47,  3,
		/* s4 */   34, 53, 54,  5, 13, 18,
		/* s5 */    7, 29, 23, 38, 49, 50,
		/* s6 */   16,  0, 22, 35, 37,  1,
		/* s7 */   28,  8, 52,  2, 44, 36,
		/* s8 */   42, 14, 15, 31,  9, 30,
	// round 11
		/* s1 */   34, 55, 13, 47, 25, 53,
		/* s2 */   39, 18, 41,  3, 26,  6,
		/* s3 */   40, 12, 20, 46,  4, 17,
		/* s4 */   48, 10, 11, 19, 27, 32,
		/* s5 */   21, 43, 37, 52,  8,  9,
		/* s6 */   30, 14, 36, 49, 51, 15,
		/* s7 */   42, 22,  7, 16, 31, 50,
		/* s8 */    1, 28, 29, 45, 23, 44,
		
	// round 15
		/* s1 */   33, 54, 12, 46, 24, 27,
		/* s2 */   13, 17, 40, 34, 25,  5,
		/* s3 */   39, 11, 19, 20,  3, 48,
		/* s4 */   47, 41, 10, 18, 26,  6,
		/* s5 */   22, 44,  7, 49,  9, 38,
		/* s6 */    0, 15, 37, 50, 21, 16,
		/* s7 */   43, 23,  8, 45, 28, 51,
		/* s8 */    2, 29, 30, 42, 52, 14,
	// round 16
		/* s1 */   40,  4, 19, 53,  6, 34,
		/* s2 */   20, 24, 47, 41, 32, 12,
		/* s3 */   46, 18, 26, 27, 10, 55,
		/* s4 */   54, 48, 17, 25, 33, 13,
		/* s5 */   29, 51, 14,  1, 16, 45,
		/* s6 */    7, 22, 44,  2, 28, 23,
		/* s7 */   50, 30, 15, 52, 35, 31,
		/* s8 */    9, 36, 37, 49,  0, 21,
			};


static const unsigned char initPL_source[] = {
    6,14,22,30,38,46,54,62,4,12,20,28,36,44,52,60,2,10,18,26,34,42,50,58,0,8,16,24,32,40,48,56};
static const unsigned char initCL_source[] = {
    5,3,51,49,37,25,15,11,59,61,41,47,9,27,13,7,63,45,1,23,31,33,21,19,57,29,43,55,39,17,53,35};
static const unsigned char initCLCR_dest[] = {
    8,16,22,30,12,27,1,17,23,15,29,5,25,19,9,0,7,13,24,2,3,28,10,18,31,11,21,6,4,26,14,20};

// eax = mmxParams
// ebx = S
// ecx = (used)
// edx = (used)
// esi = K
// edi = kb
// ebp = D

#define mmNOT  " 0(%%eax)"
#define a1     " 8(%%eax)"
#define a2     "16(%%eax)"
#define a3     "24(%%eax)"
#define a4     "32(%%eax)"
#define a5     "40(%%eax)"
#define a6     "48(%%eax)"
#define i0     "56(%%eax)"
#define i1     "64(%%eax)"
#define i2     "72(%%eax)"
#define i3     "80(%%eax)"
#define o0     "88(%%eax)"
#define o1     "92(%%eax)"
#define o2     "96(%%eax)"
#define o3     "100(%%eax)"
#define locals "104(%%eax)"

#define kmmNOT " 0(%%eax)"
#define ki0    " 8(%%eax)"
#define ki1    "16(%%eax)"
#define ki2    "24(%%eax)"
#define ki3    "32(%%eax)"

#define load(S1,S2,S3,S4,S5,S6,  K1,K2,K3,K4,K5,K6,  MD0,MD1,MD2,MD3) \
"	movl	"K1"*4(%%edi), %%ecx	# ecx = kb[ 0]
	movl	"K2"*4(%%edi), %%edx
	movq	"S1"*8(%%ebx), %%mm0	# mm0 = S[31]
	movq	"S2"*8(%%ebx), %%mm1
	pxor	(%%esi,%%ecx,8), %%mm0	# mm0 = S[31]^K[kb[ 0]]
	pxor	(%%esi,%%edx,8), %%mm1
	movq	%%mm0, "a1"		# a1 = S[31]^K[kb[ 0]]
	movq	%%mm1, "a2"		# a2 = S[0]^[K[kb[1]]

	movl	"K3"*4(%%edi), %%ecx
	movl	"K4"*4(%%edi), %%edx
	movq	"S3"*8(%%ebx), %%mm0
	movq	"S4"*8(%%ebx), %%mm1
	pxor	(%%esi,%%ecx,8), %%mm0
	pxor	(%%esi,%%edx,8), %%mm1
	movq	%%mm0, "a3"		# a3 = S[1]^[K[kb[2]]
	movq	%%mm1, "a4"		# a4 = S[2]^[K[kb[3]]

	movl	"K5"*4(%%edi), %%ecx
	movl	"K6"*4(%%edi), %%edx
	movq	"S5"*8(%%ebx), %%mm0
	movq	"S6"*8(%%ebx), %%mm1
	pxor	(%%esi,%%ecx,8), %%mm0
	pxor	(%%esi,%%edx,8), %%mm1
	movq	%%mm0, "a5"		# a5 = S[3]^[K[kb[4]]
	movq	%%mm1, "a6"		# a6 = S[4]^[K[kb[5]]

	movl	%3, %%edx		# edx = &M
	leal	"MD0"*8(%%ebp), %%ecx
	movq	"MD0"*8(%%edx), %%mm1
	movl	%%ecx,  "o0"
	leal	"MD1"*8(%%ebp), %%ecx
	movq	%%mm1, "i0"
	movq	"MD1"*8(%%edx), %%mm1
	movl	%%ecx,  "o1"
	leal	"MD2"*8(%%ebp), %%ecx
	movq	%%mm1, "i1"
	movq	"MD2"*8(%%edx), %%mm1
	movl	%%ecx, "o2"
	leal	"MD3"*8(%%ebp), %%ecx
	movq	%%mm1, "i2"
	movq	"MD3"*8(%%edx), %%mm1
	movl	%%ecx, "o3"
	movq	%%mm1, "i3" \n"


#define call_kwan(mmfunc, S1,S2,S3,S4,S5,S6,  K1,K2,K3,K4,K5,K6, MD0,MD1,MD2,MD3 ) \
"	movl	"K1"*4(%%edi), %%ecx	# ecx = kb[K1]
	movl	"K2"*4(%%edi), %%edx
	movq	"S1"*8(%%ebx), %%mm0	# mm0 = S[S1]
	movq	"S2"*8(%%ebx), %%mm1
	pxor	(%%esi,%%ecx,8), %%mm0	# mm0 = S[S1]^K[kb[K1]]
	pxor	(%%esi,%%edx,8), %%mm1

	movl	"K3"*4(%%edi), %%ecx
	movl	"K4"*4(%%edi), %%edx
	movq	"S3"*8(%%ebx), %%mm2
	movq	"S4"*8(%%ebx), %%mm3
	pxor	(%%esi,%%ecx,8), %%mm2
	pxor	(%%esi,%%edx,8), %%mm3

	movl	"K6"*4(%%edi), %%edx
	movl	"K5"*4(%%edi), %%ecx
	movq	"S6"*8(%%ebx), %%mm5
	movq	"S5"*8(%%ebx), %%mm4
	pxor	(%%esi,%%edx,8), %%mm5
	movl	%3, %%edx		# edx = &M
	pxor	(%%esi,%%ecx,8), %%mm4

	movq	"MD0"*8(%%edx), %%mm6
	movq	"MD1"*8(%%edx), %%mm7
	movq	%%mm6, "ki0"
	movq	%%mm7, "ki1"
	movq	"MD2"*8(%%edx), %%mm6
	movq	"MD3"*8(%%edx), %%mm7
	movq	%%mm6, "ki2"
	movq	%%mm7, "ki3"

	"CALL(mmfunc)" \n"

#define save_kwan1(MD0,MD1,MD2,MD3) \
"	movq	%%mm5, "MD0"*8(%%ebp)
	movq	%%mm7, "MD1"*8(%%ebp)
	movq	%%mm2, "MD2"*8(%%ebp)
	movq	%%mm0, "MD3"*8(%%ebp) \n"

#define save_kwan2(MD0,MD1,MD2,MD3) \
"	movq	%%mm1, "MD0"*8(%%ebp)
	movq	%%mm5, "MD1"*8(%%ebp)
	movq	%%mm7, "MD2"*8(%%ebp)
	movq	%%mm2, "MD3"*8(%%ebp) \n"

#define save_kwan3(MD0,MD1,MD2,MD3) \
"	movq	%%mm2, "MD0"*8(%%ebp)
	movq	%%mm6, "MD1"*8(%%ebp)
	movq	%%mm3, "MD2"*8(%%ebp)
	movq	%%mm7, "MD3"*8(%%ebp) \n"

#define save_kwan4(MD0,MD1,MD2,MD3) \
"	movq	%%mm1, "MD0"*8(%%ebp)
	movq	%%mm0, "MD1"*8(%%ebp)
	movq	%%mm6, "MD2"*8(%%ebp)
	movq	%%mm5, "MD3"*8(%%ebp) \n"

#define save_kwan5(MD0,MD1,MD2,MD3) \
"	movq	%%mm5, "MD0"*8(%%ebp)
	movq	%%mm7, "MD1"*8(%%ebp)
	movq	%%mm6, "MD2"*8(%%ebp)
	movq	%%mm4, "MD3"*8(%%ebp) \n"

#define save_kwan6(MD0,MD1,MD2,MD3) \
"	movq	%%mm0, "MD0"*8(%%ebp)
	movq	%%mm1, "MD1"*8(%%ebp)
	movq	%%mm2, "MD2"*8(%%ebp)
	movq	%%mm4, "MD3"*8(%%ebp) \n"

#define save_kwan7(MD0,MD1,MD2,MD3) \
"	movq	%%mm7, "MD0"*8(%%ebp)
	movq	%%mm1, "MD1"*8(%%ebp)
	movq	%%mm3, "MD2"*8(%%ebp)
	movq	%%mm0, "MD3"*8(%%ebp) \n"

#define save_kwan8(MD0,MD1,MD2,MD3) \
"	movq	%%mm6, "MD0"*8(%%ebp)
	movq	%%mm2, "MD1"*8(%%ebp)
	movq	%%mm5, "MD2"*8(%%ebp)
	movq	%%mm1, "MD3"*8(%%ebp) \n"

static void partialround( slice S[32], slice M[32], slice D[32], slice K[56], int ks, int select, stMmxParams *mmxParams )
{
    unsigned int save_ebp;
    
    asm ("
	movl	%%ebp, %0
	leal	%8(,%%edi,4), %%edi	# const unsigned int *kb = keybits + ks;
	movl	%2, %%ebp

	testb	$0x80, %1
	jz	__and_0x80
"	call_kwan (mmxs1_kwan,"31","0","1","2","3","4",  "0","1","2","3","4","5",  "8","16","22","30")"
"	save_kwan1 ("8","16","22","30")"
"ALIGN4"
__and_0x80:
	testb	$0x40, %1
	jz	__and_0x40
"	call_kwan (mmxs2_kwan,"3","4","5","6","7","8",  "6","7","8","9","10","11",  "12","27","1","17")"
"	save_kwan2 ("12","27","1","17")"
"ALIGN4"
__and_0x40:
	testb	$0x20, %1
	jz	__and_0x20
"	call_kwan (mmxs3_kwan,"7","8","9","10","11","12",  "12","13","14","15","16","17", "23","15","29","5")"
"	save_kwan3 ("23","15","29","5")"
"ALIGN4"
__and_0x20:
	addl	$11*8, %%ebx		# helps reduce code size
"	call_kwan (mmxs4_kwan,"0","1","2","3","4","5",  "18","19","20","21","22","23",  "25","19","9","0")"
"	save_kwan4 ("25","19","9","0")"

	testb	$0x08, %1
	jz	__and_0x08
"	call_kwan (mmxs5_kwan,"4","5","6","7","8","9",  "24","25","26","27","28","29",  "7","13","24","2")"
"	save_kwan5 ("7","13","24","2")"
"ALIGN4"
__and_0x08:
	addl	$30*4, %%edi		# helps reduce code size
	testb	$0x04, %1
	jz	__and_0x04
"	call_kwan (mmxs6_kwan,"8","9","10","11","12","13",  "0","1","2","3","4","5",  "3","28","10","18")"
"	save_kwan6 ("3","28","10","18")"
"ALIGN4"
__and_0x04:
	addl	$12*8, %%ebx		# helps reduce code size
	testb	$0x02, %1
	jz	__and_0x02
"	call_kwan (mmxs7_kwan,"0","1","2","3","4","5",  "6","7","8","9","10","11",  "31","11","21","6")"
"	save_kwan7 ("31","11","21","6")"
"ALIGN4"
__and_0x02:
	testb	$0x01, %1
	jz	__and_0x01
"	call_kwan (mmxs8_kwan, "4","5","6","7","8","-23",  "12","13","14","15","16","17",  "4","26","14","20")"
"	save_kwan8 ("4","26","14","20")"
"ALIGN4"
__and_0x01:

	movl	%0, %%ebp \n"

     : "=m"(save_ebp)
     : "m"(select),
       "m"(D),
       "m"(M),
       "a"(mmxParams),
       "b"(S),
       "S"(K),
       "D"(ks),
       "m"(keybits)
     : "%eax","%ebx","%ecx","%edx","%edi","%cc");
}

/*
void partialround( slice S[32], slice M[32], slice D[32], slice K[56], int ks, int select )
{
    const unsigned char *kb = keybits + ks;
	
    if (select & 0x80) 
	xs1 (S[31]^K[kb[ 0]], S[ 0]^K[kb[ 1]], S[ 1]^K[kb[ 2]], 
	     S[ 2]^K[kb[ 3]], S[ 3]^K[kb[ 4]], S[ 4]^K[kb[ 5]],
	     M[ 8], D[ 8], M[16], D[16],
	     M[22], D[22], M[30], D[30]);
	    
    if (select & 0x40) 
	xs2 (S[ 3]^K[kb[ 6]], S[ 4]^K[kb[ 7]], S[ 5]^K[kb[ 8]], 
	     S[ 6]^K[kb[ 9]], S[ 7]^K[kb[10]], S[ 8]^K[kb[11]],
	     M[12], D[12], M[27], D[27],
	     M[ 1], D[ 1], M[17], D[17]);
	    
    if (select & 0x20) 
	xs3 (S[ 7]^K[kb[12]], S[ 8]^K[kb[13]], S[ 9]^K[kb[14]], 
	     S[10]^K[kb[15]], S[11]^K[kb[16]], S[12]^K[kb[17]],
	     M[23], D[23], M[15], D[15],
	     M[29], D[29], M[ 5], D[ 5]);
	    
    //if (select & 0x10)
	xs4 (S[11]^K[kb[18]], S[12]^K[kb[19]], S[13]^K[kb[20]], 
	     S[14]^K[kb[21]], S[15]^K[kb[22]], S[16]^K[kb[23]],
	     M[25], D[25], M[19], D[19],
	     M[ 9], D[ 9], M[ 0], D[ 0]);
	    
    if (select & 0x08)
	xs5 (S[15]^K[kb[24]], S[16]^K[kb[25]], S[17]^K[kb[26]], 
	     S[18]^K[kb[27]], S[19]^K[kb[28]], S[20]^K[kb[29]],
	     M[ 7], D[ 7], M[13], D[13],
	     M[24], D[24], M[ 2], D[ 2]);
	    
    if (select & 0x04)
	xs6 (S[19]^K[kb[30]], S[20]^K[kb[31]], S[21]^K[kb[32]], 
	     S[22]^K[kb[33]], S[23]^K[kb[34]], S[24]^K[kb[35]],
	     M[ 3], D[ 3], M[28], D[28],
	     M[10], D[10], M[18], D[18]);
	    
    if (select & 0x02) 
	xs7 (S[23]^K[kb[36]], S[24]^K[kb[37]], S[25]^K[kb[38]], 
	     S[26]^K[kb[39]], S[27]^K[kb[40]], S[28]^K[kb[41]],
	     M[31], D[31], M[11], D[11],
	     M[21], D[21], M[ 6], D[ 6]);
	    
    if (select & 0x01)
	xs8 (S[27]^K[kb[42]], S[28]^K[kb[43]], S[29]^K[kb[44]], 
	     S[30]^K[kb[45]], S[31]^K[kb[46]], S[ 0]^K[kb[47]],
	     M[ 4], D[ 4], M[26], D[26],
	     M[14], D[14], M[20], D[20]);
}*/

static void multiround( slice S[32], slice N[32], slice M[32], slice D[32], slice K[56], stMmxParams *mmxParams )
{
    unsigned int i, save_ebp;
    const unsigned int *kb = keybits + 144;
	
    asm volatile ("
	movl	%%ebp, %8
	movl	%2, %%ebp

	movl	$7, %0		# i = 7

"ALIGN16"
__loop_multiround:
 
"	call_kwan (mmxs1_kwan,"31","0","1","2","3","4",  "0","1","2","3","4","5",  "8","16","22","30")"
"	save_kwan1 ("8","16","22","30")"

"	call_kwan (mmxs2_kwan,"3", "4", "5", "6", "7", "8",   "6", "7", "8", "9","10","11",  "12","27", "1","17")"
"	save_kwan2 ("12","27", "1","17")"

"	call_kwan (mmxs3_kwan,"7", "8", "9","10","11","12", "12","13","14","15","16","17", "23","15","29","5")"
"	save_kwan3 ("23","15","29","5")"

	addl	$11*8, %%ebx	# helps reduce code size
"	call_kwan (mmxs4_kwan,"0","1","2","3","4","5", "18","19","20","21","22","23", "25","19","9","0")"
"	save_kwan4 ("25","19","9","0")"

"	call_kwan (mmxs5_kwan,"4","5","6","7","8","9", "24","25","26","27","28","29", "7","13","24","2")"
"	save_kwan5 ("7","13","24","2")"

	addl	$30*4, %%edi	# helps reduce code size
"	call_kwan (mmxs6_kwan,"8","9","10","11","12","13", "0","1","2","3","4","5", "3","28","10","18")"
"	save_kwan6 ("3","28","10","18")"

	addl	$12*8, %%ebx	# helps reduce code size
"	call_kwan (mmxs7_kwan,"0","1","2","3","4","5",  "6","7","8","9","10","11",  "31","11","21","6")"
"	save_kwan7 ("31","11","21","6")"

"	call_kwan (mmxs8_kwan,"4","5","6","7","8","-23", "12","13","14","15","16","17", "4","26","14","20")"
"	save_kwan8 ("4","26","14","20")"

	addl	$(48-30)*4, %%edi	# kb += 48;
	subl	$23*8, %%ebx

	movl	%%ebx, %3	# M = S;
	movl	%%ebp, %%ecx	# slice *swap = D;
	movl	%7,    %%ebp	# D = N;
	movl	%%ecx, %%ebx	# S = swap;
	movl	%%ecx, %7	# N = swap;

	decl	%0
	jg	__loop_multiround

	movl	%8, %%ebp \n"
         
     : "=m"(i)
       
     : "a"(mmxParams),
       "m"(D), // %ebp
       "m"(M),
       "b"(S),
       "S"(K),
       "D"(kb),
       "m"(N),
       "m"(save_ebp)           
     : "%ebx","%ecx","%edx","%edi");

}

#undef load
#undef load_kwan

#undef a1
#undef a2
#undef a3
#undef a4
#undef a5
#undef a6
#undef i0
#undef i1
#undef i2
#undef i3
#undef mmNOT
#undef o0
#undef o1
#undef o2
#undef o3

#undef ki0
#undef ki1
#undef ki2
#undef ki3
#undef kmmNOT

/*void multiround( slice S[32], slice N[32], slice M[32], slice D[32], slice K[56], stMmxParams *mmxParams )
{
    const unsigned int *kb = keybits + 144;
	
    for (int i = 4; i <= 10; ++i) {
        xs1 (S[31]^K[kb[ 0]], S[ 0]^K[kb[ 1]], S[ 1]^K[kb[ 2]],
             S[ 2]^K[kb[ 3]], S[ 3]^K[kb[ 4]], S[ 4]^K[kb[ 5]],
             M[ 8], D[ 8], M[16], D[16],
             M[22], D[22], M[30], D[30]);        
        xs2 (S[ 3]^K[kb[ 6]], S[ 4]^K[kb[ 7]], S[ 5]^K[kb[ 8]],
             S[ 6]^K[kb[ 9]], S[ 7]^K[kb[10]], S[ 8]^K[kb[11]],
             M[12], D[12], M[27], D[27],
             M[ 1], D[ 1], M[17], D[17]);
        xs3 (S[ 7]^K[kb[12]], S[ 8]^K[kb[13]], S[ 9]^K[kb[14]],
             S[10]^K[kb[15]], S[11]^K[kb[16]], S[12]^K[kb[17]],
             M[23], D[23], M[15], D[15],
             M[29], D[29], M[ 5], D[ 5]);
        xs4 (S[11]^K[kb[18]], S[12]^K[kb[19]], S[13]^K[kb[20]],
             S[14]^K[kb[21]], S[15]^K[kb[22]], S[16]^K[kb[23]],
             M[25], D[25], M[19], D[19],
             M[ 9], D[ 9], M[ 0], D[ 0]);
        xs5 (S[15]^K[kb[24]], S[16]^K[kb[25]], S[17]^K[kb[26]],
             S[18]^K[kb[27]], S[19]^K[kb[28]], S[20]^K[kb[29]],
             M[ 7], D[ 7], M[13], D[13],
             M[24], D[24], M[ 2], D[ 2]);
        xs6 (S[19]^K[kb[30]], S[20]^K[kb[31]], S[21]^K[kb[32]],
             S[22]^K[kb[33]], S[23]^K[kb[34]], S[24]^K[kb[35]],
             M[ 3], D[ 3], M[28], D[28],
             M[10], D[10], M[18], D[18]);
        xs7 (S[23]^K[kb[36]], S[24]^K[kb[37]], S[25]^K[kb[38]],
             S[26]^K[kb[39]], S[27]^K[kb[40]], S[28]^K[kb[41]],
             M[31], D[31], M[11], D[11],
             M[21], D[21], M[ 6], D[ 6]);
        xs8 (S[27]^K[kb[42]], S[28]^K[kb[43]], S[29]^K[kb[44]],
             S[30]^K[kb[45]], S[31]^K[kb[46]], S[ 0]^K[kb[47]],
             M[ 4], D[ 4], M[26], D[26],
             M[14], D[14], M[20], D[20]);
        kb += 48;
        slice *swap = D;
        D = N;
        M = S;
        S = N = swap;
    }
}*/



// test all combinations of the easily-toggled bits
slice whack16(slice *P, slice *C, slice *K)
{
    struct stWork {
        slice _R14[16][32];
        slice _L13[16][32];

        slice _R12_23[16], _R12_15[16], _R12_29[16], _R12__5[16];
        slice _R12__8[16], _R12_16[16], _R12_22[16], _R12_30[16];

        slice _L1[32];
        slice _R2[32];
        slice _L3[32];
        
        slice _R[32];
        slice _L[32];
        
        slice _PL[32], _PR[32], _CL[32], _CR[32];

        char filer[7];
        
    };// work;

    // malloc & free to be compatible with VC++
    // with stack allocation cygwin gcc generates a call to __alloca
    // which VC++ can't grok
    stWork *work = (stWork*) malloc (sizeof(stWork));
	
    slice t1, t2, t3, t4;
    slice save;
    slice result;
    int regen23;

    char mmxBuffer[sizeof(stMmxParams)+7];

    // how to align things on a 64-bit boundary
    stMmxParams *mmxParams = (stMmxParams*) ((((unsigned)&mmxBuffer)+7) & (unsigned)(-8));
    mmxParams->older.mmNOT = 0xFFFFFFFFFFFFFFFFull;

    stWork *work2 = (stWork*) ((((unsigned)work)+7) & (unsigned)(-8));
    slice (*R14)[16][32] = &(work2->_R14);
    slice (*L13)[16][32] = &(work2->_L13);
    slice *R12_23 = &(work2->_R12_23[0]);
    slice *R12_15 = &(work2->_R12_15[0]);
    slice *R12_29 = &(work2->_R12_29[0]);
    slice *R12__5 = &(work2->_R12__5[0]);
    slice *R12__8 = &(work2->_R12__8[0]);
    slice *R12_16 = &(work2->_R12_16[0]);
    slice *R12_22 = &(work2->_R12_22[0]);
    slice *R12_30 = &(work2->_R12_30[0]);
    slice *L1 = &(work2->_L1[0]);
    slice *R2 = &(work2->_R2[0]);
    slice *L3 = &(work2->_L3[0]);
    slice *R = &(work2->_R[0]);
    slice *L = &(work2->_L[0]);
    slice *PL = &(work2->_PL[0]);
    slice *PR = &(work2->_PR[0]);
    slice *CL = &(work2->_CL[0]);
    slice *CR = &(work2->_CR[0]);


    for (int i=0; i<32; i++) {
	int src = initPL_source[i];
	PL[i] = P[src];
	PR[i] = P[src+1];
	int dest = initCLCR_dest[i];
	src = initCL_source[i];
	CL[dest] = C[src];
	CR[dest] = C[src-1];
    }
	
    /* Assume key bits 3, 5, 8, 10, 11, 12, 15, 18, 42, 43, 45, 46, 49, 50
       are zero on entry. */
    /* Toggling order: Head: 10 18 46 49
       then tail: 03 11 42 05 43 08
       then rest of head: 12 15 45 50 */

#if !(defined(BITSLICER_WITH_LESS_BITS) && defined(BIT_64))
    int hs2 = 0; // secondary, outermost head-stepper
#endif
    int ts = 0; // tail-stepper (outer loop)
    int hs = 0; // head-stepper (inner loop)
    for (;;) {
	//whack_init_00();
	{
	    /* First set up the tails for the 16 head possibilities. We could gain speed
	       here by using the gray codes to derive one tail from another, but since this is
	       done only once per 1024 keys, I'm not bothering */
	    for (int i = 0; i < 16;) {
		// inv-round 16
		partialround( CL, CR, (*R14)[i], K, 576, 0xff, mmxParams );
		// inv-round 15
		partialround( (*R14)[i], CL, (*L13)[i], K, 528, 0xff, mmxParams );
		// inv-round 14
		xs1( (*L13)[i][31]^K[19], (*L13)[i][ 0]^K[40], (*L13)[i][ 1]^K[55], 
		     (*L13)[i][ 2]^K[32], (*L13)[i][ 3]^K[10], (*L13)[i][ 4]^K[13],
		     (*R14)[i][ 8], R12__8[i], (*R14)[i][16], R12_16[i], 
		     (*R14)[i][22], R12_22[i], (*R14)[i][30], R12_30[i] );
				
		xs3( (*L13)[i][ 7]^K[25], (*L13)[i][ 8]^K[54], (*L13)[i][ 9]^K[ 5], 
		     (*L13)[i][10]^K[ 6], (*L13)[i][11]^K[46], (*L13)[i][12]^K[34],
		     (*R14)[i][23], R12_23[i], (*R14)[i][15], R12_15[i], 
		     (*R14)[i][29], R12_29[i], (*R14)[i][ 5], R12__5[i] );
				
		++i;
		     if (i & 0x01) K[10] = ~K[10];
		else if (i & 0x02) K[18] = ~K[18];
		else if (i & 0x04) K[46] = ~K[46];
		else               K[49] = ~K[49];
	    }
			
	    // now initialize the head:
			
	    // round 1
			
	    partialround( PR, PL, L1, K, 0, 0xff, mmxParams );
			
	    // round 2
	    // regen23 |= 0xff00
			
	    // round 3
	    // regen23 |= 0x00ff
	}
	regen23 = 0xffff;
	for (;;) {
	    partialround( L1, PR, R2, K, 48, (regen23 >> 8) | 0x02, mmxParams );
	    for (;;) {
		// You'll undoubtedly be tempted to inline at least the call to multiround(). However,
		// the resulting explosion of temporary variables produced a significant slowdown
		// in the code from several compilers (Metrowerks and MrCpp) that had trouble doing
		// optimal register allocation, so by default these are out of line. To try inlining
		// them, define INLINEPARTIAL and/or INLINEMULTI when compiling.
		partialround( R2, L1, L3, K, 96, regen23, mmxParams );
		multiround( L3, L, R2, R, K, mmxParams );
		partialround( R, L, L, K, 480, 0xDE, mmxParams );
		{ // now we start checking the outputs...
		    // round 12
		    save = R[29];
		    s3 (L[ 7]^K[54], L[ 8]^K[26], L[ 9]^K[34],
                        L[10]^K[ 3], L[11]^K[18], L[12]^K[ 6],  R[23], R[15], R[29], R[5] );
		    result = ~(R[23] ^ R12_23[hs]) & ~(R[15] ^ R12_15[hs]) &
                             ~(R[29] ^ R12_29[hs]) & ~(R[5] ^ R12__5[hs]);
		    if (!result) goto stepper;

		    // get here 87.3% of the time for 32 bits, 98.4% for 64
		    // from round 11:
		    s8( R[27]^K[ 1], R[28]^K[28], save^K[29],
                        R[30]^K[45], R[31]^K[23], R[ 0]^K[44],  L[ 4], L[26], L[14], L[20] );
		    save = R[ 8];
		    s1 (L[31]^K[48], L[ 0]^K[12], L[ 1]^K[27],
                        L[ 2]^K[ 4], L[ 3]^K[39], L[ 4]^K[10],  R[ 8], R[16], R[22], R[30] );
		    result &= ~(R[ 8] ^ R12__8[hs]) & ~(R[16] ^ R12_16[hs]) &
                              ~(R[22] ^ R12_22[hs]) & ~(R[30] ^ R12_30[hs]);
		    if (!result) goto stepper;

		    // get here 11.8% of the time for 32, 22.2% for 64
		    // last of round 11:
		    s3 (R[ 7]^K[40], save^K[12], R[ 9]^K[20],
                        R[10]^K[46], R[11]^K[ 4], R[12]^K[17],  L[23], L[15], L[29], L[ 5] );
		    // no more cleverness, just finish inv-14 and 12 piece by piece and compare as we go
		    xs2 ((*L13)[hs][ 3] ^ K[24], (*L13)[hs][ 4] ^ K[ 3], (*L13)[hs][ 5] ^ K[26],
			 (*L13)[hs][ 6] ^ K[20], (*L13)[hs][ 7] ^ K[11], (*L13)[hs][ 8] ^ K[48],
			 (*R14)[hs][12], t1, (*R14)[hs][27], t2, 
			 (*R14)[hs][ 1], t3, (*R14)[hs][17], t4 );
		    s2 (L[ 3]^K[53], L[ 4]^K[32], L[ 5]^K[55], L[ 6]^K[17], L[ 7]^K[40], L[ 8]^K[20],
			R[12], R[27], R[ 1], R[17] );
		    result &= ~(R[12] ^ t1) & ~(R[27] ^ t2) & ~(R[ 1] ^ t3) & ~(R[17] ^ t4);
		    if (!result) goto stepper;
					
		    // get here 0.8% of the time for 32, 1.6% for 64
		    xs4 ((*L13)[hs][11]^K[33], (*L13)[hs][12]^K[27], (*L13)[hs][13]^K[53],
			 (*L13)[hs][14]^K[ 4], (*L13)[hs][15]^K[12], (*L13)[hs][16]^K[17],
			 (*R14)[hs][25], t1, (*R14)[hs][19], t2, 
			 (*R14)[hs][ 9], t3, (*R14)[hs][ 0], t4 );
		    s4 (L[11]^K[ 5], L[12]^K[24], L[13]^K[25], L[14]^K[33], L[15]^K[41], L[16]^K[46],
			R[25], R[19], R[ 9], R[ 0] );
		    result &= ~(R[25] ^ t1) & ~(R[19] ^ t2) & ~(R[ 9] ^ t3) & ~(R[ 0] ^ t4);
		    if (!result) goto stepper;
					
		    goto moved_code;
		}
	      stepper:
		++hs;
		if (hs & (1 << 0)) {
		    K[10] = ~K[10];
				
		  changeR2S1: // also for toggling bit 18
		    // update in round 2
		    xs1 (L1[31]^K[54], L1[ 0]^K[18], L1[ 1]^K[33], 
			 L1[ 2]^K[10], L1[ 3]^K[20], L1[ 4]^K[48],
			 PR[ 8], R2[ 8], PR[16], R2[16], PR[22], R2[22], PR[30], R2[30] );
					
		    // and dependent boxes in round 3
		    regen23 = 0x7d;
		    continue;
		}
		if (hs & (1 << 1)) {
		    K[18] = ~K[18];
		    goto changeR2S1;
		}
		if (hs & (1 << 2)) {
		    K[46] = ~K[46];
					
		    // update in round 2
		    xs2 (L1[ 3]^K[34], L1[ 4]^K[13], L1[ 5]^K[ 4], 
			 L1[ 6]^K[55], L1[ 7]^K[46], L1[ 8]^K[26],
			 P[37], R2[12], P[25], R2[27], P[15], R2[ 1], P[11], R2[17] );

		    // and dependent boxes in round 3
		    regen23 = 0xbb;
		    continue;
		}
		if (hs & (1 << 3)) {
		    K[49] = ~K[49];
					
		    // update in round 2
		    xs7 (L1[23]^K[ 9], L1[24]^K[44], L1[25]^K[29], 
			 L1[26]^K[ 7], L1[27]^K[49], L1[28]^K[45],
			 P[57], R2[31], P[29], R2[11], P[43], R2[21], P[55], R2[ 6] );
					
		    // and dependent boxes in round 3
		    regen23 = 0xf5;
		    continue;
		}
		break;
	    }
	    // now step the tail
	    hs = 0;
	    K[49] = ~K[49];
	    ++ts;
	    if (ts & (1 << 0)) {
		K[ 3] = ~K[ 3];
	      changeR1S1R15S3:
		// update in round 1
		xs1 (P[57]^K[47], P[ 7]^K[11], P[15]^K[26], 
		     P[23]^K[ 3], P[31]^K[13], P[39]^K[41],
		     P[ 4], L1[ 8], P[ 2], L1[16], P[50], L1[22], P[48], L1[30] );
				
		// and dependent boxes in round 2
		regen23 = 0x7dff;
				
		for (int i = 0; i < 16; ++i) {
		    // fix box in round 15
		    xs3 ((*R14)[i][ 7]^K[39], (*R14)[i][ 8]^K[11], (*R14)[i][ 9]^K[19],
			 (*R14)[i][10]^K[20], (*R14)[i][11]^K[ 3], (*R14)[i][12]^K[48],
			 CL[23], (*L13)[i][23], CL[15], (*L13)[i][15],
			 CL[29], (*L13)[i][29], CL[ 5], (*L13)[i][ 5] );
		}
		continue;
	    }
	    if (ts & (1 << 1)) {
		K[11] = ~K[11];
		goto changeR1S1R15S3;
	    }
	    if (ts & (1 << 2)) {
		K[42] = ~K[42];
		// update in round 1
		xs7 (P[59]^K[ 2], P[ 1]^K[37], P[ 9]^K[22], 
		     P[17]^K[ 0], P[25]^K[42], P[33]^K[38],
		     P[56], L1[31], P[28], L1[11], P[42], L1[21], P[54], L1[ 6] );
				
		// and dependent boxes in round 2
		regen23 = 0xf5ff;
				
		for (int i = 0; i < 16;) {
		    // fix box in round 15
		    xs8 ((*R14)[i][27]^K[ 2], (*R14)[i][28]^K[29], (*R14)[i][29]^K[30], 
			 (*R14)[i][30]^K[42], (*R14)[i][31]^K[52], (*R14)[i][ 0]^K[14],
			 C[39], (*L13)[i][ 4], C[17], (*L13)[i][26], 
			 C[53], (*L13)[i][14], C[35], (*L13)[i][20] );
					
		    // fix s1 and/or s3 in round 14, if necessary
		    xs1 ((*L13)[i][31]^K[19], (*L13)[i][ 0]^K[40], (*L13)[i][ 1]^K[55], 
			 (*L13)[i][ 2]^K[32], (*L13)[i][ 3]^K[10], (*L13)[i][ 4]^K[13],
			 (*R14)[i][ 8], R12__8[i], (*R14)[i][16], R12_16[i], 
			 (*R14)[i][22], R12_22[i], (*R14)[i][30], R12_30[i] );
					
		    // and step
		    ++i;
		    if (i & 0x01) K[10] = ~K[10];
		}
		continue;
	    }
	    if (ts & (1 << 3)) {
		K[ 5] = ~K[ 5];
		// update in round 1
		xs3 (P[63]^K[53], P[ 5]^K[25], P[13]^K[33], 
		     P[21]^K[34], P[29]^K[17], P[37]^K[ 5],
		     P[58], L1[23], P[60], L1[15], P[40], L1[29], P[46], L1[ 5] );
				
		// and dependent boxes in round 2
		regen23 = 0x5fff;
				
		for (int i = 0; i < 16;) {
		    // fix box in round 15
		    xs2 ((*R14)[i][ 3]^K[13], (*R14)[i][ 4]^K[17], (*R14)[i][ 5]^K[40], 
			 (*R14)[i][ 6]^K[34], (*R14)[i][ 7]^K[25], (*R14)[i][ 8]^K[ 5],
			 C[37], (*L13)[i][12], C[25], (*L13)[i][27], 
			 C[15], (*L13)[i][ 1], C[11], (*L13)[i][17] );
					
		    // fix s1 and/or s3 in round 14, if necessary
		    xs1 ((*L13)[i][31]^K[19], (*L13)[i][ 0]^K[40], (*L13)[i][ 1]^K[55], 
			 (*L13)[i][ 2]^K[32], (*L13)[i][ 3]^K[10], (*L13)[i][ 4]^K[13],
			 (*R14)[i][ 8], R12__8[i], (*R14)[i][16], R12_16[i], 
			 (*R14)[i][22], R12_22[i], (*R14)[i][30], R12_30[i] );
					
		    xs3 ((*L13)[i][ 7]^K[25], (*L13)[i][ 8]^K[54], (*L13)[i][ 9]^K[ 5], 
			 (*L13)[i][10]^K[ 6], (*L13)[i][11]^K[46], (*L13)[i][12]^K[34],
			 (*R14)[i][23], R12_23[i], (*R14)[i][15], R12_15[i], 
			 (*R14)[i][29], R12_29[i], (*R14)[i][ 5], R12__5[i] );
					
		    // and step
		    ++i;
		         if (i & 0x01) K[10] = ~K[10];
		    else if (i & 0x02) K[18] = ~K[18];
		    else if (i & 0x04) K[46] = ~K[46];
		  //else if            K[49] = ~K[49];
		}
		continue;
	    }
	    if (ts & (1 << 4)) {
		K[43] = ~K[43];
		// update in round 1
		xs8 (P[25]^K[16], P[33]^K[43], P[41]^K[44], 
		     P[49]^K[ 1], P[57]^K[ 7], P[ 7]^K[28],
		     P[38], L1[ 4], P[16], L1[26], P[52], L1[14], P[34], L1[20] );
				
		// and dependent boxes in round 2
		regen23 = 0xdeff;
	      changeR15S7:
		for (int i = 0; i < 16;) {
		    // fix box in round 15
		    xs7 ((*R14)[i][23]^K[43], (*R14)[i][24]^K[23], (*R14)[i][25]^K[ 8], 
			 (*R14)[i][26]^K[45], (*R14)[i][27]^K[28], (*R14)[i][28]^K[51],
			 C[57], (*L13)[i][31], C[29], (*L13)[i][11], 
			 C[43], (*L13)[i][21], C[55], (*L13)[i][ 6] );
					
		    // fix s1 and/or s3 in round 14, if necessary
		    xs1 ((*L13)[i][31]^K[19], (*L13)[i][ 0]^K[40], (*L13)[i][ 1]^K[55], 
			 (*L13)[i][ 2]^K[32], (*L13)[i][ 3]^K[10], (*L13)[i][ 4]^K[13],
			 (*R14)[i][ 8], R12__8[i], (*R14)[i][16], R12_16[i], 
			 (*R14)[i][22], R12_22[i], (*R14)[i][30], R12_30[i] );
					
		    xs3 ((*L13)[i][ 7]^K[25], (*L13)[i][ 8]^K[54], (*L13)[i][ 9]^K[ 5], 
			 (*L13)[i][10]^K[ 6], (*L13)[i][11]^K[46], (*L13)[i][12]^K[34],
			 (*R14)[i][23], R12_23[i], (*R14)[i][15], R12_15[i], 
			 (*R14)[i][29], R12_29[i], (*R14)[i][ 5], R12__5[i] );
					
		    // and step
		    ++i;
		         if (i & 0x01) K[10] = ~K[10];
		    else if (i & 0x02) K[18] = ~K[18];
		    else if (i & 0x04) K[46] = ~K[46];
		  //else if            K[49] = ~K[49];
		}
		continue;
	    }
	    if (ts & (1 << 5)) {
		K[ 8] = ~K[ 8];
		// update in round 1
		xs5 (P[61]^K[36], P[ 3]^K[31], P[11]^K[21], 
		     P[19]^K[ 8], P[27]^K[23], P[35]^K[52],
		     P[62], L1[ 7], P[44], L1[13], P[ 0], L1[24], P[22], L1[ 2] );
				
		// and dependent boxes in round 2
		regen23 = 0xf7ff;
				
		goto changeR15S7;
	    }			
	    break;
	}
#if !(defined(BITSLICER_WITH_LESS_BITS) && defined(BIT_64))
	ts = 0;
	K[ 8] = ~K[ 8];
	++hs2;
		
	if (hs2 & (1 << 0)) {
	    K[12] = ~K[12];
	    continue;
	}
	if (hs2 & (1 << 1)) {
	    K[15] = ~K[15];
	    continue;
	}
	if (hs2 & (1 << 2)) {
	    K[45] = ~K[45];
	    continue;
	}
	if (hs2 & (1 << 3)) {
	    K[50] = ~K[50];
	    continue;
	}
#endif
	break;
    }
    asm ("emms\n");
    free (work);
    return 0;


  moved_code:
    // executes just over 1 time in 2048 for 32-bit, in 1024 for 64
    xs5 ((*L13)[hs][15]^K[ 8], (*L13)[hs][16]^K[30], (*L13)[hs][17]^K[52],
	 (*L13)[hs][18]^K[35], (*L13)[hs][19]^K[50], (*L13)[hs][20]^K[51],
	 (*R14)[hs][ 7], t1, (*R14)[hs][13], t2, 
	 (*R14)[hs][24], t3, (*R14)[hs][ 2], t4 );
    s5 (L[15]^K[35], L[16]^K[ 2], L[17]^K[51], 
	L[18]^K[ 7], L[19]^K[22], L[20]^K[23],
	R[ 7], R[13], R[24], R[ 2] );
    result &= ~(R[ 7] ^ t1) & ~(R[13] ^ t2) & ~(R[24] ^ t3) & ~(R[ 2] ^ t4);
    if (!result) goto stepper;
					
    // executes just over 1 time in 32768 for 32-bit, in 16384 for 64
    xs6 ((*L13)[hs][19]^K[45], (*L13)[hs][20]^K[ 1], (*L13)[hs][21]^K[23],
	 (*L13)[hs][22]^K[36], (*L13)[hs][23]^K[ 7], (*L13)[hs][24]^K[ 2],
	 (*R14)[hs][ 3], t1, (*R14)[hs][28], t2, 
	 (*R14)[hs][10], t3, (*R14)[hs][18], t4 );
    s6 (L[19]^K[44], L[20]^K[28], L[21]^K[50], 
	L[22]^K[ 8], L[23]^K[38], L[24]^K[29],
	R[ 3], R[28], R[10], R[18] );
    result &= ~(R[ 3] ^ t1) & ~(R[28] ^ t2) & ~(R[10] ^ t3) & ~(R[18] ^ t4);
    if (!result) goto stepper;
					
    // 1 in 2^19 (32) or 2^18 (64)
    xs7 ((*L13)[hs][23]^K[29], (*L13)[hs][24]^K[ 9], (*L13)[hs][25]^K[49],
	 (*L13)[hs][26]^K[31], (*L13)[hs][27]^K[14], (*L13)[hs][28]^K[37],
	 (*R14)[hs][31], t1, (*R14)[hs][11], t2, 
	 (*R14)[hs][21], t3, (*R14)[hs][ 6], t4 );
    s7 (L[23]^K[ 1], L[24]^K[36], L[25]^K[21], 
	L[26]^K[30], L[27]^K[45], L[28]^K[ 9],
	R[31], R[11], R[21], R[ 6] );
    result &= ~(R[31] ^ t1) & ~(R[11] ^ t2) & ~(R[21] ^ t3) & ~(R[ 6] ^ t4);
    if (!result) goto stepper;
					
    // 1 in 2^23 (32) or 2^22 (64)
    xs8 ((*L13)[hs][27]^K[43], (*L13)[hs][28]^K[15], (*L13)[hs][29]^K[16],
	 (*L13)[hs][30]^K[28], (*L13)[hs][31]^K[38], (*L13)[hs][ 0]^K[ 0],
	 (*R14)[hs][ 4], t1, (*R14)[hs][26], t2, 
	 (*R14)[hs][14], t3, (*R14)[hs][20], t4 );
    s8 (L[27]^K[15], L[28]^K[42], L[29]^K[43], 
	L[30]^K[ 0], L[31]^K[37], L[ 0]^K[31],
	R[ 4], R[26], R[14], R[20] );
    result &= ~(R[ 4] ^ t1) & ~(R[26] ^ t2) & ~(R[14] ^ t3) & ~(R[20] ^ t4);
    if (!result) goto stepper;
					
    // WHEW! At least one of the crypts matches in its entire output of round 12.
    // Only 1 key in 4 billion makes it here!
					
    // Now we perform round 13 and check. This is somewhat simpler, because we
    // already know what outputs we're looking for because we had to have that
    // on hand to compute the target outputs for round 12.
					
    s1 (R[31]^K[ 5], R[ 0]^K[26], R[ 1]^K[41], R[ 2]^K[18], R[ 3]^K[53], R[ 4]^K[24],  
	L[ 8], L[16], L[22], L[30] );
    result &= ~(L[ 8] ^ (*L13)[hs][ 8]) & ~(L[16] ^ (*L13)[hs][16]) &
              ~(L[22] ^ (*L13)[hs][22]) & ~(L[30] ^ (*L13)[hs][30]);
    if (!result) goto stepper;
					
    s2 (R[ 3]^K[10], R[ 4]^K[46], R[ 5]^K[12], R[ 6]^K[ 6], R[ 7]^K[54], R[ 8]^K[34],    
	L[12], L[27], L[ 1], L[17] );
    result &= ~(L[12] ^ (*L13)[hs][12]) & ~(L[27] ^ (*L13)[hs][27]) & 
	      ~(L[ 1] ^ (*L13)[hs][ 1]) & ~(L[17] ^ (*L13)[hs][17]);
    if (!result) goto stepper;
					
    s3 (R[ 7]^K[11], R[ 8]^K[40], R[ 9]^K[48], R[10]^K[17], R[11]^K[32], R[12]^K[20],    
	L[23], L[15], L[29], L[ 5] );
    result &= ~(L[23] ^ (*L13)[hs][23]) & ~(L[15] ^ (*L13)[hs][15]) & 
	      ~(L[29] ^ (*L13)[hs][29]) & ~(L[ 5] ^ (*L13)[hs][ 5]);
    if (!result) goto stepper;
					
    s4 (R[11]^K[19], R[12]^K[13], R[13]^K[39], R[14]^K[47], R[15]^K[55], R[16]^K[ 3],    
	L[25], L[19], L[ 9], L[ 0] );
    result &= ~(L[25] ^ (*L13)[hs][25]) & ~(L[19] ^ (*L13)[hs][19]) & 
	      ~(L[ 9] ^ (*L13)[hs][ 9]) & ~(L[ 0] ^ (*L13)[hs][ 0]);
    if (!result) goto stepper;
					
    s5 (R[15]^K[49], R[16]^K[16], R[17]^K[38], R[18]^K[21], R[19]^K[36], R[20]^K[37],    
	L[ 7], L[13], L[24], L[ 2] );
    result &= ~(L[ 7] ^ (*L13)[hs][ 7]) & ~(L[13] ^ (*L13)[hs][13]) & 
	      ~(L[24] ^ (*L13)[hs][24]) & ~(L[ 2] ^ (*L13)[hs][ 2]);
    if (!result) goto stepper;
					
    s6 (R[19]^K[31], R[20]^K[42], R[21]^K[ 9], R[22]^K[22], R[23]^K[52], R[24]^K[43],    
	L[ 3], L[28], L[10], L[18] );
    result &= ~(L[ 3] ^ (*L13)[hs][ 3]) & ~(L[28] ^ (*L13)[hs][28]) & 
	      ~(L[10] ^ (*L13)[hs][10]) & ~(L[18] ^ (*L13)[hs][18]);
    if (!result) goto stepper;

    s7 (R[23]^K[15], R[24]^K[50], R[25]^K[35], R[26]^K[44], R[27]^K[ 0], R[28]^K[23],    
	L[31], L[11], L[21], L[ 6] );
    result &= ~(L[31] ^ (*L13)[hs][31]) & ~(L[11] ^ (*L13)[hs][11]) & 
	      ~(L[21] ^ (*L13)[hs][21]) & ~(L[ 6] ^ (*L13)[hs][ 6]);
    if (!result) goto stepper;
					
    s8 (R[27]^K[29], R[28]^K[ 1], R[29]^K[ 2], R[30]^K[14], R[31]^K[51], R[ 0]^K[45],    
	L[ 4], L[26], L[14], L[20] );
    result &= ~(L[ 4] ^ (*L13)[hs][ 4]) & ~(L[26] ^ (*L13)[hs][26]) & 
	      ~(L[14] ^ (*L13)[hs][14]) & ~(L[20] ^ (*L13)[hs][20]);
    if (!result) goto stepper;
					
    // whoomp, there it is!
    asm ("emms\n");
    free (work);
    return result;
}
