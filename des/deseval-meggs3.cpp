//
// $Log: deseval-meggs3.cpp,v $
// Revision 1.16  1999/11/27 06:57:34  sampo
// Mac command-line client checkins round 1...all changes commented with /* Mindmorph */ until everything settles out.
//
// Revision 1.15  1999/11/17 21:51:52  sampo
// remove Dakidd's cvs merge error in previous revision in regards to ambiguous
// type 'slice'
//
// Revision 1.14  1999/10/06 19:37:43  dakidd
// All occurrences of "slice" changed to "SliceType" to resolve "ambiguous class reference - found slice/std::slice" error under CW pro 5. (Apparently, CWP5 has a "std::slice" class)
//
// Revision 1.13  1999/01/09 08:57:41  remi
// Fixed the previous fix : it's only for alpha/nt + defined(bit_64) + msvc++
// Removed the rcv copyright as we don't use rcv' sboxes anymore.
//
// Revision 1.12  1999/01/08 02:53:37  michmarc
// Added __int64 type support for Alpha/NT (whose "unsigned long" type
// is only 32 bits)
//
// Revision 1.11  1999/01/06 06:54:24  dicamillo
// Change MacOS yield routine.
//
// Revision 1.10  1998/12/14 06:08:26  pct
// Sorry.  Uploaded the wrong file on previous CVS.  Don't use previous
// version.
//
// Revision 1.8  1998/12/14 01:56:43  dicamillo
// MacOS: allow use of extern "C" for whack16.
//
// Revision 1.7  1998/12/13 21:53:33  dicamillo
// Mac OS change for compilation and Mac client scheduling.
//
// Revision 1.6  1998/07/14 10:43:40  remi
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
// Revision 1.5  1998/07/08 23:42:12  remi
// Added support for CliIdentifyModules().
//
// Revision 1.4  1998/06/15 09:21:53  jlawson
// eliminated unused label warning
//
// Revision 1.3  1998/06/14 08:27:07  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.2  1998/06/14 08:13:21  friedbait
// 'Log' keywords added to maintain automatic change history
//
//

#if (!defined(lint) && defined(__showids__))
const char *deseval_meggs3_cpp(void) {
return "@(#)$Id: deseval-meggs3.cpp,v 1.16 1999/11/27 06:57:34 sampo Exp $"; }
#endif

#include <cputypes.h>		/* Isn't this needed for using CLIENT_OS defines? */

#if (CLIENT_OS != OS_MACOS)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

/* Bitslice driver copyright (C) 1998 Andrew Meggs / casa de la cabeza explosiva */
/* All rights reserved. A non-transferrable, royalty free license to this code   */
/* is granted to distributed.net for use exclusively in the DES Challenge II,    */
/* but ownership remains with the author.                                        */

#if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32) && defined(BIT_64) && (_MSC_VER >= 11)
// int and long are 32bits under Alpha/NT for compatability with x86/NT
#define WORD_TYPE unsigned __int64
#else
#define WORD_TYPE unsigned long
#endif

#include "kwan-sboxes.h"

#if (CLIENT_OS == OS_MACOS)
#define TICKS ((unsigned long *)0x16a)
#define slice unsigned long
//extern void DES_YieldToMain(void);
extern void mac_yield(void);
unsigned long DES_ticks_to_use = 6; /* hardcode 100ms for now */
unsigned long DES_yield_ticks = 0;
#endif

static void
f_s1 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_0_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_0_BIT_0( out0, out0 )
	SBOX_0_BIT_1( out1, out1 )
	SBOX_0_BIT_2( out2, out2 )
	SBOX_0_BIT_3( out3, out3 )
}

static void
f_s2 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_1_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_1_BIT_0( out0, out0 )
	SBOX_1_BIT_1( out1, out1 )
	SBOX_1_BIT_2( out2, out2 )
	SBOX_1_BIT_3( out3, out3 )
}

static void
f_s3 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_2_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_2_BIT_0( out0, out0 )
	SBOX_2_BIT_1( out1, out1 )
	SBOX_2_BIT_2( out2, out2 )
	SBOX_2_BIT_3( out3, out3 )
}

static void
f_s4 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_3_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_3_BIT_0( out0, out0 )
	SBOX_3_BIT_1( out1, out1 )
	SBOX_3_BIT_2( out2, out2 )
	SBOX_3_BIT_3( out3, out3 )
}

static void
f_s5 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_4_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_4_BIT_0( out0, out0 )
	SBOX_4_BIT_1( out1, out1 )
	SBOX_4_BIT_2( out2, out2 )
	SBOX_4_BIT_3( out3, out3 )
}

static void
f_s6 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_5_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_5_BIT_0( out0, out0 )
	SBOX_5_BIT_1( out1, out1 )
	SBOX_5_BIT_2( out2, out2 )
	SBOX_5_BIT_3( out3, out3 )
}

static void
f_s7 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_6_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_6_BIT_0( out0, out0 )
	SBOX_6_BIT_1( out1, out1 )
	SBOX_6_BIT_2( out2, out2 )
	SBOX_6_BIT_3( out3, out3 )
}

static void
f_s8 (
 WORD_TYPE	in0,
 WORD_TYPE	in1,
 WORD_TYPE	in2,
 WORD_TYPE	in3,
 WORD_TYPE	in4,
 WORD_TYPE	in5,
 WORD_TYPE	&out0,
 WORD_TYPE	&out1,
 WORD_TYPE	&out2,
 WORD_TYPE	&out3
) {
	SBOX_7_INIT( in0, in1, in2, in3, in4, in5 )
	SBOX_7_BIT_0( out0, out0 )
	SBOX_7_BIT_1( out1, out1 )
	SBOX_7_BIT_2( out2, out2 )
	SBOX_7_BIT_3( out3, out3 )
}


static const unsigned char keybits[ 13 /*rounds*/ * 8 /*boxes*/ * 6 /*bits*/ ] = {
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


#if INLINEPARTIAL
inline
#else
static
#endif
void partialround( WORD_TYPE S[32], WORD_TYPE M[32], WORD_TYPE D[32], WORD_TYPE K[56], int ks, int select )
{
	const unsigned char *kb = keybits + ks;
	
	if (select & 0x80) {
			SBOX_0_INIT( S[31]^K[kb[ 0]], S[ 0]^K[kb[ 1]], S[ 1]^K[kb[ 2]], S[ 2]^K[kb[ 3]], S[ 3]^K[kb[ 4]], S[ 4]^K[kb[ 5]] )
			SBOX_0_BIT_0( M[ 8], D[ 8] )
			SBOX_0_BIT_1( M[16], D[16] )
			SBOX_0_BIT_2( M[22], D[22] )
			SBOX_0_BIT_3( M[30], D[30] )
		}
	if (select & 0x40) {
			SBOX_1_INIT( S[ 3]^K[kb[ 6]], S[ 4]^K[kb[ 7]], S[ 5]^K[kb[ 8]], S[ 6]^K[kb[ 9]], S[ 7]^K[kb[10]], S[ 8]^K[kb[11]] )
			SBOX_1_BIT_0( M[12], D[12] )
			SBOX_1_BIT_1( M[27], D[27] )
			SBOX_1_BIT_2( M[ 1], D[ 1] )
			SBOX_1_BIT_3( M[17], D[17] )
		}
	if (select & 0x20) {
			SBOX_2_INIT( S[ 7]^K[kb[12]], S[ 8]^K[kb[13]], S[ 9]^K[kb[14]], S[10]^K[kb[15]], S[11]^K[kb[16]], S[12]^K[kb[17]] )
			SBOX_2_BIT_0( M[23], D[23] )
			SBOX_2_BIT_1( M[15], D[15] )
			SBOX_2_BIT_2( M[29], D[29] )
			SBOX_2_BIT_3( M[ 5], D[ 5] )
		}
	/*if (select & 0x10)*/ {
			SBOX_3_INIT( S[11]^K[kb[18]], S[12]^K[kb[19]], S[13]^K[kb[20]], S[14]^K[kb[21]], S[15]^K[kb[22]], S[16]^K[kb[23]] )
			SBOX_3_BIT_0( M[25], D[25] )
			SBOX_3_BIT_1( M[19], D[19] )
			SBOX_3_BIT_2( M[ 9], D[ 9] )
			SBOX_3_BIT_3( M[ 0], D[ 0] )
		}
	if (select & 0x08) {
			SBOX_4_INIT( S[15]^K[kb[24]], S[16]^K[kb[25]], S[17]^K[kb[26]], S[18]^K[kb[27]], S[19]^K[kb[28]], S[20]^K[kb[29]] )
			SBOX_4_BIT_0( M[ 7], D[ 7] )
			SBOX_4_BIT_1( M[13], D[13] )
			SBOX_4_BIT_2( M[24], D[24] )
			SBOX_4_BIT_3( M[ 2], D[ 2] )
		}
	if (select & 0x04) {
			SBOX_5_INIT( S[19]^K[kb[30]], S[20]^K[kb[31]], S[21]^K[kb[32]], S[22]^K[kb[33]], S[23]^K[kb[34]], S[24]^K[kb[35]] )
			SBOX_5_BIT_0( M[ 3], D[ 3] )
			SBOX_5_BIT_1( M[28], D[28] )
			SBOX_5_BIT_2( M[10], D[10] )
			SBOX_5_BIT_3( M[18], D[18] )
		}
	if (select & 0x02) {
			SBOX_6_INIT( S[23]^K[kb[36]], S[24]^K[kb[37]], S[25]^K[kb[38]], S[26]^K[kb[39]], S[27]^K[kb[40]], S[28]^K[kb[41]] )
			SBOX_6_BIT_0( M[31], D[31] )
			SBOX_6_BIT_1( M[11], D[11] )
			SBOX_6_BIT_2( M[21], D[21] )
			SBOX_6_BIT_3( M[ 6], D[ 6] )
		}
	if (select & 0x01) {
			SBOX_7_INIT( S[27]^K[kb[42]], S[28]^K[kb[43]], S[29]^K[kb[44]], S[30]^K[kb[45]], S[31]^K[kb[46]], S[ 0]^K[kb[47]] )
			SBOX_7_BIT_0( M[ 4], D[ 4] )
			SBOX_7_BIT_1( M[26], D[26] )
			SBOX_7_BIT_2( M[14], D[14] )
			SBOX_7_BIT_3( M[20], D[20] )
		}
}

#if INLINEMULTI
inline
#else
static
#endif
void multiround( WORD_TYPE S[32], WORD_TYPE N[32], WORD_TYPE M[32], WORD_TYPE D[32], WORD_TYPE K[56] )
{
	const unsigned char *kb = keybits + 144;
	
	for (int i = 4; i <= 10; ++i) {
		{
			SBOX_0_INIT( S[31]^K[kb[ 0]], S[ 0]^K[kb[ 1]], S[ 1]^K[kb[ 2]], S[ 2]^K[kb[ 3]], S[ 3]^K[kb[ 4]], S[ 4]^K[kb[ 5]] )
			SBOX_0_BIT_0( M[ 8], D[ 8] )
			SBOX_0_BIT_1( M[16], D[16] )
			SBOX_0_BIT_2( M[22], D[22] )
			SBOX_0_BIT_3( M[30], D[30] )
		}
		{
			SBOX_1_INIT( S[ 3]^K[kb[ 6]], S[ 4]^K[kb[ 7]], S[ 5]^K[kb[ 8]], S[ 6]^K[kb[ 9]], S[ 7]^K[kb[10]], S[ 8]^K[kb[11]] )
			SBOX_1_BIT_0( M[12], D[12] )
			SBOX_1_BIT_1( M[27], D[27] )
			SBOX_1_BIT_2( M[ 1], D[ 1] )
			SBOX_1_BIT_3( M[17], D[17] )
		}
		{
			SBOX_2_INIT( S[ 7]^K[kb[12]], S[ 8]^K[kb[13]], S[ 9]^K[kb[14]], S[10]^K[kb[15]], S[11]^K[kb[16]], S[12]^K[kb[17]] )
			SBOX_2_BIT_0( M[23], D[23] )
			SBOX_2_BIT_1( M[15], D[15] )
			SBOX_2_BIT_2( M[29], D[29] )
			SBOX_2_BIT_3( M[ 5], D[ 5] )
		}
		{
			SBOX_3_INIT( S[11]^K[kb[18]], S[12]^K[kb[19]], S[13]^K[kb[20]], S[14]^K[kb[21]], S[15]^K[kb[22]], S[16]^K[kb[23]] )
			SBOX_3_BIT_0( M[25], D[25] )
			SBOX_3_BIT_1( M[19], D[19] )
			SBOX_3_BIT_2( M[ 9], D[ 9] )
			SBOX_3_BIT_3( M[ 0], D[ 0] )
		}
		{
			SBOX_4_INIT( S[15]^K[kb[24]], S[16]^K[kb[25]], S[17]^K[kb[26]], S[18]^K[kb[27]], S[19]^K[kb[28]], S[20]^K[kb[29]] )
			SBOX_4_BIT_0( M[ 7], D[ 7] )
			SBOX_4_BIT_1( M[13], D[13] )
			SBOX_4_BIT_2( M[24], D[24] )
			SBOX_4_BIT_3( M[ 2], D[ 2] )
		}
		{
			SBOX_5_INIT( S[19]^K[kb[30]], S[20]^K[kb[31]], S[21]^K[kb[32]], S[22]^K[kb[33]], S[23]^K[kb[34]], S[24]^K[kb[35]] )
			SBOX_5_BIT_0( M[ 3], D[ 3] )
			SBOX_5_BIT_1( M[28], D[28] )
			SBOX_5_BIT_2( M[10], D[10] )
			SBOX_5_BIT_3( M[18], D[18] )
		}
		{
			SBOX_6_INIT( S[23]^K[kb[36]], S[24]^K[kb[37]], S[25]^K[kb[38]], S[26]^K[kb[39]], S[27]^K[kb[40]], S[28]^K[kb[41]] )
			SBOX_6_BIT_0( M[31], D[31] )
			SBOX_6_BIT_1( M[11], D[11] )
			SBOX_6_BIT_2( M[21], D[21] )
			SBOX_6_BIT_3( M[ 6], D[ 6] )
		}
		{
			SBOX_7_INIT( S[27]^K[kb[42]], S[28]^K[kb[43]], S[29]^K[kb[44]], S[30]^K[kb[45]], S[31]^K[kb[46]], S[ 0]^K[kb[47]] )
			SBOX_7_BIT_0( M[ 4], D[ 4] )
			SBOX_7_BIT_1( M[26], D[26] )
			SBOX_7_BIT_2( M[14], D[14] )
			SBOX_7_BIT_3( M[20], D[20] )
		}
		kb += 48;
		WORD_TYPE *swap = D;
		D = N;
		M = S;
		S = N = swap;
	}
}



typedef WORD_TYPE SliceType; // dakidd 05-OCT-1999 - original "slice" was choking CW Pro 5 with "ambiguious class reference" errors
#if ((CLIENT_OS == OS_MACOS) && defined(MRCPP_FOR_DES))
extern "C" SliceType whack16(SliceType *P, SliceType *C, SliceType *K);
#else
extern SliceType whack16(SliceType *P, SliceType *C, SliceType *K);
#endif

#define DEBUGPRINT 0

// test all combinations of the easily-toggled bits
SliceType whack16(SliceType *P, SliceType *C, SliceType *K)
{
	SliceType R14[16][32];
	SliceType L13[16][32];

	SliceType R12_23[16], R12_15[16], R12_29[16], R12__5[16];
	SliceType R12__8[16], R12_16[16], R12_22[16], R12_30[16];

	SliceType L1[32];
	SliceType R2[32];
	SliceType L3[32];

	SliceType R[32];
	SliceType L[32];

	SliceType PL[32], PR[32], CL[32], CR[32];
	PL[0] = P[6];
	PL[1] = P[14];
	PL[2] = P[22];
	PL[3] = P[30];
	PL[4] = P[38];
	PL[5] = P[46];
	PL[6] = P[54];
	PL[7] = P[62];
	PL[8] = P[4];
	PL[9] = P[12];
	PL[10] = P[20];
	PL[11] = P[28];
	PL[12] = P[36];
	PL[13] = P[44];
	PL[14] = P[52];
	PL[15] = P[60];
	PL[16] = P[2];
	PL[17] = P[10];
	PL[18] = P[18];
	PL[19] = P[26];
	PL[20] = P[34];
	PL[21] = P[42];
	PL[22] = P[50];
	PL[23] = P[58];
	PL[24] = P[0];
	PL[25] = P[8];
	PL[26] = P[16];
	PL[27] = P[24];
	PL[28] = P[32];
	PL[29] = P[40];
	PL[30] = P[48];
	PL[31] = P[56];
	
	PR[0] = P[7];
	PR[1] = P[15];
	PR[2] = P[23];
	PR[3] = P[31];
	PR[4] = P[39];
	PR[5] = P[47];
	PR[6] = P[55];
	PR[7] = P[63];
	PR[8] = P[5];
	PR[9] = P[13];
	PR[10] = P[21];
	PR[11] = P[29];
	PR[12] = P[37];
	PR[13] = P[45];
	PR[14] = P[53];
	PR[15] = P[61];
	PR[16] = P[3];
	PR[17] = P[11];
	PR[18] = P[19];
	PR[19] = P[27];
	PR[20] = P[35];
	PR[21] = P[43];
	PR[22] = P[51];
	PR[23] = P[59];
	PR[24] = P[1];
	PR[25] = P[9];
	PR[26] = P[17];
	PR[27] = P[25];
	PR[28] = P[33];
	PR[29] = P[41];
	PR[30] = P[49];
	PR[31] = P[57];
	
	CL[ 8] = C[5];
	CL[16] = C[3];
	CL[22] = C[51];
	CL[30] = C[49];
	CL[12] = C[37];
	CL[27] = C[25];
	CL[ 1] = C[15];
	CL[17] = C[11];
	CL[23] = C[59];
	CL[15] = C[61];
	CL[29] = C[41];
	CL[ 5] = C[47];
	CL[25] = C[9];
	CL[19] = C[27];
	CL[ 9] = C[13];
	CL[ 0] = C[7];
	CL[ 7] = C[63];
	CL[13] = C[45];
	CL[24] = C[1];
	CL[ 2] = C[23];
	CL[ 3] = C[31];
	CL[28] = C[33];
	CL[10] = C[21];
	CL[18] = C[19];
	CL[31] = C[57];
	CL[11] = C[29];
	CL[21] = C[43];
	CL[ 6] = C[55];
	CL[ 4] = C[39];
	CL[26] = C[17];
	CL[14] = C[53];
	CL[20] = C[35];
	
	CR[ 8] = C[4];
	CR[16] = C[2];
	CR[22] = C[50];
	CR[30] = C[48];
	CR[12] = C[36];
	CR[27] = C[24];
	CR[ 1] = C[14];
	CR[17] = C[10];
	CR[23] = C[58];
	CR[15] = C[60];
	CR[29] = C[40];
	CR[ 5] = C[46];
	CR[25] = C[8];
	CR[19] = C[26];
	CR[ 9] = C[12];
	CR[ 0] = C[6];
	CR[ 7] = C[62];
	CR[13] = C[44];
	CR[24] = C[0];
	CR[ 2] = C[22];
	CR[ 3] = C[30];
	CR[28] = C[32];
	CR[10] = C[20];
	CR[18] = C[18];
	CR[31] = C[56];
	CR[11] = C[28];
	CR[21] = C[42];
	CR[ 6] = C[54];
	CR[ 4] = C[38];
	CR[26] = C[16];
	CR[14] = C[52];
	CR[20] = C[34];
	
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
				
				partialround( CL, CR, R14[i], K, 576, 0xff );
				
				
				// inv-round 15
				
				partialround( R14[i], CL, L13[i], K, 528, 0xff );
				
				// inv-round 14
				
				R12__8[i] = R14[i][ 8];
				R12_16[i] = R14[i][16];
				R12_22[i] = R14[i][22];
				R12_30[i] = R14[i][30];
				f_s1( L13[i][31]^K[19], L13[i][ 0]^K[40], L13[i][ 1]^K[55], L13[i][ 2]^K[32], L13[i][ 3]^K[10], L13[i][ 4]^K[13],
					R12__8[i], R12_16[i], R12_22[i], R12_30[i] );
				
				R12_23[i] = R14[i][23];
				R12_15[i] = R14[i][15];
				R12_29[i] = R14[i][29];
				R12__5[i] = R14[i][ 5];
				f_s3( L13[i][ 7]^K[25], L13[i][ 8]^K[54], L13[i][ 9]^K[ 5], L13[i][10]^K[ 6], L13[i][11]^K[46], L13[i][12]^K[34],
					R12_23[i], R12_15[i], R12_29[i], R12__5[i] );
				
				++i;
				     if (i & 0x01) K[10] = ~K[10];
				else if (i & 0x02) K[18] = ~K[18];
				else if (i & 0x04) K[46] = ~K[46];
				else               K[49] = ~K[49];
			}
			
			// now initialize the head:
			
			// round 1
			
			partialround( PR, PL, L1, K, 0, 0xff );
			
			// round 2
			 // regen23 |= 0xff00
			
			// round 3
			 // regen23 |= 0x00ff
		}
		int regen23 = 0xffff;
		for (;;) {
			//check_16();
			partialround( L1, PR, R2, K, 48, (regen23 >> 8) | 0x02 );
	/*		if (regen23 & 0x8000) {
				R2[ 8] = P[ 5];
				R2[16] = P[ 3];
				R2[22] = P[51];
				R2[30] = P[49];
				f_s1( L1[31]^K[54], L1[ 0]^K[18], L1[ 1]^K[33], L1[ 2]^K[10], L1[ 3]^K[20], L1[ 4]^K[48],
					R2[ 8], R2[16], R2[22], R2[30] );
			}
			if (regen23 & 0x4000) {
				R2[12] = P[37];
				R2[27] = P[25];
				R2[ 1] = P[15];
				R2[17] = P[11];
				f_s2( L1[ 3]^K[34], L1[ 4]^K[13], L1[ 5]^K[ 4], L1[ 6]^K[55], L1[ 7]^K[46], L1[ 8]^K[26],
					R2[12], R2[27], R2[ 1], R2[17] );
			}
			if (regen23 & 0x2000) {
				R2[23] = P[59];
				R2[15] = P[61];
				R2[29] = P[41];
				R2[ 5] = P[47];
				f_s3( L1[ 7]^K[ 3], L1[ 8]^K[32], L1[ 9]^K[40], L1[10]^K[41], L1[11]^K[24], L1[12]^K[12],
					R2[23], R2[15], R2[29], R2[ 5] );
			}
		//	if (regen23 & 0x1000) { // box 4 is *always* regenerated
				R2[25] = P[ 9];
				R2[19] = P[27];
				R2[ 9] = P[13];
				R2[ 0] = P[ 7];
				f_s4( L1[11]^K[11], L1[12]^K[ 5], L1[13]^K[ 6], L1[14]^K[39], L1[15]^K[47], L1[16]^K[27],
					R2[25], R2[19], R2[ 9], R2[ 0] );
		//	}
			if (regen23 & 0x0800) {
				R2[ 7] = P[63];
				R2[13] = P[45];
				R2[24] = P[ 1];
				R2[ 2] = P[23];
				f_s5( L1[15]^K[43], L1[16]^K[38], L1[17]^K[28], L1[18]^K[15], L1[19]^K[30], L1[20]^K[ 0],
					R2[ 7], R2[13], R2[24], R2[ 2] );
			}
			if (regen23 & 0x0400) {
				R2[ 3] = P[31];
				R2[28] = P[33];
				R2[10] = P[21];
				R2[18] = P[19];
				f_s6( L1[19]^K[21], L1[20]^K[36], L1[21]^K[31], L1[22]^K[16], L1[23]^K[42], L1[24]^K[37],
					R2[ 3], R2[28], R2[10], R2[18] );
			}
		//	if (regen23 & 0x0200) { // always, due to resetting of K[49]
				R2[31] = P[57];
				R2[11] = P[29];
				R2[21] = P[43];
				R2[ 6] = P[55];
				f_s7( L1[23]^K[ 9], L1[24]^K[44], L1[25]^K[29], L1[26]^K[ 7], L1[27]^K[49], L1[28]^K[45],
					R2[31], R2[11], R2[21], R2[ 6] );
		//	}
			if (regen23 & 0x0100) { // -- // always, due to resetting of K[50]
				R2[ 4] = P[39];
				R2[26] = P[17];
				R2[14] = P[53];
				R2[20] = P[35];
				f_s8( L1[27]^K[23], L1[28]^K[50], L1[29]^K[51], L1[30]^K[ 8], L1[31]^K[14], L1[ 0]^K[35],
					R2[ 4], R2[26], R2[14], R2[20] );
			} */
			for (;;) {
				// You'll undoubtedly be tempted to inline at least the call to multiround(). However,
				// the resulting explosion of temporary variables produced a significant slowdown
				// in the code from several compilers (Metrowerks and MrCpp) that had trouble doing
				// optimal register allocation, so by default these are out of line. To try inlining
				// them, define INLINEPARTIAL and/or INLINEMULTI when compiling.
				partialround( R2, L1, L3, K, 96, regen23 );
				multiround( L3, L, R2, R, K );
				partialround( R, L, L, K, 480, 0xDE );
				{ // now we start checking the outputs...
				// round 12
					SliceType save = R[29];
					SliceType result;
					{
						SBOX_2_INIT( L[ 7]^K[54], L[ 8]^K[26], L[ 9]^K[34], L[10]^K[ 3], L[11]^K[18], L[12]^K[ 6] );
						SliceType out;
						SBOX_2_BIT_0( R[23], out ); R[23] = out;
						result  = ~(out ^ R12_23[hs]);
						SBOX_2_BIT_1( R[15], out ); R[15] = out;
						result &= ~(out ^ R12_15[hs]);
						SBOX_2_BIT_2( R[29], out ); R[29] = out;
						result &= ~(out ^ R12_29[hs]);
						SBOX_2_BIT_3( R[ 5], out ); R[ 5] = out;
						result &= ~(out ^ R12__5[hs]);
						if (!result) goto stepper;
					}
					// f_s3( L[ 7]^K[54], L[ 8]^K[26], L[ 9]^K[34], L[10]^K[ 3], L[11]^K[18], L[12]^K[ 6],    R[23], R[15], R[29], R[ 5] );
					// result  = ~(R[23] ^ R12_23[hs]) & ~(R[15] ^ R12_15[hs]) & ~(R[29] ^ R12_29[hs]) & ~(R[ 5] ^ R12__5[hs]);
					// if (!result) goto stepper;
					
					// get here 87.3% of the time for 32 bits, 98.4% for 64
					// from round 11:
						{
							SBOX_7_INIT( R[27]^K[ 1], R[28]^K[28], save^K[29], R[30]^K[45], R[31]^K[23], R[ 0]^K[44] );
							SBOX_7_BIT_0( L[ 4], L[ 4] );
							SBOX_7_BIT_1( L[26], L[26] );
							SBOX_7_BIT_2( L[14], L[14] );
							SBOX_7_BIT_3( L[20], L[20] );
						}
						//f_s8( R[27]^K[ 1], R[28]^K[28], save^K[29], R[30]^K[45], R[31]^K[23], R[ 0]^K[44],    L[ 4], L[26], L[14], L[20] );
					save = R[ 8];
					{
						SBOX_0_INIT( L[31]^K[48], L[ 0]^K[12], L[ 1]^K[27], L[ 2]^K[ 4], L[ 3]^K[39], L[ 4]^K[10] );
						SliceType out;
						SBOX_0_BIT_0( R[ 8], out ); R[ 8] = out;
						result &= ~(out ^ R12__8[hs]);
						SBOX_0_BIT_1( R[16], out ); R[16] = out;
						result &= ~(out ^ R12_16[hs]);
						SBOX_0_BIT_2( R[22], out ); R[22] = out;
						result &= ~(out ^ R12_22[hs]);
						SBOX_0_BIT_3( R[30], out ); R[30] = out;
						result &= ~(out ^ R12_30[hs]);
						if (!result) goto stepper;
					}
					//f_s1( L[31]^K[48], L[ 0]^K[12], L[ 1]^K[27], L[ 2]^K[ 4], L[ 3]^K[39], L[ 4]^K[10],    R[ 8], R[16], R[22], R[30] );
					//result &= ~(R[ 8] ^ R12__8[hs]) & ~(R[16] ^ R12_16[hs]) & ~(R[22] ^ R12_22[hs]) & ~(R[30] ^ R12_30[hs]);
					//if (!result) goto stepper;
					
					// get here 11.8% of the time for 32, 22.2% for 64
					// last of round 11:
					f_s3( R[ 7]^K[40], save^K[12], R[ 9]^K[20], R[10]^K[46], R[11]^K[ 4], R[12]^K[17],    L[23], L[15], L[29], L[ 5] );
					
					// no more cleverness, just finish inv-14 and 12 piece by piece and compare as we go
					SliceType t1, t2, t3, t4;
					
					t1 = R14[hs][12];
					t2 = R14[hs][27];
					t3 = R14[hs][ 1];
					t4 = R14[hs][17];
					f_s2( L13[hs][ 3] ^ K[24], L13[hs][ 4] ^ K[ 3], L13[hs][ 5] ^ K[26],
						L13[hs][ 6] ^ K[20], L13[hs][ 7] ^ K[11], L13[hs][ 8] ^ K[48],
						t1, t2, t3, t4 );
					f_s2( L[ 3]^K[53], L[ 4]^K[32], L[ 5]^K[55], L[ 6]^K[17], L[ 7]^K[40], L[ 8]^K[20],
						R[12], R[27], R[ 1], R[17] );
					result &= ~(R[12] ^ t1) & ~(R[27] ^ t2) & ~(R[ 1] ^ t3) & ~(R[17] ^ t4);
					if (!result) goto stepper;
					
					 // get here 0.8% of the time for 32, 1.6% for 64
					t1 = R14[hs][25];
					t2 = R14[hs][19];
					t3 = R14[hs][ 9];
					t4 = R14[hs][ 0];
					f_s4( L13[hs][11]^K[33], L13[hs][12]^K[27], L13[hs][13]^K[53],
						L13[hs][14]^K[ 4], L13[hs][15]^K[12], L13[hs][16]^K[17],
						t1, t2, t3, t4 );
					f_s4( L[11]^K[ 5], L[12]^K[24], L[13]^K[25], L[14]^K[33], L[15]^K[41], L[16]^K[46],
						R[25], R[19], R[ 9], R[ 0] );
					result &= ~(R[25] ^ t1) & ~(R[19] ^ t2) & ~(R[ 9] ^ t3) & ~(R[ 0] ^ t4);
					if (!result) goto stepper;
					
					// executes just over 1 time in 2048 for 32-bit, in 1024 for 64
					t1 = R14[hs][ 7];
					t2 = R14[hs][13];
					t3 = R14[hs][24];
					t4 = R14[hs][ 2];
					f_s5( L13[hs][15]^K[ 8], L13[hs][16]^K[30], L13[hs][17]^K[52],
						L13[hs][18]^K[35], L13[hs][19]^K[50], L13[hs][20]^K[51],
						t1, t2, t3, t4 );
					f_s5( L[15]^K[35], L[16]^K[ 2], L[17]^K[51], L[18]^K[ 7], L[19]^K[22], L[20]^K[23],
						R[ 7], R[13], R[24], R[ 2] );
					result &= ~(R[ 7] ^ t1) & ~(R[13] ^ t2) & ~(R[24] ^ t3) & ~(R[ 2] ^ t4);
					if (!result) goto stepper;
					
					// executes just over 1 time in 32768 for 32-bit, in 16384 for 64
					t1 = R14[hs][ 3];
					t2 = R14[hs][28];
					t3 = R14[hs][10];
					t4 = R14[hs][18];
					f_s6( L13[hs][19]^K[45], L13[hs][20]^K[ 1], L13[hs][21]^K[23],
						L13[hs][22]^K[36], L13[hs][23]^K[ 7], L13[hs][24]^K[ 2],
						t1, t2, t3, t4 );
					f_s6( L[19]^K[44], L[20]^K[28], L[21]^K[50], L[22]^K[ 8], L[23]^K[38], L[24]^K[29],
						R[ 3], R[28], R[10], R[18] );
					result &= ~(R[ 3] ^ t1) & ~(R[28] ^ t2) & ~(R[10] ^ t3) & ~(R[18] ^ t4);
					if (!result) goto stepper;
					
					// 1 in 2^19 (32) or 2^18 (64)
					t1 = R14[hs][31];
					t2 = R14[hs][11];
					t3 = R14[hs][21];
					t4 = R14[hs][ 6];
					f_s7( L13[hs][23]^K[29], L13[hs][24]^K[ 9], L13[hs][25]^K[49],
						L13[hs][26]^K[31], L13[hs][27]^K[14], L13[hs][28]^K[37],
						t1, t2, t3, t4 );
					f_s7( L[23]^K[ 1], L[24]^K[36], L[25]^K[21], L[26]^K[30], L[27]^K[45], L[28]^K[ 9],
						R[31], R[11], R[21], R[ 6] );
					result &= ~(R[31] ^ t1) & ~(R[11] ^ t2) & ~(R[21] ^ t3) & ~(R[ 6] ^ t4);
					if (!result) goto stepper;
					
					// 1 in 2^23 (32) or 2^22 (64)
					t1 = R14[hs][ 4];
					t2 = R14[hs][26];
					t3 = R14[hs][14];
					t4 = R14[hs][20];
					f_s8( L13[hs][27]^K[43], L13[hs][28]^K[15], L13[hs][29]^K[16],
						L13[hs][30]^K[28], L13[hs][31]^K[38], L13[hs][ 0]^K[ 0],
						t1, t2, t3, t4 );
					f_s8( L[27]^K[15], L[28]^K[42], L[29]^K[43], L[30]^K[ 0], L[31]^K[37], L[ 0]^K[31],
						R[ 4], R[26], R[14], R[20] );
					result &= ~(R[ 4] ^ t1) & ~(R[26] ^ t2) & ~(R[14] ^ t3) & ~(R[20] ^ t4);
					if (!result) goto stepper;
					
					// WHEW! At least one of the crypts matches in its entire output of round 12.
					// Only 1 key in 4 billion makes it here!
					
					// Now we perform round 13 and check. This is somewhat simpler, because we
					// already know what outputs we're looking for because we had to have that
					// on hand to compute the target outputs for round 12.
					
					f_s1( R[31]^K[ 5], R[ 0]^K[26], R[ 1]^K[41], R[ 2]^K[18], R[ 3]^K[53], R[ 4]^K[24],    L[ 8], L[16], L[22], L[30] );
					result &= ~(L[ 8] ^ L13[hs][ 8]) & ~(L[16] ^ L13[hs][16]) & ~(L[22] ^ L13[hs][22]) & ~(L[30] ^ L13[hs][30]);
					if (!result) goto stepper;
					
					f_s2( R[ 3]^K[10], R[ 4]^K[46], R[ 5]^K[12], R[ 6]^K[ 6], R[ 7]^K[54], R[ 8]^K[34],    L[12], L[27], L[ 1], L[17] );
					result &= ~(L[12] ^ L13[hs][12]) & ~(L[27] ^ L13[hs][27]) & ~(L[ 1] ^ L13[hs][ 1]) & ~(L[17] ^ L13[hs][17]);
					if (!result) goto stepper;
					
					f_s3( R[ 7]^K[11], R[ 8]^K[40], R[ 9]^K[48], R[10]^K[17], R[11]^K[32], R[12]^K[20],    L[23], L[15], L[29], L[ 5] );
					result &= ~(L[23] ^ L13[hs][23]) & ~(L[15] ^ L13[hs][15]) & ~(L[29] ^ L13[hs][29]) & ~(L[ 5] ^ L13[hs][ 5]);
					if (!result) goto stepper;
					
					f_s4( R[11]^K[19], R[12]^K[13], R[13]^K[39], R[14]^K[47], R[15]^K[55], R[16]^K[ 3],    L[25], L[19], L[ 9], L[ 0] );
					result &= ~(L[25] ^ L13[hs][25]) & ~(L[19] ^ L13[hs][19]) & ~(L[ 9] ^ L13[hs][ 9]) & ~(L[ 0] ^ L13[hs][ 0]);
					if (!result) goto stepper;
					
					f_s5( R[15]^K[49], R[16]^K[16], R[17]^K[38], R[18]^K[21], R[19]^K[36], R[20]^K[37],    L[ 7], L[13], L[24], L[ 2] );
					result &= ~(L[ 7] ^ L13[hs][ 7]) & ~(L[13] ^ L13[hs][13]) & ~(L[24] ^ L13[hs][24]) & ~(L[ 2] ^ L13[hs][ 2]);
					if (!result) goto stepper;
					
					f_s6( R[19]^K[31], R[20]^K[42], R[21]^K[ 9], R[22]^K[22], R[23]^K[52], R[24]^K[43],    L[ 3], L[28], L[10], L[18] );
					result &= ~(L[ 3] ^ L13[hs][ 3]) & ~(L[28] ^ L13[hs][28]) & ~(L[10] ^ L13[hs][10]) & ~(L[18] ^ L13[hs][18]);
					if (!result) goto stepper;
					
					f_s7( R[23]^K[15], R[24]^K[50], R[25]^K[35], R[26]^K[44], R[27]^K[ 0], R[28]^K[23],    L[31], L[11], L[21], L[ 6] );
					result &= ~(L[31] ^ L13[hs][31]) & ~(L[11] ^ L13[hs][11]) & ~(L[21] ^ L13[hs][21]) & ~(L[ 6] ^ L13[hs][ 6]);
					if (!result) goto stepper;
					
					f_s8( R[27]^K[29], R[28]^K[ 1], R[29]^K[ 2], R[30]^K[14], R[31]^K[51], R[ 0]^K[45],    L[ 4], L[26], L[14], L[20] );
					result &= ~(L[ 4] ^ L13[hs][ 4]) & ~(L[26] ^ L13[hs][26]) & ~(L[14] ^ L13[hs][14]) & ~(L[20] ^ L13[hs][20]);
					if (!result) goto stepper;
					
					// whoomp, there it is!
					return result;
				}
			stepper:
				++hs;
				if (hs & (1 << 0)) {
					K[10] = ~K[10];
				
				changeR2S1: // also for toggling bit 18
					// update in round 2
					{
						SBOX_0_INIT( L1[31]^K[54], L1[ 0]^K[18], L1[ 1]^K[33], L1[ 2]^K[10], L1[ 3]^K[20], L1[ 4]^K[48] )
						SBOX_0_BIT_0( PR[ 8], R2[ 8] )
						SBOX_0_BIT_1( PR[16], R2[16] )
						SBOX_0_BIT_2( PR[22], R2[22] )
						SBOX_0_BIT_3( PR[30], R2[30] )
					}
			/*		R2[ 8] = P[ 5];
					R2[16] = P[ 3];
					R2[22] = P[51];
					R2[30] = P[49];
					f_s1( L1[31]^K[54], L1[ 0]^K[18], L1[ 1]^K[33], L1[ 2]^K[10], L1[ 3]^K[20], L1[ 4]^K[48],
						R2[ 8], R2[16], R2[22], R2[30] ); */
					
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
					R2[12] = P[37];
					R2[27] = P[25];
					R2[ 1] = P[15];
					R2[17] = P[11];
					f_s2( L1[ 3]^K[34], L1[ 4]^K[13], L1[ 5]^K[ 4], L1[ 6]^K[55], L1[ 7]^K[46], L1[ 8]^K[26],
						R2[12], R2[27], R2[ 1], R2[17] );
					
					
					// and dependent boxes in round 3
					regen23 = 0xbb;
					continue;
				}
				if (hs & (1 << 3)) {
					K[49] = ~K[49];
					
					// update in round 2
					R2[31] = P[57];
					R2[11] = P[29];
					R2[21] = P[43];
					R2[ 6] = P[55];
					f_s7( L1[23]^K[ 9], L1[24]^K[44], L1[25]^K[29], L1[26]^K[ 7], L1[27]^K[49], L1[28]^K[45],
						R2[31], R2[11], R2[21], R2[ 6] );
					
					
					// and dependent boxes in round 3
					regen23 = 0xf5;
					continue;
				}
				break;
			}
			// now step the tail
            #if (CLIENT_OS == OS_MACOS)
		    if (DES_yield_ticks < *TICKS) {
	            DES_yield_ticks = *TICKS + DES_ticks_to_use;
			    mac_yield(0);
			}
            #endif

			hs = 0;
			K[49] = ~K[49];
			++ts;
			if (ts & (1 << 0)) {
				K[ 3] = ~K[ 3];
				//whack_init_00(); regen23 = 0xffff; continue;	///////////////////////////////////////////////////////////////////////////
			changeR1S1R15S3:
				// update in round 1
				L1[ 8] = P[ 4];
				L1[16] = P[ 2];
				L1[22] = P[50];
				L1[30] = P[48];
				f_s1( P[57]^K[47], P[ 7]^K[11], P[15]^K[26], P[23]^K[ 3], P[31]^K[13], P[39]^K[41],
					L1[ 8], L1[16], L1[22], L1[30] );
				
				// and dependent boxes in round 2
				regen23 = 0x7dff;
/*				
			changeR15S3:
*/
				for (int i = 0; i < 16; ++i) {
					// fix box in round 15
					{
						SBOX_2_INIT( R14[i][ 7]^K[39], R14[i][ 8]^K[11], R14[i][ 9]^K[19],
									 R14[i][10]^K[20], R14[i][11]^K[ 3], R14[i][12]^K[48] )
						SBOX_2_BIT_0( CL[23], L13[i][23] )
						SBOX_2_BIT_1( CL[15], L13[i][15] )
						SBOX_2_BIT_2( CL[29], L13[i][29] )
						SBOX_2_BIT_3( CL[ 5], L13[i][ 5] )
					}
				/*	L13[i][23] = C[59];
					L13[i][15] = C[61];
					L13[i][29] = C[41];
					L13[i][ 5] = C[47];
					f_s3( R14[i][ 7]^K[39], R14[i][ 8]^K[11], R14[i][ 9]^K[19],
						R14[i][10]^K[20], R14[i][11]^K[ 3], R14[i][12]^K[48],
						L13[i][23], L13[i][15], L13[i][29], L13[i][ 5] ); */
					
					// fix s1 and/or s3 in round 14, if necessary
					
					// and step
				/*	++i;
					     if (i & 0x01) K[10] = ~K[10];
					else if (i & 0x02) K[18] = ~K[18];
					else if (i & 0x04) K[46] = ~K[46];
					else if            K[49] = ~K[49]; */
				}
				continue;
			}
			if (ts & (1 << 1)) {
				K[11] = ~K[11];
				//whack_init_00(); regen23 = 0xffff; continue;	///////////////////////////////////////////////////////////////////////////
				goto changeR1S1R15S3;
			}
			/*if (ts & (1 << 2)) {  // why, oh why, remi? I wanted this one badly...
				K[39] = ~K[39];
				// update in round 1
				L1[12] = P[36];
				L1[27] = P[24];
				L1[ 1] = P[14];
				L1[17] = P[10];
				f_s2( P[31]^K[27], P[39]^K[ 6], P[47]^K[54], P[55]^K[48], P[63]^K[39], P[ 5]^K[19],
					L1[12], L1[27], L1[ 1], L1[17] );
				
				// and dependent boxes in round 2
				regen23 = 0xbbff;
				
				goto changeR15S3;
				}*/
			if (ts & (1 << 2)) {
				K[42] = ~K[42];
				// update in round 1
				L1[31] = P[56];
				L1[11] = P[28];
				L1[21] = P[42];
				L1[ 6] = P[54];
				f_s7( P[59]^K[ 2], P[ 1]^K[37], P[ 9]^K[22], P[17]^K[ 0], P[25]^K[42], P[33]^K[38],
					L1[31], L1[11], L1[21], L1[ 6] );
				
				// and dependent boxes in round 2
				regen23 = 0xf5ff;
				
				for (int i = 0; i < 16;) {
					// fix box in round 15
					L13[i][ 4] = C[39];
					L13[i][26] = C[17];
					L13[i][14] = C[53];
					L13[i][20] = C[35];
					f_s8( R14[i][27]^K[ 2], R14[i][28]^K[29], R14[i][29]^K[30], R14[i][30]^K[42], R14[i][31]^K[52], R14[i][ 0]^K[14],
						L13[i][ 4], L13[i][26], L13[i][14], L13[i][20] );
					
					// fix s1 and/or s3 in round 14, if necessary
					R12__8[i] = R14[i][ 8];
					R12_16[i] = R14[i][16];
					R12_22[i] = R14[i][22];
					R12_30[i] = R14[i][30];
					f_s1( L13[i][31]^K[19], L13[i][ 0]^K[40], L13[i][ 1]^K[55], L13[i][ 2]^K[32], L13[i][ 3]^K[10], L13[i][ 4]^K[13],
						R12__8[i], R12_16[i], R12_22[i], R12_30[i] );
					
					// and step
					++i;
					     if (i & 0x01) K[10] = ~K[10];
				/*	else if (i & 0x02) K[18] = ~K[18];
					else if (i & 0x04) K[46] = ~K[46];
					else if            K[49] = ~K[49]; */
				}
				continue;
			}
			if (ts & (1 << 3)) {
				K[ 5] = ~K[ 5];
				// update in round 1
				L1[23] = P[58];
				L1[15] = P[60];
				L1[29] = P[40];
				L1[ 5] = P[46];
				f_s3( P[63]^K[53], P[ 5]^K[25], P[13]^K[33], P[21]^K[34], P[29]^K[17], P[37]^K[ 5],
					L1[23], L1[15], L1[29], L1[ 5] );
				
				// and dependent boxes in round 2
				regen23 = 0x5fff;
				
				for (int i = 0; i < 16;) {
					// fix box in round 15
					L13[i][12] = C[37];
					L13[i][27] = C[25];
					L13[i][ 1] = C[15];
					L13[i][17] = C[11];
					f_s2( R14[i][ 3]^K[13], R14[i][ 4]^K[17], R14[i][ 5]^K[40], R14[i][ 6]^K[34], R14[i][ 7]^K[25], R14[i][ 8]^K[ 5],
						L13[i][12], L13[i][27], L13[i][ 1], L13[i][17] );
					
					// fix s1 and/or s3 in round 14, if necessary
					R12__8[i] = R14[i][ 8];
					R12_16[i] = R14[i][16];
					R12_22[i] = R14[i][22];
					R12_30[i] = R14[i][30];
					f_s1( L13[i][31]^K[19], L13[i][ 0]^K[40], L13[i][ 1]^K[55], L13[i][ 2]^K[32], L13[i][ 3]^K[10], L13[i][ 4]^K[13],
						R12__8[i], R12_16[i], R12_22[i], R12_30[i] );
					
					R12_23[i] = R14[i][23];
					R12_15[i] = R14[i][15];
					R12_29[i] = R14[i][29];
					R12__5[i] = R14[i][ 5];
					f_s3( L13[i][ 7]^K[25], L13[i][ 8]^K[54], L13[i][ 9]^K[ 5], L13[i][10]^K[ 6], L13[i][11]^K[46], L13[i][12]^K[34],
						R12_23[i], R12_15[i], R12_29[i], R12__5[i] );
					
					// and step
					++i;
					     if (i & 0x01) K[10] = ~K[10];
					else if (i & 0x02) K[18] = ~K[18];
					else if (i & 0x04) K[46] = ~K[46];
				//	else if            K[49] = ~K[49];
				}
				continue;
			}
			if (ts & (1 << 4)) {
				K[43] = ~K[43];
				// update in round 1
				L1[ 4] = P[38];
				L1[26] = P[16];
				L1[14] = P[52];
				L1[20] = P[34];
				f_s8( P[25]^K[16], P[33]^K[43], P[41]^K[44], P[49]^K[ 1], P[57]^K[ 7], P[ 7]^K[28],
					L1[ 4], L1[26], L1[14], L1[20] );
				
				// and dependent boxes in round 2
				regen23 = 0xdeff;
			changeR15S7:
				for (int i = 0; i < 16;) {
					// fix box in round 15
					L13[i][31] = C[57];
					L13[i][11] = C[29];
					L13[i][21] = C[43];
					L13[i][ 6] = C[55];
					f_s7( R14[i][23]^K[43], R14[i][24]^K[23], R14[i][25]^K[ 8], R14[i][26]^K[45], R14[i][27]^K[28], R14[i][28]^K[51],
						L13[i][31], L13[i][11], L13[i][21], L13[i][ 6] );
					
					// fix s1 and/or s3 in round 14, if necessary
					R12__8[i] = R14[i][ 8];
					R12_16[i] = R14[i][16];
					R12_22[i] = R14[i][22];
					R12_30[i] = R14[i][30];
					f_s1( L13[i][31]^K[19], L13[i][ 0]^K[40], L13[i][ 1]^K[55], L13[i][ 2]^K[32], L13[i][ 3]^K[10], L13[i][ 4]^K[13],
						R12__8[i], R12_16[i], R12_22[i], R12_30[i] );
					
					R12_23[i] = R14[i][23];
					R12_15[i] = R14[i][15];
					R12_29[i] = R14[i][29];
					R12__5[i] = R14[i][ 5];
					f_s3( L13[i][ 7]^K[25], L13[i][ 8]^K[54], L13[i][ 9]^K[ 5], L13[i][10]^K[ 6], L13[i][11]^K[46], L13[i][12]^K[34],
						R12_23[i], R12_15[i], R12_29[i], R12__5[i] );
					
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
				L1[ 7] = P[62];
				L1[13] = P[44];
				L1[24] = P[ 0];
				L1[ 2] = P[22];
				f_s5( P[61]^K[36], P[ 3]^K[31], P[11]^K[21], P[19]^K[ 8], P[27]^K[23], P[35]^K[52],
					L1[ 7], L1[13], L1[24], L1[ 2] );
				
				// and dependent boxes in round 2
				regen23 = 0xf7ff;
				
				goto changeR15S7;
			}
			/*if (ts & (1 << 7)) {  The other mix-up
				K[38] = ~K[38];
				// update in round 1
				L1[31] = P[56];
				L1[11] = P[28];
				L1[21] = P[42];
				L1[ 6] = P[54];
				f_s7( P[59]^K[ 2], P[ 1]^K[37], P[ 9]^K[22], P[17]^K[ 0], P[25]^K[42], P[33]^K[38],
					L1[31], L1[11], L1[21], L1[ 6] );
				
				// and dependent boxes in round 2
				regen23 = 0xffff;//0xfdff;
				
				for (int i = 0; i < 256;) {
					// fix box in round 15
					L13[i][ 7] = C[63];
					L13[i][13] = C[45];
					L13[i][24] = C[ 1];
					L13[i][ 2] = C[23];
					f_s5( R14[i][15]^K[22], R14[i][16]^K[44], R14[i][17]^K[ 7], R14[i][18]^K[49], R14[i][19]^K[ 9], R14[i][20]^K[38],
						L13[i][ 7], L13[i][13], L13[i][24], L13[i][ 2] );
					
					// fix s1 and/or s3 in round 14, if necessary
					R12__8[i] = R14[i][ 8];
					R12_16[i] = R14[i][16];
					R12_22[i] = R14[i][22];
					R12_30[i] = R14[i][30];
					f_s1( L13[i][31]^K[19], L13[i][ 0]^K[40], L13[i][ 1]^K[55], L13[i][ 2]^K[32], L13[i][ 3]^K[10], L13[i][ 4]^K[13],
						R12__8[i], R12_16[i], R12_22[i], R12_30[i] );
					
					R12_23[i] = R14[i][23];
					R12_15[i] = R14[i][15];
					R12_29[i] = R14[i][29];
					R12__5[i] = R14[i][ 5];
					f_s3( L13[i][ 7]^K[25], L13[i][ 8]^K[54], L13[i][ 9]^K[ 5], L13[i][10]^K[ 6], L13[i][11]^K[46], L13[i][12]^K[34],
						R12_23[i], R12_15[i], R12_29[i], R12__5[i] );
					
					// and step
					++i;
					     if (i & 0x01) K[10] = ~K[10];
					else if (i & 0x02) K[12] = ~K[12];
					else if (i & 0x04) K[15] = ~K[15];
					else if (i & 0x08) K[18] = ~K[18];
					else if (i & 0x10) K[45] = ~K[45];
					else if (i & 0x20) K[46] = ~K[46];
					else if (i & 0x40) K[49] = ~K[49];
					//else               K[50] = ~K[50];
				}
				continue;
			}*/
			
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
	return 0;
}



