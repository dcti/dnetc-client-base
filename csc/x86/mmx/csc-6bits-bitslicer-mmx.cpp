// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-6bits-bitslicer-mmx.cpp,v $
// Revision 1.1.2.1  1999/12/12 11:05:59  remi
// Moved from directory csc/x86/
//
// Revision 1.1.2.4  1999/12/05 14:39:43  remi
// A faster 6bit MMX core.
//
// Revision 1.1.2.3  1999/11/28 20:23:15  remi
// Updated core.
//
// Revision 1.1.2.2  1999/11/23 23:39:45  remi
// csc_transP() optimized.
// modified csc_transP() calling convention.
//
// Revision 1.1.2.1  1999/11/22 18:58:11  remi
// Initial commit of MMX'fied CSC cores.

#if (!defined(lint) && defined(__showids__))
const char * PASTE(csc_6bits_bitslicer_,CSC_SUFFIX) (void) {
return "@(#)$Id: csc-6bits-bitslicer-mmx.cpp,v 1.1.2.1 1999/12/12 11:05:59 remi Exp $"; }
#endif

#include <stdio.h>

// ------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
ulong
PASTE(cscipher_bitslicer_,CSC_SUFFIX)
( ulong key[2][64], const u8 keyB[8], const ulong msg[64], const ulong cipher[64], char *membuffer );
}
#endif

/*
  Memory usage :
  	- key	    :  512 bytes -+
	- plain     :  256 bytes  | allocated in csc-6bit-driver.cpp
	- cipher    :  256 bytes -+
  	- subkey    : 2816 bytes
	- cfr       :  256 bytes
	- tp1+tp2   :  256 bytes
	- totwiddle :  696 bytes
	- tcp+tep   : 2816 bytes
	- loc.var.  :   64 bytes
		      ----
                      7928 bytes (or 15856 bytes on a 64-bit cpu or with MMX)
 */

//#include <stdio.h>

ulong
PASTE(cscipher_bitslicer_,CSC_SUFFIX)
( ulong key[2][64], const u8 keyB[8], const ulong msg[64], const ulong cipher[64], char *membuffer )
{
  //ulong cfr[64];
  ulong (*cfr)[64] = (ulong(*)[64])membuffer;
  membuffer += (sizeof(*cfr) + 15) & 0xFFFFFFF0;

  //ulong subkey[9+2][64];
  ulong (*subkey)[9+2][64] = (ulong (*)[9+2][64])membuffer;
  membuffer += (sizeof(*subkey) + 15) & 0xFFFFFFF0;

  //ulong *totwiddle[6][7*(1+3)+1];
  ulong *(*totwiddle)[6][7*(1+3)+1] = (ulong *(*)[6][7*(1+3)+1])membuffer;
  membuffer += (sizeof(*totwiddle) + 15) & 0xFFFFFFF0;

  //ulong tp1[4][8], tp2[4][8];
  ulong (*tp1)[4][8] = (ulong (*)[4][8])membuffer;
  membuffer += (sizeof(*tp1) + 15) & 0xFFFFFFF0;
  ulong (*tp2)[4][8] = (ulong (*)[4][8])membuffer;
  membuffer += (sizeof(*tp2) + 15) & 0xFFFFFFF0;

  // allocate parameters on the aligned membuffer
  csc_mmxParameters *csc_params = (csc_mmxParameters*) membuffer;
  membuffer += (sizeof(*csc_params) + 15) & 0xFFFFFFF0;

  ulong *skp;  // subkey[n]
  ulong *skp1; // subkey[n-1]
  const ulong *tcp; // pointer to tabc[] (bitslice values of c0..c8)
  const ulong *tep; // pointer to tabe[] (bitslice values of e and e')

  // for the inlined calls to csc_transP2()
#define _in0  " 0(%%edx)"
#define _in1  " 8(%%edx)"
#define _in2  "16(%%edx)"
#define _in3  "24(%%edx)"
#define _in4  "32(%%edx)"
#define _in5  "40(%%edx)"
#define _in6  "48(%%edx)"
#define _in7  "56(%%edx)"
#define _out0 " 0(%%ecx)"
#define _out1 " 4(%%ecx)"
#define _out2 " 8(%%ecx)"
#define _out3 "12(%%ecx)"
#define _out4 "16(%%ecx)"
#define _out5 "20(%%ecx)"
#define _out6 "24(%%ecx)"
#define _out7 "28(%%ecx)"
#define _mmNOT "%4"

/*
#define APPLY_MP0(adr, adl)						\
  csc_transP_call( 							\
              (*tp1)[adr/16][7], (*tp1)[adr/16][6],			\
              (*tp1)[adr/16][5], (*tp1)[adr/16][4],			\
	      (*tp1)[adr/16][3], (*tp1)[adr/16][2],			\
	      (*tp1)[adr/16][1], (*tp1)[adr/16][0],			\
	  (*cfr)[adl+7], (*cfr)[adl+6], (*cfr)[adl+5], (*cfr)[adl+4],	\
	  (*cfr)[adl+3], (*cfr)[adl+2], (*cfr)[adl+1], (*cfr)[adl+0] );	\
  csc_transP_call( 							\
	      (*tp2)[adr/16][7], (*tp2)[adr/16][6],			\
              (*tp2)[adr/16][5], (*tp2)[adr/16][4],			\
              (*tp2)[adr/16][3], (*tp2)[adr/16][2],			\
	      (*tp2)[adr/16][1], (*tp2)[adr/16][0],			\
	  (*cfr)[adr+7], (*cfr)[adr+6], (*cfr)[adr+5], (*cfr)[adr+4],	\
	  (*cfr)[adr+3], (*cfr)[adr+2], (*cfr)[adr+1], (*cfr)[adr+0] )
*/
#define APPLY_MP0(adr, adl)		\
  do {					\
  asm volatile ("
	## eax == cfr
	## ebx == tp1 / tp2
	## edx == csc_params

	movl	%1, %%ebx	# tp1
	movl	%0, %%eax	# cfr
	addl	$8*8*("#adr"/16), %%ebx
	addl	$8*"#adl", %%eax

	movq	8*0(%%ebx), %%mm0
	movl	%%eax, 8*8+4*0(%%edx)
	addl	$8, %%eax
	movq	8*1(%%ebx), %%mm1
	movl	%%eax, 8*8+4*1(%%edx)
	addl	$8, %%eax
	movq	8*2(%%ebx), %%mm2
	movl	%%eax, 8*8+4*2(%%edx)
	addl	$8, %%eax
	movq	8*3(%%ebx), %%mm3
	movl	%%eax, 8*8+4*3(%%edx)
	addl	$8, %%eax
	movq	8*4(%%ebx), %%mm4
	movl	%%eax, 8*8+4*4(%%edx)
	addl	$8, %%eax
	movq	8*5(%%ebx), %%mm5
	movl	%%eax, 8*8+4*5(%%edx)
	addl	$8, %%eax
	movq	8*6(%%ebx), %%mm6
	movl	%%eax, 8*8+4*6(%%edx)
	addl	$8, %%eax
	movq	8*7(%%ebx), %%mm7
	movl	%%eax, 8*8+4*7(%%edx)

	pushl	%%edx
	call	csc_transP2
	popl	%%edx

	movl	%2, %%ebx	# tp2
	movl	%0, %%eax	# cfr
	addl	$8*8*("#adr"/16), %%ebx
	addl	$8*"#adr", %%eax

	movq	8*0(%%ebx), %%mm0
	movl	%%eax, 8*8+4*0(%%edx)
	addl	$8, %%eax
	movq	8*1(%%ebx), %%mm1
	movl	%%eax, 8*8+4*1(%%edx)
	addl	$8, %%eax
	movq	8*2(%%ebx), %%mm2
	movl	%%eax, 8*8+4*2(%%edx)
	addl	$8, %%eax
	movq	8*3(%%ebx), %%mm3
	movl	%%eax, 8*8+4*3(%%edx)
	addl	$8, %%eax
	movq	8*4(%%ebx), %%mm4
	movl	%%eax, 8*8+4*4(%%edx)
	addl	$8, %%eax
	movq	8*5(%%ebx), %%mm5
	movl	%%eax, 8*8+4*5(%%edx)
	addl	$8, %%eax
	movq	8*6(%%ebx), %%mm6
	movl	%%eax, 8*8+4*6(%%edx)
	addl	$8, %%eax
	movq	8*7(%%ebx), %%mm7
	movl	%%eax, 8*8+4*7(%%edx)

	pushl	%%edx
	call	csc_transP2
	popl	%%edx
  " : 							\
    : "m"(cfr), "m"(tp1), "m"(tp2), "d"(csc_params)	\
    : "%ecx", "%eax","%ebx"				\
  );							\
  } while( 0 )

/*
#define APPLY_Ms(adr, adl)						\
  do {									\
  ulong x0,x1,x2,x3,x4,x5,x6,x7, y1,y3,y5,y7;				\
  ulong xy56, xy34, xy12, xy70;						\
									\
  x0 = (*cfr)[adl+0] ^ (skp[0+8] ^= skp[0+8-128]);			\
  x1 = (*cfr)[adl+1] ^ (skp[1+8] ^= skp[1+8-128]);			\
  x2 = (*cfr)[adl+2] ^ (skp[2+8] ^= skp[2+8-128]);			\
  x3 = (*cfr)[adl+3] ^ (skp[3+8] ^= skp[3+8-128]);			\
  x4 = (*cfr)[adl+4] ^ (skp[4+8] ^= skp[4+8-128]);			\
  x5 = (*cfr)[adl+5] ^ (skp[5+8] ^= skp[5+8-128]);			\
  x6 = (*cfr)[adl+6] ^ (skp[6+8] ^= skp[6+8-128]);			\
  x7 = (*cfr)[adl+7] ^ (skp[7+8] ^= skp[7+8-128]);			\
  csc_transP_call(                                                      \
          x7 ^ (y7   =      (*cfr)[adr+7] ^ (skp[7] ^= skp[7-128])),	\
	  x6 ^ (xy56 = x5 ^ (*cfr)[adr+6] ^ (skp[6] ^= skp[6-128])),	\
	  x5 ^ (y5   =      (*cfr)[adr+5] ^ (skp[5] ^= skp[5-128])),	\
	  x4 ^ (xy34 = x3 ^ (*cfr)[adr+4] ^ (skp[4] ^= skp[4-128])),	\
	  x3 ^ (y3   =      (*cfr)[adr+3] ^ (skp[3] ^= skp[3-128])),	\
	  x2 ^ (xy12 = x1 ^ (*cfr)[adr+2] ^ (skp[2] ^= skp[2-128])),	\
	  x1 ^ (y1   =      (*cfr)[adr+1] ^ (skp[1] ^= skp[1-128])),	\
	  x0 ^ (xy70 = x7 ^ (*cfr)[adr+0] ^ (skp[0] ^= skp[0-128])),	\
	  (*cfr)[adl+7], (*cfr)[adl+6], (*cfr)[adl+5], (*cfr)[adl+4],	\
	  (*cfr)[adl+3], (*cfr)[adl+2], (*cfr)[adl+1], (*cfr)[adl+0] );	\
  csc_transP_call(                                                      \
          x6 ^ y7, xy56, x4 ^ y5, xy34,					\
	  x2 ^ y3, xy12, x0 ^ y1, xy70,					\
	  (*cfr)[adr+7], (*cfr)[adr+6], (*cfr)[adr+5], (*cfr)[adr+4],	\
	  (*cfr)[adr+3], (*cfr)[adr+2], (*cfr)[adr+1], (*cfr)[adr+0] );	\
  skp += 16;								\
  } while (0)
*/

#define APPLY_Ms(adr, adl)		\
  do {					\
  ulong *x_y_xy = (ulong*)membuffer;	\
  asm volatile ("

	## eax == cfr
	## ebx == skp
	## edi == x_y_xy[8]
	## edx == csc_params
	##        y[4] == ecx + 8*8
	##        xy[4] == ecx + 8*8 + 8*4

# skp[4  ] ^= skp[4  -128]
# skp[4+8] ^= skp[4+8-128];
	movq	8*(4)(%%ebx), %%mm0
	movq	8*(4+8)(%%ebx), %%mm1
	pxor	8*(4-128)(%%ebx), %%mm0
	pxor	8*(4+8-128)(%%ebx), %%mm1
	movq	%%mm0, 8*(4)(%%ebx)
	movq	%%mm1, 8*(4+8)(%%ebx)

# skp[5  ] ^= skp[5  -128]
# skp[5+8] ^= skp[5+8-128];
	movq	8*(5)(%%ebx), %%mm2
	movq	8*(5+8)(%%ebx), %%mm3
	pxor	8*(5-128)(%%ebx), %%mm2
	pxor	8*(5+8-128)(%%ebx), %%mm3
	movq	%%mm2, 8*(5)(%%ebx)
	movq	%%mm3, 8*(5+8)(%%ebx)

# skp[6  ] ^= skp[6  -128]
# skp[6+8] ^= skp[6+8-128];
	movq	8*(6)(%%ebx), %%mm4
	movq	8*(6+8)(%%ebx), %%mm5
	pxor	8*(6-128)(%%ebx), %%mm4
	pxor	8*(6+8-128)(%%ebx), %%mm5
	movq	%%mm4, 8*(6)(%%ebx)
	movq	%%mm5, 8*(6+8)(%%ebx)

# skp[0  ] ^= skp[0  -128]
# skp[0+8] ^= skp[0+8-128];
	movq	8*(0)(%%ebx), %%mm0
	movq	8*(0+8)(%%ebx), %%mm1
	pxor	8*(0-128)(%%ebx), %%mm0
	pxor	8*(0+8-128)(%%ebx), %%mm1
	movq	%%mm0, 8*(0)(%%ebx)
	movq	%%mm1, 8*(0+8)(%%ebx)

# skp[7  ] ^= skp[7  -128]
# skp[7+8] ^= skp[7+8-128];
	movq	8*(7)(%%ebx), %%mm6
	movq	8*(7+8)(%%ebx), %%mm7
	pxor	8*(7-128)(%%ebx), %%mm6
	pxor	8*(7+8-128)(%%ebx), %%mm7
	movq	%%mm6, 8*(7)(%%ebx)
	movq	%%mm7, 8*(7+8)(%%ebx)

# skp[1  ] ^= skp[1  -128]
# skp[1+8] ^= skp[1+8-128];
	movq	8*(1)(%%ebx), %%mm0
	movq	8*(1+8)(%%ebx), %%mm2
	pxor	8*(1-128)(%%ebx), %%mm0
	pxor	8*(1+8-128)(%%ebx), %%mm2
	movq	%%mm0, 8*(1)(%%ebx)
	movq	%%mm2, 8*(1+8)(%%ebx)

# skp[2  ] ^= skp[2  -128]
# skp[2+8] ^= skp[2+8-128];
	movq	8*(2)(%%ebx), %%mm3
	movq	8*(2+8)(%%ebx), %%mm4
	pxor	8*(2-128)(%%ebx), %%mm3
	pxor	8*(2+8-128)(%%ebx), %%mm4
	movq	%%mm3, 8*(2)(%%ebx)
	movq	%%mm4, 8*(2+8)(%%ebx)

# skp[3  ] ^= skp[3  -128]
# skp[3+8] ^= skp[3+8-128];
	movq	8*(3)(%%ebx), %%mm5
	movq	8*(3+8)(%%ebx), %%mm6
	pxor	8*(3-128)(%%ebx), %%mm5
	pxor	8*(3+8-128)(%%ebx), %%mm6
	movq	%%mm5, 8*(3)(%%ebx)
	movq	%%mm6, 8*(3+8)(%%ebx)

#    (%%edi) == x0, x1y2, x2, x3y4, x4, x5y6, x6, x7y0
# 8*8(%%edi) == y1, y3, y5, y7, (in7)

# x1 = (*cfr)[adl+1] ^ skp[1+8];
# in1 = x1 ^ (y1 = (*cfr)[adr+1] ^ skp[1]);
	movq	8*("#adl"+1)(%%eax), %%mm7	# mm7 = (*cfr)[adl+1]
	movq	8*("#adr"+1)(%%eax), %%mm1	# mm1 = (*cfr)[adr+1]
	pxor	%%mm2, %%mm7			# mm7(x1) = (*cfr)[adl+1] ^ skp[1+8]
	pxor	%%mm0, %%mm1			# mm1(y1) = (*cfr)[adr+1] ^ skp[1]
	movq	%%mm1, 8*(0+8)(%%edi)		# (save y1)
	leal	8*("#adl"+0)(%%eax), %%esi
	pxor	%%mm7, %%mm1			# mm1(in1) = x1 ^ y1
# x2 = (*cfr)[adl+2] ^ skp[2+8];
# in2 = x2 ^ (x1y2 = x1 ^ (*cfr)[adr+2] ^ skp[2]);
	movl	%%esi, 8*8+4*0(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+2)(%%eax), %%mm7	# mm7 = x1 ^ (*cfr)[adr+2]
	movq	8*("#adl"+2)(%%eax), %%mm2	# mm2 = (*cfr)[adl+2]
	pxor	%%mm3, %%mm7			# mm7(x1y2) = x1 ^ (*cfr)[adr+2] ^ skp[2]
	pxor	%%mm4, %%mm2			# mm2(x2) = (*cfr)[adl+2] ^ skp[2+8]
	movq	%%mm7, 8*(1+0)(%%edi)		# (save x1y2)
	movq	%%mm2, 8*(2+0)(%%edi)		# (save x2)
	movl	%%esi, 8*8+4*1(%%edx)
	pxor	%%mm7, %%mm2			# mm2(in2) = x2 ^ x1y2

# x3 = (*cfr)[adl+3] ^ skp[3+8];
# in3 = x3 ^ (y3 = (*cfr)[adr+3] ^ skp[3]);
	movq	8*("#adl"+3)(%%eax), %%mm7
	movq	8*("#adr"+3)(%%eax), %%mm3
	pxor	%%mm6, %%mm7
	pxor	%%mm5, %%mm3
	movq	%%mm3, 8*(1+8)(%%edi)
	addl	$8, %%esi
	pxor	%%mm7, %%mm3
# x4 = (*cfr)[adl+4] ^ skp[4+8];
# in4 = x4 ^ (x3y4 = x3 ^ (*cfr)[adr+4] ^ skp[4]);
	movl	%%esi, 8*8+4*2(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+4)(%%eax), %%mm7
	movq	8*("#adl"+4)(%%eax), %%mm4
	pxor	8*(4+0)(%%ebx), %%mm7
	pxor	8*(4+8)(%%ebx), %%mm4
	movq	%%mm7, 8*(3+0)(%%edi)
	movq	%%mm4, 8*(4+0)(%%edi)
	movl	%%esi, 8*8+4*3(%%edx)
	pxor	%%mm7, %%mm4

# x5 = (*cfr)[adl+5] ^ skp[5+8];
# in5 = x5 ^ (y5 = (*cfr)[adr+5] ^ skp[5]);
	movq	8*("#adl"+5)(%%eax), %%mm7
	movq	8*("#adr"+5)(%%eax), %%mm5
	pxor	8*(5+8)(%%ebx), %%mm7
	pxor	8*(5+0)(%%ebx), %%mm5
	movq	%%mm5, 8*(2+8)(%%edi)
	addl	$8, %%esi
	pxor	%%mm7, %%mm5
# x6 = (*cfr)[adl+6] ^ skp[6+8];
# in6 = x6 ^ (x5y6 = x5 ^ (*cfr)[adr+6] ^ skp[6]);
	movl	%%esi, 8*8+4*4(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+6)(%%eax), %%mm7
	movq	8*("#adl"+6)(%%eax), %%mm6
	pxor	8*(6+0)(%%ebx), %%mm7
	pxor	8*(6+8)(%%ebx), %%mm6
	movq	%%mm7, 8*(5+0)(%%edi)
	movq	%%mm6, 8*(6+0)(%%edi)
	movl	%%esi, 8*8+4*5(%%edx)
	pxor	%%mm7, %%mm6

# x7 = (*cfr)[adl+7] ^ skp[7+8];
# in7 = x7 ^ (y7 = (*cfr)[adr+7] ^ skp[7]);
	movq	8*("#adl"+7)(%%eax), %%mm0	# mm0 = (*cfr)[adl+7]
	movq	8*("#adr"+7)(%%eax), %%mm7	# mm7 = (*cfr)[adr+7]
	pxor	8*(7+8)(%%ebx), %%mm0		# mm0(x7) = (*cfr)[adl+7] ^ skp[7+8]
	pxor	8*(7+0)(%%ebx), %%mm7		# mm7(y7) = (*cfr)[adr+7] ^ skp[7]
	movq	%%mm7, 8*(3+8)(%%edi)		# (save y7)
	pxor	%%mm0, %%mm7			# mm7(in7) = x7 ^ y7
	movq	%%mm7, 8*(4+8)(%%edi)		# (save in7)
	addl	$8, %%esi
# x0 = (*cfr)[adl+0] ^ skp[0+8];
# in0 = x0 ^ (x7y0 = x7 ^ (*cfr)[adr+0] ^ skp[0]);
	movl	%%esi, 8*8+4*6(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+0)(%%eax), %%mm0	# mm0 = x7 ^ (*cfr)[adr+0]
	movq	8*("#adl"+0)(%%eax), %%mm7	# mm7 = (*cfr)[adl+0]
	pxor	8*(0+0)(%%ebx), %%mm0		# mm0(x7y0) = x7 ^ (*cfr)[adr+0] ^ skp[0]
	pxor	8*(0+8)(%%ebx), %%mm7		# mm7(x0) = (*cfr)[adl+0] ^ skp[0+8]
	movq	%%mm0, 8*(7+0)(%%edi)		#(store x7y0)
	movq	%%mm7, 8*(0+0)(%%edi)		#(store x0)
	pxor	%%mm7, %%mm0			# mm0(in0) = x0 ^ x7y0
	movq	8*(4+8)(%%edi), %%mm7		# mm7(in7) = in7

	movl	%%esi, 8*8+4*7(%%edx)
	
	pushl	%%eax
	pushl	%%edx
	call	csc_transP2
	popl	%%edx
	popl	%%eax

	# in7 = x6 ^ y7;
	# in6 = x5y6;
	leal	8*("#adr"+0)(%%eax), %%esi
	movq	8*(3+8)(%%edi), %%mm7
	movl	%%esi, 8*8+4*0(%%edx)
	movq	8*(5+0)(%%edi), %%mm6
	addl	$8, %%esi
	pxor	8*(6+0)(%%edi), %%mm7
	movl	%%esi, 8*8+4*1(%%edx)

	# in5 = x4 ^ y5;
	# in4 = x3y4;
	addl	$8, %%esi
	movq	8*(2+8)(%%edi), %%mm5
	movl	%%esi, 8*8+4*2(%%edx)
	movq	8*(3+0)(%%edi), %%mm4
	addl	$8, %%esi
	pxor	8*(4+0)(%%edi), %%mm5
	movl	%%esi, 8*8+4*3(%%edx)

	# in3 = x2 ^ y3;
	# in2 = x1y2;
	addl	$8, %%esi
	movq	8*(1+8)(%%edi), %%mm3
	movl	%%esi, 8*8+4*4(%%edx)
	movq	8*(1+0)(%%edi), %%mm2
	addl	$8, %%esi
	pxor	8*(2+0)(%%edi), %%mm3
	movl	%%esi, 8*8+4*5(%%edx)

	# in1 = x0 ^ y1;
	# in0 = x7y0;
	addl	$8, %%esi
	movq	8*(0+8)(%%edi), %%mm1
	movl	%%esi, 8*8+4*6(%%edx)
	movq	8*(7+0)(%%edi), %%mm0
	addl	$8, %%esi
	pxor	8*(0+0)(%%edi), %%mm1
	movl	%%esi, 8*8+4*7(%%edx)

	pushl	%%eax
	pushl	%%edx
	call	csc_transP2
	popl	%%edx
	popl	%%eax

	addl	$16*8, %%ebx	# skp += 16;

  " : "+b"(skp)					\
    : "D"(x_y_xy), "a"(cfr), "d"(csc_params)	\
    : "%ecx", "%esi"				\
  );						\
  } while( 0 )

/*
#define APPLY_Me(adr, adl)						\
  do {									\
  ulong x0,x1,x2,x3,x4,x5,x6,x7, y1,y3,y5,y7;				\
  ulong xy56, xy34, xy12, xy70;						\
									\
  x0 = (*cfr)[adl+0] ^ tep[0+8]; x1 = (*cfr)[adl+1] ^ tep[1+8];		\
  x2 = (*cfr)[adl+2] ^ tep[2+8]; x3 = (*cfr)[adl+3] ^ tep[3+8];		\
  x4 = (*cfr)[adl+4] ^ tep[4+8]; x5 = (*cfr)[adl+5] ^ tep[5+8];		\
  x6 = (*cfr)[adl+6] ^ tep[6+8]; x7 = (*cfr)[adl+7] ^ tep[7+8];		\
  csc_transP_call( 							\
	      x7 ^ (y7   =      (*cfr)[adr+7] ^ tep[7]),		\
	      x6 ^ (xy56 = x5 ^ (*cfr)[adr+6] ^ tep[6]),		\
	      x5 ^ (y5   =      (*cfr)[adr+5] ^ tep[5]),		\
	      x4 ^ (xy34 = x3 ^ (*cfr)[adr+4] ^ tep[4]),		\
	      x3 ^ (y3   =      (*cfr)[adr+3] ^ tep[3]),		\
	      x2 ^ (xy12 = x1 ^ (*cfr)[adr+2] ^ tep[2]),		\
	      x1 ^ (y1   =      (*cfr)[adr+1] ^ tep[1]),		\
	      x0 ^ (xy70 = x7 ^ (*cfr)[adr+0] ^ tep[0]),		\
	  (*cfr)[adl+7], (*cfr)[adl+6], (*cfr)[adl+5], (*cfr)[adl+4],	\
	  (*cfr)[adl+3], (*cfr)[adl+2], (*cfr)[adl+1], (*cfr)[adl+0] );	\
  csc_transP_call( 							\
	      x6 ^ y7, xy56, x4 ^ y5, xy34,				\
   	      x2 ^ y3, xy12, x0 ^ y1, xy70,				\
	  (*cfr)[adr+7], (*cfr)[adr+6], (*cfr)[adr+5], (*cfr)[adr+4],	\
	  (*cfr)[adr+3], (*cfr)[adr+2], (*cfr)[adr+1], (*cfr)[adr+0] );	\
  tep += 16;								\
  } while (0)
*/

#define APPLY_Me(adr, adl)		\
  do {					\
  ulong *x_y_xy = (ulong*)membuffer;	\
  asm volatile ("

	## eax == cfr
	## ebx == tep
	## edi == x_y_xy[8]
	## edx == csc_params
	##        y[4] == ecx + 8*8
	##        xy[4] == ecx + 8*8 + 8*4

#    (%%edi) == x0, x1y2, x2, x3y4, x4, x5y6, x6, x7y0
# 8*8(%%edi) == y1, y3, y5, y7, (in7)

# x1 = (*cfr)[adl+1] ^ tep[1+8];
# in1 = x1 ^ (y1 = (*cfr)[adr+1] ^ tep[1]);
	movq	8*("#adl"+1)(%%eax), %%mm7	# mm7 = (*cfr)[adl+1]
	movq	8*("#adr"+1)(%%eax), %%mm1	# mm1 = (*cfr)[adr+1]
	pxor	8*(1+8)(%%ebx), %%mm7		# mm7(x1) = (*cfr)[adl+1] ^ tep[1+8]
	pxor	8*(1+0)(%%ebx), %%mm1		# mm1(y1) = (*cfr)[adr+1] ^ tep[1]
	movq	%%mm1, 8*(0+8)(%%edi)		# (save y1)
	leal	8*("#adl"+0)(%%eax), %%esi
	pxor	%%mm7, %%mm1			# mm1(in1) = x1 ^ y1
# x2 = (*cfr)[adl+2] ^ tep[2+8];
# in2 = x2 ^ (x1y2 = x1 ^ (*cfr)[adr+2] ^ tep[2]);
	movl	%%esi, 8*8+4*0(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+2)(%%eax), %%mm7	# mm7 = x1 ^ (*cfr)[adr+2]
	movq	8*("#adl"+2)(%%eax), %%mm2	# mm2 = (*cfr)[adl+2]
	pxor	8*(2+0)(%%ebx), %%mm7		# mm7(x1y2) = x1 ^ (*cfr)[adr+2] ^ tep[2]
	pxor	8*(2+8)(%%ebx), %%mm2		# mm2(x2) = (*cfr)[adl+2] ^ tep[2+8]
	movq	%%mm7, 8*(1+0)(%%edi)		# (save x1y2)
	movq	%%mm2, 8*(2+0)(%%edi)		# (save x2)
	movl	%%esi, 8*8+4*1(%%edx)
	pxor	%%mm7, %%mm2			# mm2(in2) = x2 ^ x1y2

# x3 = (*cfr)[adl+3] ^ tep[3+8];
# in3 = x3 ^ (y3 = (*cfr)[adr+3] ^ tep[3]);
	movq	8*("#adl"+3)(%%eax), %%mm7
	movq	8*("#adr"+3)(%%eax), %%mm3
	pxor	8*(3+8)(%%ebx), %%mm7
	pxor	8*(3+0)(%%ebx), %%mm3
	movq	%%mm3, 8*(1+8)(%%edi)
	addl	$8, %%esi
	pxor	%%mm7, %%mm3
# x4 = (*cfr)[adl+4] ^ tep[4+8];
# in4 = x4 ^ (x3y4 = x3 ^ (*cfr)[adr+4] ^ tep[4]);
	movl	%%esi, 8*8+4*2(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+4)(%%eax), %%mm7
	movq	8*("#adl"+4)(%%eax), %%mm4
	pxor	8*(4+0)(%%ebx), %%mm7
	pxor	8*(4+8)(%%ebx), %%mm4
	movq	%%mm7, 8*(3+0)(%%edi)
	movq	%%mm4, 8*(4+0)(%%edi)
	movl	%%esi, 8*8+4*3(%%edx)
	pxor	%%mm7, %%mm4

# x5 = (*cfr)[adl+5] ^ tep[5+8];
# in5 = x5 ^ (y5 = (*cfr)[adr+5] ^ tep[5]);
	movq	8*("#adl"+5)(%%eax), %%mm7
	movq	8*("#adr"+5)(%%eax), %%mm5
	pxor	8*(5+8)(%%ebx), %%mm7
	pxor	8*(5+0)(%%ebx), %%mm5
	movq	%%mm5, 8*(2+8)(%%edi)
	addl	$8, %%esi
	pxor	%%mm7, %%mm5
# x6 = (*cfr)[adl+6] ^ tep[6+8];
# in6 = x6 ^ (x5y6 = x5 ^ (*cfr)[adr+6] ^ tep[6]);
	movl	%%esi, 8*8+4*4(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+6)(%%eax), %%mm7
	movq	8*("#adl"+6)(%%eax), %%mm6
	pxor	8*(6+0)(%%ebx), %%mm7
	pxor	8*(6+8)(%%ebx), %%mm6
	movq	%%mm7, 8*(5+0)(%%edi)
	movq	%%mm6, 8*(6+0)(%%edi)
	movl	%%esi, 8*8+4*5(%%edx)
	pxor	%%mm7, %%mm6

# x7 = (*cfr)[adl+7] ^ tep[7+8];
# in7 = x7 ^ (y7 = (*cfr)[adr+7] ^ tep[7]);
	movq	8*("#adl"+7)(%%eax), %%mm0	# mm0 = (*cfr)[adl+7]
	movq	8*("#adr"+7)(%%eax), %%mm7	# mm7 = (*cfr)[adr+7]
	pxor	8*(7+8)(%%ebx), %%mm0		# mm0(x7) = (*cfr)[adl+7] ^ tep[7+8]
	pxor	8*(7+0)(%%ebx), %%mm7		# mm7(y7) = (*cfr)[adr+7] ^ tep[7]
	movq	%%mm7, 8*(3+8)(%%edi)		# (save y7)
	pxor	%%mm0, %%mm7			# mm7(in7) = x7 ^ y7
	movq	%%mm7, 8*(4+8)(%%edi)		# (save in7)
	addl	$8, %%esi
# x0 = (*cfr)[adl+0] ^ tep[0+8];
# in0 = x0 ^ (x7y0 = x7 ^ (*cfr)[adr+0] ^ tep[0]);
	movl	%%esi, 8*8+4*6(%%edx)
	addl	$8, %%esi
	pxor	8*("#adr"+0)(%%eax), %%mm0	# mm0 = x7 ^ (*cfr)[adr+0]
	movq	8*("#adl"+0)(%%eax), %%mm7	# mm7 = (*cfr)[adl+0]
	pxor	8*(0+0)(%%ebx), %%mm0		# mm0(x7y0) = x7 ^ (*cfr)[adr+0] ^ tep[0]
	pxor	8*(0+8)(%%ebx), %%mm7		# mm7(x0) = (*cfr)[adl+0] ^ tep[0+8]
	movq	%%mm0, 8*(7+0)(%%edi)		#(store x7y0)
	movq	%%mm7, 8*(0+0)(%%edi)		#(store x0)
	pxor	%%mm7, %%mm0			# mm0(in0) = x0 ^ x7y0
	movq	8*(4+8)(%%edi), %%mm7		# mm7(in7) = in7

	movl	%%esi, 8*8+4*7(%%edx)
	
	pushl	%%eax
# ----------------------------------------------------
#	pushl	%%edx
#	call	csc_transP2
	
	leal	64(%%edx), %%ecx

  ## //csc_transF( in3, in2, in1, in0,	// in
  ## //            in7, in6, in5, in4 );// xor-out
  ## {
  ## ulong t04 = ~in3;		       ulong t06 =  in2 | in3;
  ## in4      ^=  t04 | in0;	       ulong t07 =  t06 ^ t04;
  ## ulong t09 =  (t07 ^ in1) | in2;   in7      ^=  t07;
  ## in6      ^=  t09;                 in5      ^=  (t09 ^ in0) | in1;
  ## }
	movq	%%mm0, (%%edx)		# (store _in0)

	movq	%%mm7, 8*7(%%edx)	# (store _in7)
	movq	%%mm3, %%mm7		# mm7 = in3
	movq	%%mm3, 8*3(%%edx)	# (store _in3)
	pxor	%4, %%mm3		# mm3 = t04 = ~in3
	movq	%%mm6, 8*6(%%edx)	# (store _in6)
	movq	%%mm3, %%mm6		# mm6 = t04
	por	%%mm2, %%mm7		# mm7 = t06 = in2 | in3
	por	%%mm0, %%mm3		# mm3 = t04 | in0
	movq	%%mm5, 8*5(%%edx)	# (store _in5)
	pxor	%%mm6, %%mm7		# mm7 = t07 = t06 ^ to4
	pxor	%%mm3, %%mm4		# mm4 = in4 ^= t04 | in0
	movq	%%mm7, %%mm5		# mm5 = t07
	pxor	%%mm1, %%mm5		# mm5 = t07 ^ in1
	pxor	8*7(%%edx), %%mm7	# mm7 = in7 ^= t07
	por	%%mm2, %%mm5		# mm5 = t09 = (t07 ^ in1) | in2
	movq	%%mm5, %%mm6		# mm6 = t09
	pxor	(%%edx), %%mm6		# mm6 = t09 ^ in0
	#pxor	%%mm0, %%mm6
	pxor	8*6(%%edx), %%mm5	# mm5 = in6 ^= t09
	por	%%mm1, %%mm6		# mm6 = (t09 ^ in0) | in1
	pxor	8*5(%%edx), %%mm6	# mm6 = in5 ^= (t09 ^ in0) | in1
	
  ## // csc_transG( in7, in6, in5, in4,  // in
  ## //             in3, in2, in1, in0 );// xor-out
  ## {
  ## ulong t06 = (in4 & in5) ^ (in4 | in7);   ulong t08 = (in4 | in6) ^ in7;
  ## out2 = (in2 ^= t06);                     ulong t10 = (t08 | t06) ^ in4;
  ## ulong t13 = (in4 & in6) ^ (in4 | in5);   out0 = (in0 ^= t10);
  ## out3 = (in3 ^= ~t13);                    out1 = (in1 ^= ~(t10 ^ t08) ^ (t13 | t06));
  ## }

	movq	%%mm6, 8*5(%%edx)	# -- mm6 free
	pand	%%mm4, %%mm6		# mm6 = in4 & in5
	movq	%%mm7, 8*7(%%edx)	# -- mm7 free
	movq	%%mm7, %%mm3		# mm3 = in7
	por	%%mm4, %%mm7		# mm7 = in4 | in7
	movq	%%mm5, 8*6(%%edx)	# mm5 free
	pxor	%%mm6, %%mm7		# + mm7 = t06 = (in4 & in5) ^ (in4 | in7)
					# -- mm6 free
	por	%%mm4, %%mm5		# mm5 = in4 | in6
	pxor	%%mm7, %%mm2		#### mm2 = out2 = in2 ^= t06
	pxor	%%mm3, %%mm5		# + mm5 = t08 = (in4 | in6) ^ in7;
	movq	%%mm7, %%mm3		# mm3 = t06
	movl	4*2(%%ecx),%%eax
	movq	%%mm2, (%%eax)		# out2 = in2
	movq	8*6(%%edx),%%mm6	# mm6 = in6
	por	%%mm5, %%mm7		# mm7 = t08 | t06
	pand	%%mm4, %%mm6		# mm6 = in4 & in6
	pxor	%%mm4, %%mm7		# + mm7 = t10 = (t08 | t06) ^ in4
	pxor	%%mm7, %%mm5		# mm5 = t10 ^ t08
	pxor	%%mm7, %%mm0		#### mm0 = in0 ^= t10
					# -- mm7 free
	movq	8*5(%%edx),%%mm7	# mm7 = in5
	pxor	%4,%%mm5		# mm5 = ~(t10 ^ t08)
	por	%%mm4, %%mm7		# mm7 = in4 | in5
	movl	(%%ecx),%%eax
	movq	%%mm0, (%%eax)		# out0 = in0
	pxor	%%mm6, %%mm7		# + mm7 = t13 = (in4 & in6) ^ (in4 | in5)
					# -- mm6 free
	movq	%%mm7, %%mm6		# mm6 = t13
	por	%%mm3, %%mm7		# mm7 = t13 | t06
					# -- mm3 free
	pxor	%4,%%mm6		# mm6 = ~t13
	pxor	%%mm7, %%mm5		# mm5 = ~(t10 ^ t08) ^ (t13 | t06)
					# -- mm7 free
	pxor	8*3(%%edx),%%mm6	#### mm6 = out3 = in3 ^ ~t13
	pxor	%%mm5, %%mm1		#### mm1 = out1 = (in1 ^= ~(t10 ^ t08) ^ (t13 | t06))
					# -- mm5 free

  ## // csc_transF( in3, in2, in1, in0,	 // in
  ## //             in7, in6, in5, in4 );// xor-out
  ## {
  ## ulong t04 = ~in3;           ulong t06 = in2 | in3;
  ## out4 = in4 ^ (t04 | in0);   ulong t07 = t06 ^ t04;
  ## ulong t08 = t07 ^ in1;      out7 = in7 ^ t07;	   
  ## ulong t09 = t08 | in2;
  ## out6 = in6 ^ t09;           out5 = in5 ^ ((t09 ^ in0) | in1);
  ## }

	movl	4*3(%%ecx),%%eax
	movq	%%mm6, %%mm3		# mm3 = in3
	movq	%%mm6, (%%eax)
	pxor	%4, %%mm6		# mm6 = t04 = ~in3
	por	%%mm2, %%mm3		# mm3 = t06 = in2 | in3
	movl	4*1(%%ecx),%%eax
	movq	%%mm6, %%mm7		# mm7 = t04
	movq	%%mm1, (%%eax)
	por	%%mm0, %%mm6		# mm6 = t04 | in0
	pxor	%%mm7, %%mm3		# mm3 = t07 = t06 ^ t04
					# -- mm7 free
	movq	8*7(%%edx), %%mm7	# mm7 = in7
	pxor	%%mm4, %%mm6		### mm6 = out4 = in4 ^ (t04 | in0)
					# -- mm4 free
	movl	4*4(%%ecx),%%eax
	movq	%%mm3, %%mm4		# mm4 = t07
	movq	8*6(%%edx), %%mm5	# mm5 = in6
	pxor	%%mm1, %%mm3		# mm3 = t08 = t07 ^ in1
	movq	%%mm6, (%%eax)		# -- mm6 free
	pxor	%%mm4, %%mm7		### mm7 = out7 = in7 ^ t07
					# -- mm4 free
	por	%%mm2, %%mm3		# mm3 = t09 = t08 | in2
					# -- mm2 free
	movl	4*7(%%ecx),%%eax
	movq	%%mm7, (%%eax)		# -- mm7 free
	movq	%%mm3, %%mm4		# mm4 = t09
	movq	8*5(%%edx), %%mm2	# mm2 = in5
	pxor	%%mm0, %%mm4		# mm4 = t09 ^ in0
	pxor	%%mm5, %%mm3		### mm3 = out6 = in6 ^ t09
	por	%%mm1, %%mm4		# mm4 = (t09 ^ in0) | in1
	movl	4*6(%%ecx),%%eax
	movq	%%mm3, (%%eax)		# -- mm3 free
	pxor	%%mm2, %%mm4		### mm4 = out5 = in5 ^ ((t09 ^ in0) | in1)
	movl	4*5(%%ecx),%%eax
	movq	%%mm4, (%%eax)

#	popl	%%edx
# ----------------------------------------------------
	popl	%%eax

	# in7 = x6 ^ y7;
	# in6 = x5y6;
	leal	8*("#adr"+0)(%%eax), %%esi
	movq	8*(3+8)(%%edi), %%mm7
	movl	%%esi, 8*8+4*0(%%edx)
	movq	8*(5+0)(%%edi), %%mm6
	addl	$8, %%esi
	pxor	8*(6+0)(%%edi), %%mm7
	movl	%%esi, 8*8+4*1(%%edx)

	# in5 = x4 ^ y5;
	# in4 = x3y4;
	addl	$8, %%esi
	movq	8*(2+8)(%%edi), %%mm5
	movl	%%esi, 8*8+4*2(%%edx)
	movq	8*(3+0)(%%edi), %%mm4
	addl	$8, %%esi
	pxor	8*(4+0)(%%edi), %%mm5
	movl	%%esi, 8*8+4*3(%%edx)

	# in3 = x2 ^ y3;
	# in2 = x1y2;
	addl	$8, %%esi
	movq	8*(1+8)(%%edi), %%mm3
	movl	%%esi, 8*8+4*4(%%edx)
	movq	8*(1+0)(%%edi), %%mm2
	addl	$8, %%esi
	pxor	8*(2+0)(%%edi), %%mm3
	movl	%%esi, 8*8+4*5(%%edx)

	# in1 = x0 ^ y1;
	# in0 = x7y0;
	addl	$8, %%esi
	movq	8*(0+8)(%%edi), %%mm1
	movl	%%esi, 8*8+4*6(%%edx)
	movq	8*(7+0)(%%edi), %%mm0
	addl	$8, %%esi
	pxor	8*(0+0)(%%edi), %%mm1
	movl	%%esi, 8*8+4*7(%%edx)

	pushl	%%eax
	pushl	%%edx
	call	csc_transP2
	popl	%%edx
	popl	%%eax

	addl	$16*8, %%ebx	# tep += 16;

  " : "+b"(tep)							\
    : "D"(x_y_xy), "a"(cfr), "d"(csc_params), "m"(mmNOT)	\
    : "%ecx", "%esi"						\
  );								\
  } while( 0 )


  // global initializations
  memcpy( &(*subkey)[0], &key[1], sizeof((*subkey)[0]) );
  memcpy( &(*subkey)[1], &key[0], sizeof((*subkey)[1]) );
  int hs = 0;

  // cache initialization
  (*subkey)[2][56] = (*subkey)[2][48] = 
  (*subkey)[2][40] = (*subkey)[2][32] = (*subkey)[2][24] = _0;
  (*subkey)[2][16] = (*subkey)[2][ 8] = (*subkey)[2][ 0] = _1;
  tcp = &csc_tabc[0][8];
  skp = &(*subkey)[2][1];
  skp1 = &(*subkey)[1][8];
  {
  /*for( int n=7; n; n--,tcp+=8,skp1+=8,skp++ )
    csc_transP_call( 
            skp1[7] ^ tcp[7], skp1[6] ^ tcp[6], skp1[5] ^ tcp[5], skp1[4] ^ tcp[4],
	    skp1[3] ^ tcp[3], skp1[2] ^ tcp[2], skp1[1] ^ tcp[1], skp1[0] ^ tcp[0],
	    skp[56], skp[48], skp[40], skp[32], skp[24], skp[16], skp[ 8], skp[ 0] );
  */
      asm volatile ("
	# eax == skp1
	# ebx == tcp
	# edi == skp
	movl	$7, %%esi

.balign 4
.loop0:
	movq	7*8(%%eax), %%mm7
	movl	%%edi, 0*4+8*8(%%edx)
	pxor	7*8(%%ebx), %%mm7
	addl	$8*8, %%edi
	movl	%%edi, 1*4+8*8(%%edx)

	movq	6*8(%%eax), %%mm6
	addl	$8*8, %%edi
	pxor	6*8(%%ebx), %%mm6
	movl	%%edi, 2*4+8*8(%%edx)

	movq	5*8(%%eax), %%mm5
	addl	$8*8, %%edi
	pxor	5*8(%%ebx), %%mm5
	movl	%%edi, 3*4+8*8(%%edx)

	movq	4*8(%%eax), %%mm4
	addl	$8*8, %%edi
	pxor	4*8(%%ebx), %%mm4
	movl	%%edi, 4*4+8*8(%%edx)

	movq	3*8(%%eax), %%mm3
	addl	$8*8, %%edi
	pxor	3*8(%%ebx), %%mm3
	movl	%%edi, 5*4+8*8(%%edx)

	movq	2*8(%%eax), %%mm2
	addl	$8*8, %%edi
	pxor	2*8(%%ebx), %%mm2
	movl	%%edi, 6*4+8*8(%%edx)

	movq	1*8(%%eax), %%mm1
	addl	$8*8, %%edi
	pxor	1*8(%%ebx), %%mm1
	movl	%%edi, 7*4+8*8(%%edx)

	movq	0*8(%%eax), %%mm0
	subl	$(7*8)*8, %%edi
	pxor	0*8(%%ebx), %%mm0

	pushl	%%eax
	pushl	%%edx
	call	csc_transP2
	popl	%%edx
	popl	%%eax

	addl	$8*8, %%eax
	addl	$8*8, %%ebx
	addl	$1*8, %%edi

	decl	%%esi
	jg	.loop0

      " : "+D"(skp), "+a"(skp1), "+b"(tcp)
        : "d"(csc_params)
        : "%ecx", "%esi"
      );
  }

  // bit 0 : average of 11.41 bits to twiddle
  // bit 1 : average of 11.12 bits to twiddle
  // bit 2 : average of 11.41 bits to twiddle
  // bit 3 : average of 12.16 bits to twiddle
  // bit 4 : average of 12.44 bits to twiddle
  // bit 5 : average of 12.06 bits to twiddle
  // bit 6 : average of 10.00 bits to twiddle
  // bit 7 : average of 10.94 bits to twiddle
  for( int i=2; i<8; i++ ) {
    u8 x = keyB[7-i] ^ csc_tabp[7-i]; x = csc_tabp[x] ^ csc_tabp[x ^ (1<<6)];
    int ntt = 0;
    (*totwiddle)[i-2][ntt++] = &(*subkey)[1][i*8+6];
    for( int j=0; j<8; j++ ) {
      if( x & (1<<j) ) {
	unsigned n = j*8+i;
	(*totwiddle)[i-2][ntt++] = &(*subkey)[2][n];
	(*totwiddle)[i-2][ntt++] = &(*tp1)[n/16][n & 7];
	if( (n & 15) <= 7 )
	  (*totwiddle)[i-2][ntt++] = &(*tp2)[n/16][n & 15];
	else {
	  (*totwiddle)[i-2][ntt++] = &(*tp2)[n/16][(n+1) & 7];
	  if( n & 1 )
	    (*totwiddle)[i-2][ntt++] = &(*tp1)[n/16][(n+1) & 7];
	}
      }
    }
    (*totwiddle)[i-2][ntt] = NULL;
  }

  skp = &(*subkey)[2][0];
  {
  for( int n=0; n<4; n++ ) {
    ulong x0,x1,x2,x3,x4,x5,x6,x7, y1,y3,y5,y7;

    x0 = msg[n*16+8+0] ^ skp[0+8]; x1 = msg[n*16+8+1] ^ skp[1+8];
    x2 = msg[n*16+8+2] ^ skp[2+8]; x3 = msg[n*16+8+3] ^ skp[3+8];
    x4 = msg[n*16+8+4] ^ skp[4+8]; x5 = msg[n*16+8+5] ^ skp[5+8];
    x6 = msg[n*16+8+6] ^ skp[6+8]; x7 = msg[n*16+8+7] ^ skp[7+8];

    (*tp1)[n][7] = x7 ^ (y7 = msg[n*16+7] ^ skp[7]);
    (*tp1)[n][6] = x6 ^ ((*tp2)[n][6] = x5 ^ msg[n*16+6] ^ skp[6]);
    (*tp1)[n][5] = x5 ^ (y5 = msg[n*16+5] ^ skp[5]);
    (*tp1)[n][4] = x4 ^ ((*tp2)[n][4] = x3 ^ msg[n*16+4] ^ skp[4]);
    (*tp1)[n][3] = x3 ^ (y3 = msg[n*16+3] ^ skp[3]);
    (*tp1)[n][2] = x2 ^ ((*tp2)[n][2] = x1 ^ msg[n*16+2] ^ skp[2]);
    (*tp1)[n][1] = x1 ^ (y1 = msg[n*16+1] ^ skp[1]);
    (*tp1)[n][0] = x0 ^ ((*tp2)[n][0] = x7 ^ msg[n*16+0] ^ skp[0]);

    (*tp2)[n][7] = x6 ^ y7;
    (*tp2)[n][5] = x4 ^ y5;
    (*tp2)[n][3] = x2 ^ y3;
    (*tp2)[n][1] = x0 ^ y1;

    skp += 16;
  }
  }

  for( ;; ) {

    //extern void printkey( ulong key[64], int n, bool tab );
    //printkey( subkey[1], 17, 1 );

    // local initializations
    memcpy( cfr, msg, sizeof(*cfr) );

    // ROUND 1
    APPLY_MP0(  0,  8);
    APPLY_MP0( 16, 24);
    APPLY_MP0( 32, 40);
    APPLY_MP0( 48, 56);

    tep = &csc_tabe[0][0];
    APPLY_Me(  0, 16);
    APPLY_Me( 32, 48);
    APPLY_Me(  8, 24);
    APPLY_Me( 40, 56);
    APPLY_Me(  0, 32);
    APPLY_Me(  8, 40);
    APPLY_Me( 16, 48);
    APPLY_Me( 24, 56);

    // ROUNDS 2..8
    skp = &(*subkey)[3][0];
    skp1 = &(*subkey)[2][0];
    tcp = &csc_tabc[1][0];
    for( int sk=7; sk; sk-- ) {
      /*for( int n=8; n; n--,tcp+=8,skp1+=8,skp++ )
	csc_transP_call( 
	        skp1[7] ^ tcp[7], skp1[6] ^ tcp[6], skp1[5] ^ tcp[5], skp1[4] ^ tcp[4],
		skp1[3] ^ tcp[3], skp1[2] ^ tcp[2], skp1[1] ^ tcp[1], skp1[0] ^ tcp[0],
		skp[56], skp[48], skp[40], skp[32], skp[24], skp[16], skp[ 8], skp[ 0] );
      skp -= 8;
      */
      asm volatile ("
	# eax == skp1
	# ebx == tcp
	# edi == skp
	movl	$8, %%esi

.balign 4
.loop:
	movq	7*8(%%eax), %%mm7
	movl	%%edi, 0*4+8*8(%%edx)
	pxor	7*8(%%ebx), %%mm7
	addl	$8*8, %%edi
	movl	%%edi, 1*4+8*8(%%edx)

	movq	6*8(%%eax), %%mm6
	addl	$8*8, %%edi
	pxor	6*8(%%ebx), %%mm6
	movl	%%edi, 2*4+8*8(%%edx)

	movq	5*8(%%eax), %%mm5
	addl	$8*8, %%edi
	pxor	5*8(%%ebx), %%mm5
	movl	%%edi, 3*4+8*8(%%edx)

	movq	4*8(%%eax), %%mm4
	addl	$8*8, %%edi
	pxor	4*8(%%ebx), %%mm4
	movl	%%edi, 4*4+8*8(%%edx)

	movq	3*8(%%eax), %%mm3
	addl	$8*8, %%edi
	pxor	3*8(%%ebx), %%mm3
	movl	%%edi, 5*4+8*8(%%edx)

	movq	2*8(%%eax), %%mm2
	addl	$8*8, %%edi
	pxor	2*8(%%ebx), %%mm2
	movl	%%edi, 6*4+8*8(%%edx)

	movq	1*8(%%eax), %%mm1
	addl	$8*8, %%edi
	pxor	1*8(%%ebx), %%mm1
	movl	%%edi, 7*4+8*8(%%edx)

	movq	0*8(%%eax), %%mm0
	subl	$(7*8)*8, %%edi
	pxor	0*8(%%ebx), %%mm0

	pushl	%%eax
# ----------------------------------------------------
#	pushl	%%edx
#	call	csc_transP2
	
	leal	64(%%edx), %%ecx

  ## //csc_transF( in3, in2, in1, in0,	// in
  ## //            in7, in6, in5, in4 );// xor-out
  ## {
  ## ulong t04 = ~in3;		    ulong t06 =  in2 | in3;
  ## in4      ^=  t04 | in0;	    ulong t07 =  t06 ^ t04;
  ## ulong t09 =  (t07 ^ in1) | in2;   in7      ^=  t07;
  ## in6      ^=  t09;                 in5      ^=  (t09 ^ in0) | in1;
  ## }
	movq	%%mm0, "_in0"

	movq	%%mm7, "_in7"
	movq	%%mm3, %%mm7	# mm7 = in3
	movq	%%mm3, "_in3"
	pxor	"_mmNOT", %%mm3	# mm3 = t04 = ~in3
	movq	%%mm6, "_in6"
	movq	%%mm3, %%mm6	# mm6 = t04
	por	%%mm2, %%mm7	# mm7 = t06 = in2 | in3
	por	%%mm0, %%mm3	# mm3 = t04 | in0
	movq	%%mm5, "_in5"
	pxor	%%mm6, %%mm7	# mm7 = t07 = t06 ^ to4
	pxor	%%mm3, %%mm4	# mm4 = in4 ^= t04 | in0
	movq	%%mm7, %%mm5	# mm5 = t07
	pxor	%%mm1, %%mm5	# mm5 = t07 ^ in1
	pxor	"_in7", %%mm7	# mm7 = in7 ^= t07
	por	%%mm2, %%mm5	# mm5 = t09 = (t07 ^ in1) | in2
	movq	%%mm5, %%mm6	# mm6 = t09
	pxor	"_in0", %%mm6	# mm6 = t09 ^ in0
	#pxor	%%mm0, %%mm6
	pxor	"_in6", %%mm5	# mm5 = in6 ^= t09
	por	%%mm1, %%mm6	# mm6 = (t09 ^ in0) | in1
	pxor	"_in5", %%mm6	# mm6 = in5 ^= (t09 ^ in0) | in1
	
  ## // csc_transG( in7, in6, in5, in4,  // in
  ## //             in3, in2, in1, in0 );// xor-out
  ## {
  ## ulong t06 = (in4 & in5) ^ (in4 | in7);   ulong t08 = (in4 | in6) ^ in7;
  ## out2 = (in2 ^= t06);                     ulong t10 = (t08 | t06) ^ in4;
  ## ulong t13 = (in4 & in6) ^ (in4 | in5);   out0 = (in0 ^= t10);
  ## out3 = (in3 ^= ~t13);                    out1 = (in1 ^= ~(t10 ^ t08) ^ (t13 | t06));
  ## }

	movq	%%mm6, "_in5"	# -- mm6 free
	pand	%%mm4, %%mm6	# mm6 = in4 & in5
	movq	%%mm7, "_in7"	# -- mm7 free
	movq	%%mm7, %%mm3	# mm3 = in7
	por	%%mm4, %%mm7	# mm7 = in4 | in7
	movq	%%mm5, "_in6"	# mm5 free
	pxor	%%mm6, %%mm7	# + mm7 = t06 = (in4 & in5) ^ (in4 | in7)
				# -- mm6 free
	por	%%mm4, %%mm5	# mm5 = in4 | in6
	pxor	%%mm7, %%mm2	#### mm2 = out2 = in2 ^= t06
	pxor	%%mm3, %%mm5	# + mm5 = t08 = (in4 | in6) ^ in7;
	movq	%%mm7, %%mm3	# mm3 = t06
	movl	"_out2",%%eax
	movq	%%mm2, (%%eax)	# out2 = in2
	movq	"_in6",%%mm6	# mm6 = in6
	por	%%mm5, %%mm7	# mm7 = t08 | t06
	pand	%%mm4, %%mm6	# mm6 = in4 & in6
	pxor	%%mm4, %%mm7	# + mm7 = t10 = (t08 | t06) ^ in4
	pxor	%%mm7, %%mm5	# mm5 = t10 ^ t08
	pxor	%%mm7, %%mm0	#### mm0 = in0 ^= t10
				# -- mm7 free
	movq	"_in5",%%mm7	# mm7 = in5
	pxor	"_mmNOT",%%mm5	# mm5 = ~(t10 ^ t08)
	por	%%mm4, %%mm7	# mm7 = in4 | in5
	movl	"_out0",%%eax
	movq	%%mm0, (%%eax)	# out0 = in0
	pxor	%%mm6, %%mm7	# + mm7 = t13 = (in4 & in6) ^ (in4 | in5)
				# -- mm6 free
	movq	%%mm7, %%mm6	# mm6 = t13
	por	%%mm3, %%mm7	# mm7 = t13 | t06
				# -- mm3 free
	pxor	"_mmNOT",%%mm6	# mm6 = ~t13
	pxor	%%mm7, %%mm5	# mm5 = ~(t10 ^ t08) ^ (t13 | t06)
				# -- mm7 free
	pxor	"_in3",%%mm6	#### mm6 = out3 = in3 ^ ~t13
	pxor	%%mm5, %%mm1	#### mm1 = out1 = (in1 ^= ~(t10 ^ t08) ^ (t13 | t06))
				# -- mm5 free

  ## // csc_transF( in3, in2, in1, in0,	 // in
  ## //             in7, in6, in5, in4 );// xor-out
  ## {
  ## ulong t04 = ~in3;           ulong t06 = in2 | in3;
  ## out4 = in4 ^ (t04 | in0);   ulong t07 = t06 ^ t04;
  ## ulong t08 = t07 ^ in1;      out7 = in7 ^ t07;	   
  ## ulong t09 = t08 | in2;
  ## out6 = in6 ^ t09;           out5 = in5 ^ ((t09 ^ in0) | in1);
  ## }

	movl	"_out3",%%eax
	movq	%%mm6, %%mm3	# mm3 = in3
	movq	%%mm6, (%%eax)
	pxor	"_mmNOT", %%mm6	# mm6 = t04 = ~in3
	por	%%mm2, %%mm3	# mm3 = t06 = in2 | in3
	movl	"_out1",%%eax
	movq	%%mm6, %%mm7	# mm7 = t04
	movq	%%mm1, (%%eax)
	por	%%mm0, %%mm6	# mm6 = t04 | in0
	pxor	%%mm7, %%mm3	# mm3 = t07 = t06 ^ t04
				# -- mm7 free
	movq	"_in7", %%mm7	# mm7 = in7
	pxor	%%mm4, %%mm6	### mm6 = out4 = in4 ^ (t04 | in0)
				# -- mm4 free
	movl	"_out4",%%eax
	movq	%%mm3, %%mm4	# mm4 = t07
	movq	"_in6", %%mm5	# mm5 = in6
	pxor	%%mm1, %%mm3	# mm3 = t08 = t07 ^ in1
	movq	%%mm6, (%%eax)	# -- mm6 free
	pxor	%%mm4, %%mm7	### mm7 = out7 = in7 ^ t07
				# -- mm4 free
	por	%%mm2, %%mm3	# mm3 = t09 = t08 | in2
				# -- mm2 free
	movl	"_out7",%%eax
	movq	%%mm7, (%%eax)	# -- mm7 free
	movq	%%mm3, %%mm4	# mm4 = t09
	movq	"_in5", %%mm2	# mm2 = in5
	pxor	%%mm0, %%mm4	# mm4 = t09 ^ in0
	pxor	%%mm5, %%mm3	### mm3 = out6 = in6 ^ t09
	por	%%mm1, %%mm4	# mm4 = (t09 ^ in0) | in1
	movl	"_out6",%%eax
	movq	%%mm3, (%%eax)	# -- mm3 free
	pxor	%%mm2, %%mm4	### mm4 = out5 = in5 ^ ((t09 ^ in0) | in1)
	movl	"_out5",%%eax
	movq	%%mm4, (%%eax)

#	popl	%%edx
# ----------------------------------------------------
	popl	%%eax

	addl	$8*8, %%eax
	addl	$8*8, %%ebx
	addl	$1*8, %%edi

	decl	%%esi
	jg	.loop

	subl	$8*8, %%edi	# skp -= 8;

      " : "+D"(skp), "+a"(skp1), "+b"(tcp)
	: "d"(csc_params), "m"(mmNOT)
	: "%ecx", "%esi"
      );

      APPLY_Ms(  0,  8);
      APPLY_Ms( 16, 24);
      APPLY_Ms( 32, 40);
      APPLY_Ms( 48, 56);

      tep = &csc_tabe[0][0];
      APPLY_Me(  0, 16);
      APPLY_Me( 32, 48);
      APPLY_Me(  8, 24);
      APPLY_Me( 40, 56);
      APPLY_Me(  0, 32);
      APPLY_Me(  8, 40);
      APPLY_Me( 16, 48);
      APPLY_Me( 24, 56);
    }

    // ROUND 9
    ulong result = _1;
    {
    /*for( int n=0; n<8; n++,tcp+=8,skp1+=8,skp++ ) {
      csc_transP_call( skp1[7] ^ tcp[7], skp1[6] ^ tcp[6], skp1[5] ^ tcp[5], skp1[4] ^ tcp[4],
		       skp1[3] ^ tcp[3], skp1[2] ^ tcp[2], skp1[1] ^ tcp[1], skp1[0] ^ tcp[0],
		       skp[56], skp[48], skp[40], skp[32], skp[24], skp[16], skp[ 8], skp[ 0] );

      result &= ~(cipher[56+n] ^ (*cfr)[56+n] ^ skp[56] ^ skp[56-128]); if( !result ) goto stepper;
      result &= ~(cipher[48+n] ^ (*cfr)[48+n] ^ skp[48] ^ skp[48-128]); if( !result ) goto stepper;
      result &= ~(cipher[40+n] ^ (*cfr)[40+n] ^ skp[40] ^ skp[40-128]); if( !result ) goto stepper;
      result &= ~(cipher[32+n] ^ (*cfr)[32+n] ^ skp[32] ^ skp[32-128]); if( !result ) goto stepper;
      result &= ~(cipher[24+n] ^ (*cfr)[24+n] ^ skp[24] ^ skp[24-128]); if( !result ) goto stepper;
      result &= ~(cipher[16+n] ^ (*cfr)[16+n] ^ skp[16] ^ skp[16-128]); if( !result ) goto stepper;
      result &= ~(cipher[ 8+n] ^ (*cfr)[ 8+n] ^ skp[ 8] ^ skp[ 8-128]); if( !result ) goto stepper;
      result &= ~(cipher[ 0+n] ^ (*cfr)[ 0+n] ^ skp[ 0] ^ skp[ 0-128]); if( !result ) goto stepper;
    }*/
    asm volatile ("
	# eax == skp1
	# ebx == tcp
	# ecx == skp
	movl	$0, %%esi

.balign 4
loop2:
	movq	7*8(%%eax), %%mm7
	movl	%%ecx, 0*4+8*8(%%edx)
	pxor	7*8(%%ebx), %%mm7
	addl	$8*8, %%ecx
	movl	%%ecx, 1*4+8*8(%%edx)

	movq	6*8(%%eax), %%mm6
	addl	$8*8, %%ecx
	pxor	6*8(%%ebx), %%mm6
	movl	%%ecx, 2*4+8*8(%%edx)

	movq	5*8(%%eax), %%mm5
	addl	$8*8, %%ecx
	pxor	5*8(%%ebx), %%mm5
	movl	%%ecx, 3*4+8*8(%%edx)

	movq	4*8(%%eax), %%mm4
	addl	$8*8, %%ecx
	pxor	4*8(%%ebx), %%mm4
	movl	%%ecx, 4*4+8*8(%%edx)

	movq	3*8(%%eax), %%mm3
	addl	$8*8, %%ecx
	pxor	3*8(%%ebx), %%mm3
	movl	%%ecx, 5*4+8*8(%%edx)

	movq	2*8(%%eax), %%mm2
	addl	$8*8, %%ecx
	pxor	2*8(%%ebx), %%mm2
	movl	%%ecx, 6*4+8*8(%%edx)

	movq	1*8(%%eax), %%mm1
	addl	$8*8, %%ecx
	pxor	1*8(%%ebx), %%mm1
	movl	%%ecx, 7*4+8*8(%%edx)

	movq	0*8(%%eax), %%mm0
	subl	$(7*8)*8, %%ecx
	pxor	0*8(%%ebx), %%mm0

	pushl	%%eax
	pushl	%%ecx
	pushl	%%edx
	call	csc_transP2
	popl	%%edx
	popl	%%ecx
	popl	%%eax

	movq	%3, %%mm0	# result
	movl	%4, %%edi	# cipher
	movl	%5, %%ebp	# cfr

	pushl	%%eax
	pushl	%%ebx

	# disp32 + index
	# disp32 + scale * index
	# base + index
	# base + scale * index
	# (%%esi)

	leal	(,%%esi,8), %%eax
	addl	%%eax, %%edi
	addl	%%eax, %%ebp

     # result &= ~(cipher[ 0+n] ^ (*cfr)[ 0+n] ^ skp[ 0] ^ skp[ 0-128]); if( !result ) goto stepper;
     # result &= ~(cipher[ 8+n] ^ (*cfr)[ 8+n] ^ skp[ 8] ^ skp[ 8-128]); if( !result ) goto stepper;
	movq	8*8(%%edi), %%mm1
	movq	   (%%edi), %%mm2
	pxor	8*8(%%ebp), %%mm1
	pxor	   (%%ebp), %%mm2
	pxor	8*8(%%ecx), %%mm1
	pxor	   (%%ecx), %%mm2
	pxor	8*(8-128)(%%ecx), %%mm1
	pxor	8*(0-128)(%%ecx), %%mm2
	pandn	%%mm0, %%mm1
	addl	$8*16, %%edi
	pandn	%%mm1, %%mm2
	addl	$8*16, %%ebp
	movq	%%mm2, %%mm3
	movq	%%mm2, %%mm0
	punpckhdq	%%mm3, %%mm3
	addl	$8*16, %%ecx
	movd	%%mm2, %%eax
	movd	%%mm3, %%ebx
	cmpl	$0, %%eax
	jne	.next_test1
	cmpl	$0, %%ebx
	je	.goto_stepper
.next_test1:

     # result &= ~(cipher[16+n] ^ (*cfr)[16+n] ^ skp[16] ^ skp[16-128]); if( !result ) goto stepper;
     # result &= ~(cipher[24+n] ^ (*cfr)[24+n] ^ skp[24] ^ skp[24-128]); if( !result ) goto stepper;
	movq	8*8(%%edi), %%mm1
	movq	   (%%edi), %%mm2
	pxor	8*8(%%ebp), %%mm1
	pxor	   (%%ebp), %%mm2
	pxor	8*8(%%ecx), %%mm1
	pxor	   (%%ecx), %%mm2
	pxor	8*(8-128)(%%ecx), %%mm1
	pxor	8*(0-128)(%%ecx), %%mm2
	pandn	%%mm0, %%mm1
	addl	$8*16, %%edi
	pandn	%%mm1, %%mm2
	addl	$8*16, %%ebp
	movq	%%mm2, %%mm3
	movq	%%mm2, %%mm0
	punpckhdq	%%mm3, %%mm3
	addl	$8*16, %%ecx
	movd	%%mm2, %%eax
	movd	%%mm3, %%ebx
	cmpl	$0, %%eax
	jne	.next_test2
	cmpl	$0, %%ebx
	je	.goto_stepper
.next_test2:

     # result &= ~(cipher[32+n] ^ (*cfr)[32+n] ^ skp[32] ^ skp[32-128]); if( !result ) goto stepper;
     # result &= ~(cipher[40+n] ^ (*cfr)[40+n] ^ skp[40] ^ skp[40-128]); if( !result ) goto stepper;
	movq	8*8(%%edi), %%mm1
	movq	   (%%edi), %%mm2
	pxor	8*8(%%ebp), %%mm1
	pxor	   (%%ebp), %%mm2
	pxor	8*8(%%ecx), %%mm1
	pxor	   (%%ecx), %%mm2
	pxor	8*(8-128)(%%ecx), %%mm1
	pxor	8*(0-128)(%%ecx), %%mm2
	pandn	%%mm0, %%mm1
	addl	$8*16, %%edi
	pandn	%%mm1, %%mm2
	addl	$8*16, %%ebp
	movq	%%mm2, %%mm3
	movq	%%mm2, %%mm0
	punpckhdq	%%mm3, %%mm3
	addl	$8*16, %%ecx
	movd	%%mm2, %%eax
	movd	%%mm3, %%ebx
	cmpl	$0, %%eax
	jne	.next_test3
	cmpl	$0, %%ebx
	je	.goto_stepper
.next_test3:

     # result &= ~(cipher[48+n] ^ (*cfr)[48+n] ^ skp[48] ^ skp[48-128]); if( !result ) goto stepper;
     # result &= ~(cipher[56+n] ^ (*cfr)[56+n] ^ skp[56] ^ skp[56-128]); if( !result ) goto stepper;
	movq	8*8(%%edi), %%mm1
	movq	   (%%edi), %%mm2
	pxor	8*8(%%ebp), %%mm1
	pxor	   (%%ebp), %%mm2
	pxor	8*8(%%ecx), %%mm1
	pxor	   (%%ecx), %%mm2
	pxor	8*(8-128)(%%ecx), %%mm1
	pxor	8*(0-128)(%%ecx), %%mm2
	pandn	%%mm0, %%mm1
	pandn	%%mm1, %%mm2
	movq	%%mm2, %%mm3
	movq	%%mm2, %%mm0
	punpckhdq	%%mm3, %%mm3
	movd	%%mm2, %%eax
	movd	%%mm3, %%ebx
	cmpl	$0, %%eax
	jne	.next2
	cmpl	$0, %%ebx
	jne	.next2

.balign 4
.goto_stepper:
	add	$8, %%esp
	jmp	.stepper
	
.next2:
	popl	%%ebx
	popl	%%eax
	movq	%%mm0, %3

	addl	$8*8, %%eax
	addl	$8*8, %%ebx
	subl	$8*(16*3-1), %%ecx

	incl	%%esi
	cmpl	$8, %%esi
	jl	loop2

      " : "+c"(skp), "+a"(skp1), "+b"(tcp), "=m"(result)
	: "m"(cipher), "m"(cfr), "d"(csc_params)
	: "%esi", "%edi", "%ebp"
      );
    }
    memcpy( &key[0], &(*subkey)[1], sizeof(key[0]) );
  //memcpy( &key[1], &(*subkey)[0], sizeof(key[1]) );
    if( result == 0xdeedbeef ) goto stepper;
    return result;

  stepper:
    asm volatile ("
.balign 4
	.stepper:
    ");
    // increment the key in gray order
    hs++;
    // bits 6
    if( hs & (1 << 0) ) { //(*subkey)[1][22] = ~subkey[1][22];
      for( ulong **p = &(*totwiddle)[0][0]; *p; p++ ) **p ^= _1;
      continue;
    }
    if( hs & (1 << 1) ) { //(*subkey)[1][30] = ~(*subkey)[1][30];
      for( ulong **p = &(*totwiddle)[1][0]; *p; p++ ) **p ^= _1;
      continue;
    }
    if( hs & (1 << 2) ) { //(*subkey)[1][38] = ~(*subkey)[1][38];
      for( ulong **p = &(*totwiddle)[2][0]; *p; p++ ) **p ^= _1;
      continue;
    }
    if( hs & (1 << 3) ) { //(*subkey)[1][46] = ~(*subkey)[1][46];
      for( ulong **p = &(*totwiddle)[3][0]; *p; p++ ) **p ^= _1;
      continue;
    }
    if( hs & (1 << 4) ) { //(*subkey)[1][54] = ~(*subkey)[1][54];
      for( ulong **p = &(*totwiddle)[4][0]; *p; p++ ) **p ^= _1;
      continue;
    }
    if( hs & (1 << 5) ) { //(*subkey)[1][62] = ~(*subkey)[1][62];
      for( ulong **p = &(*totwiddle)[5][0]; *p; p++ ) **p ^= _1;
      continue;
    }
    break;
  }

  return 0;
}

#undef _in0
#undef _in1
#undef _in2
#undef _in3
#undef _in4
#undef _in5
#undef _in6
#undef _in7
#undef _out0
#undef _out1
#undef _out2
#undef _out3
#undef _out4
#undef _out5
#undef _out6
#undef _out7
#undef _mmNOT
