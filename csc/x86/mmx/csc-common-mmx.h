// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-common-mmx.h,v $
// Revision 1.1.2.1  1999/12/12 11:06:00  remi
// Moved from directory csc/x86/
//
// Revision 1.1.2.5  1999/12/11 00:34:13  cyp
// made mmx cores not collide with normal cores
//
// Revision 1.1.2.4  1999/12/06 11:21:06  remi
// Moved csc_transP2() from csc-common-mmx.h to csc-common-mmx.cpp.
//
// Revision 1.1.2.3  1999/12/05 14:39:43  remi
// A faster 6bit MMX core.
//
// Revision 1.1.2.2  1999/11/23 23:39:45  remi
// csc_transP() optimized.
// modified csc_transP() calling convention.
//
// Revision 1.1.2.1  1999/11/22 18:58:12  remi
// Initial commit of MMX'fied CSC cores.

#ifndef __CSC_COMMON_H
#define __CSC_COMMON_H "@(#)$Id: csc-common-mmx.h,v 1.1.2.1 1999/12/12 11:06:00 remi Exp $"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "cputypes.h" /* u32, s32 */

#define csc_tabc csc_mmx_tabc
#define csc_tabe csc_mmx_tabe
#define csc_tabp csc_mmx_tabp
#define csc_transP csc_mmx_transP

#define _FORCE64_

#if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
  #define _FORCE64_
#endif  

#if defined(_FORCE64_)
  #define CSC_BIT_64
  #if defined(__GNUC__)
    typedef unsigned long long ulong;
    #define CASTNUM64(n) n##ull
  #elif ( defined(_MSC_VER) /*>=11*/ || defined(__WATCOMC__) /*>=10*/)
    typedef unsigned __int64 ulong;
    #define CASTNUM64(n) n##ul
  #else
    #error something missing
  #endif
#elif (ULONG_MAX == 0xfffffffful)
  #define CSC_BIT_32
  typedef unsigned long ulong;  
#elif (ULONG_MAX == 0xfffffffffffffffful)
  #define CSC_BIT_64
  typedef unsigned long ulong;  
  #define CASTNUM64(n) n##ul
#else
  #error define either CSC_BIT_32 or CSC_BIT_64
#endif

#define _0 ( (ulong)0)
#define _1 (~(_0))

#define PASTE( xx, yy ) PASTE1( xx, yy )
#define PASTE1( xx, yy ) xx##yy

// ------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif
// bitslice version of c0..c8
extern const ulong csc_tabc[9][64];

// bitslice version of e and e'
extern const ulong csc_tabe[2][64];

// table-lookup implementation of csc_transP()
extern const u8 csc_tabp[256];

extern const ulong mmNOT;

#ifdef __cplusplus
}
#endif

#if 0
// just for reference, these functions aren't used anymore
// ------------------------------------------------------------------
inline void csc_transF( ulong t00, ulong t01, ulong t02, ulong t03,
		        ulong &out1, ulong &out2, ulong &out3, ulong &out4 ) 
{
  // out1 <- 0xff0f
  // out2 <- 0xf3f3
  // out3 <- 0xdddd
  // out4 <- 0xaaff

//ulong t00 = 0xff00;
//ulong t01 = 0xf0f0;
//ulong t02 = 0xcccc;
//ulong t03 = 0xaaaa;

  ulong t04 = ~t00;              // 0x00ff
  out4     ^=  t04 | t03;        // 0xaaff <-----
  ulong t06 =  t01 | t00;        // 0xfff0
  ulong t07 =  t06 ^ t04;        // 0xff0f <-----
  out1     ^=  t07;
  ulong t08 =  t07 ^ t02;        // 0x33c3
  ulong t09 =  t08 | t01;        // 0xf3f3 <-----
  out2     ^=  t09;
  ulong t10 =  t09 ^ t03;        // 0x5959
  out3     ^=  t10 | t02;        // 0xdddd <-----
}

// ------------------------------------------------------------------
inline void csc_transG( ulong t00, ulong t01, ulong t02, ulong t03,
		        ulong &out1, ulong &out2, ulong &out3, ulong &out4 ) 
{
  // out1 <- 0xb1b1
  // out2 <- 0x7722
  // out3 <- 0x583b
  // out4 <- 0xdd50

//ulong t00 = 0xff00;
//ulong t01 = 0xf0f0;
//ulong t02 = 0xcccc;
//ulong t03 = 0xaaaa;

  ulong t04 =  t03 | t00;        // 0xffaa
  ulong t05 =  t03 & t02;        // 0x8888
  ulong t06 =  t05 ^ t04;        // 0x7722 <-----
  out2     ^=  t06;
  ulong t07 =  t03 | t01;        // 0xfafa
  ulong t08 =  t07 ^ t00;        // 0x05fa
  ulong t09 =  t08 | t06;        // 0x77fa
  ulong t10 =  t09 ^ t03;        // 0xdd50 <-----
  out4     ^=  t10;
  ulong t11 =  t03 | t02;        // 0xeeee
  ulong t12 =  t03 & t01;        // 0xa0a0
  ulong t13 =  t12 ^ t11;        // 0x4e4e
  out1     ^= ~t13;              // 0xb1b1 <-----
  ulong t15 =  t13 | t06;        // 0x7f6e
  ulong t16 =  t10 ^ t08;        // 0xd8aa
  ulong t17 =  t16 ^ t15;        // 0xa7c4
  out3     ^= ~t17;              // 0x583b <-----
}
#endif

// ------------------------------------------------------------------

typedef struct {
  ulong in[8];
  ulong *out[8];
} csc_mmxParameters;

#define csc_transP_call(in7, in6, in5, in4, in3, in2, in1, in0,		\
			out7,out6,out5,out4,out3,out2,out1,out0)	\
do {									\
  csc_params->in[0] = (in0);						\
  csc_params->in[1] = (in1);						\
  csc_params->in[2] = (in2);						\
  csc_params->in[3] = (in3);						\
  csc_params->in[4] = (in4);						\
  csc_params->in[5] = (in5);						\
  csc_params->in[6] = (in6);						\
  csc_params->in[7] = (in7);						\
  csc_params->out[0] = &(out0);						\
  csc_params->out[1] = &(out1);						\
  csc_params->out[2] = &(out2);						\
  csc_params->out[3] = &(out3);						\
  csc_params->out[4] = &(out4);						\
  csc_params->out[5] = &(out5);						\
  csc_params->out[6] = &(out6);						\
  csc_params->out[7] = &(out7);						\
  csc_transP( csc_params );						\
} while (0)

extern "C" void csc_transP2( csc_mmxParameters *params ) asm ("csc_transP2");

#ifdef CSC_TRANSP_CLASS
#undef CSC_TRANSP_CLASS
#endif
#if defined( __IN_CSC_COMMON_CPP )      /* need callable csc_transp here */
  #define CSC_TRANSP_CLASS extern "C" 
#elif defined( INLINE_TRANSP )          /* use the inline version */
  //#define csc_transP csc_transPi
  #define CSC_TRANSP_CLASS inline
#else                              /* reference the one in csc_common.cpp */
  extern "C" void csc_transP( csc_mmxParameters *params );
#endif
#ifdef CSC_TRANSP_CLASS
CSC_TRANSP_CLASS void csc_transP( csc_mmxParameters *params )
{

#define _in0  " 0(%0)"
#define _in1  " 8(%0)"
#define _in2  "16(%0)"
#define _in3  "24(%0)"
#define _in4  "32(%0)"
#define _in5  "40(%0)"
#define _in6  "48(%0)"
#define _in7  "56(%0)"
#define _out0 " 0(%1)"
#define _out1 " 4(%1)"
#define _out2 " 8(%1)"
#define _out3 "12(%1)"
#define _out4 "16(%1)"
#define _out5 "20(%1)"
#define _out6 "24(%1)"
#define _out7 "28(%1)"
#define _mmNOT "%2"

// mmNOT      = 4
// in 5,6,7   = 4
// in 0,1,2,3 = 5
// in 4       = 8

// global allocation : in4 == %%mm4
  //		       in0 == %%mm0
  //                   in1 == %%mm1
  //                   in2 == %%mm2

  asm volatile ("

  ## //csc_transF( in3, in2, in1, in0,	// in
  ## //            in7, in6, in5, in4 );// xor-out
  ## {
  ## ulong t04 = ~in3;		    ulong t06 =  in2 | in3;
  ## in4      ^=  t04 | in0;	    ulong t07 =  t06 ^ t04;
  ## ulong t09 =  (t07 ^ in1) | in2;   in7      ^=  t07;
  ## in6      ^=  t09;                 in5      ^=  (t09 ^ in0) | in1;
  ## }
	movq	"_in3", %%mm3	# mm3 = in3
	movq	%%mm3, %%mm7	# mm7 = in3
	pxor	"_mmNOT", %%mm3	# mm3 = t04 = ~in3
	movq	"_in2", %%mm2	# mm2 = in2
	movq	%%mm3, %%mm6	# mm6 = t04
	movq	"_in0", %%mm0	# mm0 = in0
	por	%%mm2, %%mm7	# mm7 = t06 = in2 | in3
	movq	"_in4", %%mm4	# mm4 = in4
	por	%%mm0, %%mm3	# mm3 = t04 | in0
	movq	"_in1", %%mm1	# mm1 = in1
	pxor	%%mm6, %%mm7	# mm7 = t07 = t06 ^ to4
	pxor	%%mm3, %%mm4	# mm4 = in4 ^= t04 | in0
	movq	%%mm7, %%mm5	# mm5 = t07
	pxor	%%mm1, %%mm5	# mm5 = t07 ^ in1
	pxor	"_in7", %%mm7	# mm7 = in7 ^= t07
	por	%%mm2, %%mm5	# mm5 = t09 = (t07 ^ in1) | in2
	movq	%%mm5, %%mm6	# mm6 = t09
	#pxor	"_in0", %%mm6	# mm6 = t09 ^ in0
	pxor	%%mm0, %%mm6
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

  ": : "c"(&(params->in[0])), "d"(&(params->out[0])), "m"(mmNOT)
   : "%eax"

  );

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

#undef CSC_TRANSP_CLASS
#endif

#endif /* ifndef __CSC_COMMON_H */

// ------------------------------------------------------------------
