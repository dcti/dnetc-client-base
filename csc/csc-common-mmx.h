// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-common-mmx.h,v $
// Revision 1.1.2.1  1999/11/22 18:58:12  remi
// Initial commit of MMX'fied CSC cores.
//
// Revision 1.1.2.4  1999/11/01 17:23:23  cyp
// renamed transX(...) to csc_transX(...) to avoid potential (future) symbol
// collisions.
//
// Revision 1.1.2.3  1999/10/08 14:20:24  remi
// More extern "C" declarations
//
// Revision 1.1.2.2  1999/10/07 23:33:01  cyp
// added some things to help test/force use of 64bit 'registers' on '32bit' cpus
//
// Revision 1.1.2.1  1999/10/07 18:41:14  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:06  fordbr
// CSC cores added
//
//

#ifndef __CSC_COMMON_H
#define __CSC_COMMON_H "@(#)$Id: csc-common-mmx.h,v 1.1.2.1 1999/11/22 18:58:12 remi Exp $"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "cputypes.h" /* u32, s32 */

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

#define _out1  "%0"
#define _out2  "%1"
#define _out3  "%2"
#define _out4  "%3"
#define _t00   "%4"
#define _t01   "%5"
#define _t02   "%6"
#define _t03   "%7"
#define _mmNOT "%8"

//* ulong t04 = ~t00;              // 0x00ff
//* out4     ^=  t04 | t03;        // 0xaaff <-----
//* ulong t06 =  t01 | t00;        // 0xfff0
//* ulong t07 =  t06 ^ t04;        // 0xff0f <-----
//* out1     ^=  t07;
//* ulong t08 =  t07 ^ t02;        // 0x33c3
//* ulong t09 =  t08 | t01;        // 0xf3f3 <-----
//* out2     ^=  t09;
//* ulong t10 =  t09 ^ t03;        // 0x5959
//* out3     ^=  t10 | t02;        // 0xdddd <-----

//* ulong t04 = ~t00;           ulong t06 =  t01 | t00;
//* out4     ^=  t04 | t03;     ulong t07 =  t06 ^ t04;
//* ulong t08 =  t07 ^ t02;     out1     ^=  t07;	   
//* ulong t09 =  t08 | t01;
//* out2     ^=  t09;	        out3     ^=  (t09 ^ t03) | t02;

  asm volatile ("
	movq	"_t00", %%mm0	# mm0 = t00
	movq	%%mm0, %%mm1	# mm1 = t00
	movq	"_t01", %%mm3   # mm3 = t01
	pxor	"_mmNOT", %%mm0	# mm0(t04) = ~t00

	por	%%mm3, %%mm1	# mm1(t06) = t01 | t00

	movq	%%mm0, %%mm2	# mm2 = t04
	pxor	%%mm1, %%mm2	# mm2(t07) = t04 ^ t06

	movq	"_t03", %%mm5
	por	%%mm5, %%mm0	# mm0 = t04 | t03
	pxor	"_out4", %%mm0	# out4 ^= t04 | t03
	movq	%%mm0, "_out4"

	movq	"_out1", %%mm4
	pxor	%%mm2, %%mm4	# out1 ^= t07
	movq	%%mm4, "_out1"

	movq	"_t02", %%mm6
	pxor	%%mm6, %%mm2	# mm2(t08) = t07 ^ t02
	por	%%mm3, %%mm2	# mm2(t09) = t08 | t01

	movq	"_out2", %%mm4
	pxor	%%mm2, %%mm4	# out2 ^= t09
	movq	%%mm4, "_out2"

	pxor	%%mm5, %%mm2	# mm2(t10) = t09 ^ t03
	por	%%mm6, %%mm2	# mm2 = t10 | t02
	pxor	"_out3", %%mm2	# out3 ^= t10 | t02
	movq	%%mm2, "_out3"

  ": "=m"(out1)/*0*/, "=m"(out2)/*1*/, "=m"(out3)/*2*/, "=m"(out4)/*3*/
   :  "m"(t00)/*4*/,   "m"(t01)/*5*/,   "m"(t02)/*6*/,   "m"(t03)/*7*/,  "m"(mmNOT)/*8*/
  );
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

#define _out1  "%0"
#define _out2  "%1"
#define _out3  "%2"
#define _out4  "%3"
#define _t00   "%4"
#define _t01   "%5"
#define _t02   "%6"
#define _t03   "%7"
#define _mmNOT "%8"

//ulong t04 =  t03 | t00;        // 0xffaa
//ulong t05 =  t03 & t02;        // 0x8888
//ulong t06 =  t05 ^ t04;        // 0x7722 <-----
//out2     ^=  t06;
//ulong t07 =  t03 | t01;        // 0xfafa
//ulong t08 =  t07 ^ t00;        // 0x05fa
//ulong t09 =  t08 | t06;        // 0x77fa
//ulong t10 =  t09 ^ t03;        // 0xdd50 <-----
//out4     ^=  t10;
//ulong t11 =  t03 | t02;        // 0xeeee
//ulong t12 =  t03 & t01;        // 0xa0a0
//ulong t13 =  t12 ^ t11;        // 0x4e4e
//out1     ^= ~t13;              // 0xb1b1 <-----
//ulong t15 =  t13 | t06;        // 0x7f6e
//ulong t16 =  t10 ^ t08;        // 0xd8aa
//ulong t17 =  t16 ^ t15;        // 0xa7c4
//out3     ^= ~t17;              // 0x583b <-----


  //ulong t06 =  (t03 & t02) ^ (t03 | t00);    ulong t08 =  (t03 | t01) ^ t00;
  //out2     ^=  t06;                          ulong t10 =  (t08 | t06) ^ t03;
  //                                           out4     ^=  t10;
  //ulong t13 =  (t03 & t01) ^ (t03 | t02);
  //out1     ^= ~t13;                          out3     ^= ~(t10 ^ t08 ^ (t13 | t06));

  asm volatile ("
	movq	"_t03",%%mm0	# mm0 = t03
	movq	%%mm0, %%mm1	# mm1 = t03
	movq	"_t00",%%mm3	# mm3 = t00
	movq	%%mm0, %%mm5	# mm5 = t03
	movq	"_t02",%%mm7	# mm7 = t02
	movq	%%mm0, %%mm2	# mm2 = t03

	por	%%mm3, %%mm0	# mm0 = t03 | t00
	pand	%%mm7, %%mm1	# mm1 = t03 & t02
	por	"_t01",%%mm2	# mm2 = t03 | t01
	pxor	%%mm1, %%mm0	# mm0(t06) = (t03 | t00) ^ (t03 & t02)
	movq	"_out2",%%mm1	# mm1 = out2
	movq	%%mm5, %%mm6	# mm6 = t03
	pxor	%%mm3, %%mm2	# mm2(t08) = (t03 | t01) ^ t00
	movq	%%mm0, %%mm3	# mm3 = t06
	pxor	%%mm0, %%mm1	# mm1 = out2 ^ t06
	por	%%mm2, %%mm3	# mm3 = t06 | t08
	movq	%%mm1,"_out2"	# ** out2 ^= t06
	movq	%%mm0, %%mm1	# mm1 = t06
	pxor	"_t03",%%mm3	# mm3(t10) = (t06 | t08) ^ t03
	movq	%%mm3, %%mm4	# mm4 = t10
	pxor	"_out4",%%mm3	# mm3 = out4 ^ t10
	pxor	%%mm4, %%mm2	# mm2 = t10 ^ t08
	movq	%%mm3, "_out4"  # ** out4 ^= t10
	por	%%mm5, %%mm7	# mm7 = t03 | t02
	pand	"_t01",%%mm5	# mm5 = t03 & t01
	pxor	%%mm7, %%mm5	# mm5(t13) = (t03 & t01) ^ (t03 | t02)
	movq	"_out1", %%mm0	# mm0 = out1
	por	%%mm5, %%mm1	# mm1 = t13 | t06
	movq	"_mmNOT", %%mm3	# mm3 = mmNOT
	pxor	%%mm5, %%mm0	# mm0 = out1 ^ t13
	pxor	%%mm2, %%mm1	# mm1 = t10 ^ t08 ^ (t13 | t06)
	pxor	"_out3", %%mm1	# mm1 = out3 ^ t10 ^ t08 ^ (t13 | t06)
	pxor	%%mm3, %%mm0	# mm0 = out1 ^ ~t13
	pxor	%%mm3, %%mm1	# mm1 = out3 ^ ~(t10 ^ t08 ^ (t13 | t06))
	movq	%%mm0, "_out1"	# ** out1 ^= t13
	movq	%%mm1, "_out3"	# ** out3 ^= t10 ^ t08 ^ (t13 | t06)

  ": "=m"(out1), "=m"(out2), "=m"(out3), "=m"(out4)
   :  "m"(t00),   "m"(t01),   "m"(t02),   "m"(t03),  "m"(mmNOT)
  );
}

// ------------------------------------------------------------------

#ifdef CSC_TRANSP_CLASS
#undef CSC_TRANSP_CLASS
#endif
#if defined( __IN_CSC_COMMON_CPP )      /* need callable csc_transp here */
  #define CSC_TRANSP_CLASS extern "C" 
#elif defined( INLINE_TRANSP )          /* use the inline version */
  //#define csc_transP csc_transPi
  #define CSC_TRANSP_CLASS inline
#else                              /* reference the one in csc_common.cpp */
  extern "C" void csc_transP( ulong in7, ulong in6, ulong in5, ulong in4, 
 	                ulong in3, ulong in2, ulong in1, ulong in0,
		        ulong &out7, ulong &out6, ulong &out5, ulong &out4, 
  		        ulong &out3, ulong &out2, ulong &out1, ulong &out0 );
#endif
#ifdef CSC_TRANSP_CLASS
CSC_TRANSP_CLASS void csc_transP( ulong in7, ulong in6, ulong in5, ulong in4, 
		        ulong in3, ulong in2, ulong in1, ulong in0,
		        ulong &out7, ulong &out6, ulong &out5, ulong &out4, 
  		        ulong &out3, ulong &out2, ulong &out1, ulong &out0 ) 
{

/*
  // csc_transF( in3, in2, in1, in0,	// in
  //	         in7, in6, in5, in4 );	// xor-out
  {
  ulong t04 = ~in3;           ulong t06 =  in2 | in3;
  in4      ^=  t04 | in0;     ulong t07 =  t06 ^ t04;
  ulong t08 =  t07 ^ in1;     in7      ^=  t07;	   
  ulong t09 =  t08 | in2;
  in6      ^=  t09;           in5      ^=  (t09 ^ in0) | in1;
  }
  asm volatile ("
	
  ": "=m"(in7), "=m"(in6), "=m"(in5), "=m"(in4),
     "=m"(in3), "=m"(in2), "=m"(in1), "=m"(in0) : "m"(mmNOT)
  );

  // csc_transG( in7, in6, in5, in4,	// in
  //	         in3, in2, in1, in0 );	// xor-out
  {
  ulong t06 =  (in4 & in5) ^ (in4 | in7);    ulong t08 =  (in4 | in6) ^ in7;
  out2 = (in2 ^= t06);                       ulong t10 =  (t08 | t06) ^ in4;
                                             out0 = (in0 ^= t10);
  ulong t13 =  (in4 & in6) ^ (in4 | in5);
  out3 = (in3 ^= ~t13);                      out1 = (in1 ^= ~(t10 ^ t08 ^ (t13 | t06)));
  }

  // csc_transF( in3, in2, in1, in0,	// in
  // 	         in7, in6, in5, in4 );	// xor-out
  {
  ulong t04 = ~in3;           ulong t06 =  in2 | in3;
  out4 = in4 ^ (t04 | in0);   ulong t07 =  t06 ^ t04;
  ulong t08 =  t07 ^ in1;     out7 = in7 ^  t07;	   
  ulong t09 =  t08 | in2;
  out6 = in6 ^ t09;           out5 = in5 ^ ((t09 ^ in0) | in1);
  }
*/

#define _in7 "%0"
#define _in6 "%1"
#define _in5 "%2"
#define _in4 "%3"
#define _in3 "%4"
#define _in2 "%5"
#define _in1 "%6"
#define _in0 "%7"
#define _mmNOT "%8"

  asm volatile ("

  # y = f(xr) ^ xl
  # csc_transF( in3, in2, in1, in0,	// in
  #	        in7, in6, in5, in4 );	// xor-out

	movq	"_in3", %%mm0	# mm0 = t00
	movq	%%mm0, %%mm1	# mm1 = t00
	movq	"_in2", %%mm3   # mm3 = t01
	pxor	"_mmNOT", %%mm0	# mm0(t04) = ~t00
	por	%%mm3, %%mm1	# mm1(t06) = t01 | t00
	movq	"_in0", %%mm5
	movq	%%mm0, %%mm2	# mm2 = t04
	por	%%mm5, %%mm0	# mm0 = t04 | t03
	pxor	%%mm1, %%mm2	# mm2(t07) = t04 ^ t06
	pxor	"_in4", %%mm0	# out4 ^= t04 | t03
	movq	%%mm0, "_in4"
	movq	"_in7", %%mm4
	pxor	%%mm2, %%mm4	# out1 ^= t07
	movq	%%mm4, "_in7"
	movq	"_in1", %%mm6
	pxor	%%mm6, %%mm2	# mm2(t08) = t07 ^ t02
	por	%%mm3, %%mm2	# mm2(t09) = t08 | t01
	movq	"_in6", %%mm4
	pxor	%%mm2, %%mm4	# out2 ^= t09
	movq	%%mm4, "_in6"
	pxor	%%mm5, %%mm2	# mm2(t10) = t09 ^ t03
	por	%%mm6, %%mm2	# mm2 = t10 | t02
	pxor	"_in5", %%mm2	# out3 ^= t10 | t02
	movq	%%mm2, "_in5"

  # zr = g(y) ^ xr
  # csc_transG( in7, in6, in5, in4,	// in
  #	        in3, in2, in1, in0 );	// xor-out

	movq	"_in4",%%mm0	# mm0 = t03
	movq	%%mm0, %%mm1	# mm1 = t03
	movq	"_in7",%%mm3	# mm3 = t00
	movq	%%mm0, %%mm5	# mm5 = t03
	movq	"_in5",%%mm7	# mm7 = t02
	movq	%%mm0, %%mm2	# mm2 = t03

	por	%%mm3, %%mm0	# mm0 = t03 | t00
	pand	%%mm7, %%mm1	# mm1 = t03 & t02
	por	"_in6",%%mm2	# mm2 = t03 | t01
	pxor	%%mm1, %%mm0	# mm0(t06) = (t03 | t00) ^ (t03 & t02)
	movq	"_in2",%%mm1	# mm1 = out2
	movq	%%mm5, %%mm6	# mm6 = t03
	pxor	%%mm3, %%mm2	# mm2(t08) = (t03 | t01) ^ t00
	movq	%%mm0, %%mm3	# mm3 = t06
	pxor	%%mm0, %%mm1	# mm1 = out2 ^ t06
	por	%%mm2, %%mm3	# mm3 = t06 | t08
	movq	%%mm1,"_in2"	# ** out2 ^= t06
	movq	%%mm0, %%mm1	# mm1 = t06
	pxor	"_in4",%%mm3	# mm3(t10) = (t06 | t08) ^ t03
	movq	%%mm3, %%mm4	# mm4 = t10
	pxor	"_in0",%%mm3	# mm3 = out4 ^ t10
	pxor	%%mm4, %%mm2	# mm2 = t10 ^ t08
	movq	%%mm3, "_in0"  # ** out4 ^= t10
	por	%%mm5, %%mm7	# mm7 = t03 | t02
	pand	"_in6",%%mm5	# mm5 = t03 & t01
	pxor	%%mm7, %%mm5	# mm5(t13) = (t03 & t01) ^ (t03 | t02)
	movq	"_in3", %%mm0	# mm0 = out1
	por	%%mm5, %%mm1	# mm1 = t13 | t06
	movq	"_mmNOT", %%mm3	# mm3 = mmNOT
	pxor	%%mm5, %%mm0	# mm0 = out1 ^ t13
	pxor	%%mm2, %%mm1	# mm1 = t10 ^ t08 ^ (t13 | t06)
	pxor	"_in1", %%mm1	# mm1 = out3 ^ t10 ^ t08 ^ (t13 | t06)
	pxor	%%mm3, %%mm0	# mm0 = out1 ^ ~t13
	pxor	%%mm3, %%mm1	# mm1 = out3 ^ ~(t10 ^ t08 ^ (t13 | t06))
	movq	%%mm0, "_in3"	# ** out1 ^= t13
	movq	%%mm1, "_in1"	# ** out3 ^= t10 ^ t08 ^ (t13 | t06)

  # zl = f(zr) ^ y
  # csc_transF( in3, in2, in1, in0,	// in
  # 	        in7, in6, in5, in4 );	// xor-out

	movq	"_in3", %%mm0	# mm0 = t00
	movq	%%mm0, %%mm1	# mm1 = t00
	movq	"_in2", %%mm3   # mm3 = t01
	pxor	"_mmNOT", %%mm0	# mm0(t04) = ~t00
	por	%%mm3, %%mm1	# mm1(t06) = t01 | t00
	movq	"_in0", %%mm5
	movq	%%mm0, %%mm2	# mm2 = t04
	por	%%mm5, %%mm0	# mm0 = t04 | t03
	pxor	%%mm1, %%mm2	# mm2(t07) = t04 ^ t06
	pxor	"_in4", %%mm0	# out4 ^= t04 | t03
	movq	%%mm0, "_in4"
	movq	"_in7", %%mm4
	pxor	%%mm2, %%mm4	# out1 ^= t07
	movq	%%mm4, "_in7"
	movq	"_in1", %%mm6
	pxor	%%mm6, %%mm2	# mm2(t08) = t07 ^ t02
	por	%%mm3, %%mm2	# mm2(t09) = t08 | t01
	movq	"_in6", %%mm4
	pxor	%%mm2, %%mm4	# out2 ^= t09
	movq	%%mm4, "_in6"
	pxor	%%mm5, %%mm2	# mm2(t10) = t09 ^ t03
	por	%%mm6, %%mm2	# mm2 = t10 | t02
	pxor	"_in5", %%mm2	# out3 ^= t10 | t02
	movq	%%mm2, "_in5"

  ": "=m"(in7), "=m"(in6), "=m"(in5), "=m"(in4),
     "=m"(in3), "=m"(in2), "=m"(in1), "=m"(in0) : "m"(mmNOT)
  );


  // Output
  out7 = in7; out6 = in6; out5 = in5; out4 = in4;
  out3 = in3; out2 = in2; out1 = in1; out0 = in0;
}
#undef CSC_TRANSP_CLASS
#endif

#endif /* ifndef __CSC_COMMON_H */

// ------------------------------------------------------------------
