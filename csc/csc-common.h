// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-common.h,v $
// Revision 1.1.2.7  1999/11/29 19:12:53  lyndon
// Document the reason for the MIPS-specific tests.
//
// Revision 1.1.2.6  1999/11/29 00:29:44  lyndon
// Irix MIPSpro incremental commit:
//
// * Adds support for 64 bit builds
// * Re-enable DES and CSC
// * Heavy optimizations enabled
// * Portability fixes in CSC
//
// Revision 1.1.2.5  1999/11/28 06:17:05  lyndon
//
// Irix builds were botching the ULONG_MAX test, thus the code was defaulting
// to 64 bits everywhere. (I don't know why ...)
//
// Both the MIPSpro and GNU compilers define _MIPS_SZLONG. If that define
// is present, use it in preference to the ULONG_MAX-based tests.
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
#define __CSC_COMMON_H "@(#)$Id: csc-common.h,v 1.1.2.7 1999/11/29 19:12:53 lyndon Exp $"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "cputypes.h" /* u32, s32 */

//#define _FORCE64_

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
#elif defined(_MIPS_SZLONG)
/*
 * The tests against ULONG_MAX in the following section fail on
 * Irix/MIPSpro due to sign-extension of the constants. The
 * Irix compilers explicitly provide the information we need, so
 * we use that instead.
 */
  #if (_MIPS_SZLONG == 32)
    #define CSC_BIT_32
    typedef unsigned long ulong;
  #elif (_MIPS_SZLONG == 64)
    #define CSC_BIT_64
    typedef unsigned long ulong;
    #define CASTNUM64(n) n##ul
  #else
    #error Insane value of _MIPS_SZLONG
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
  // y = f(xr) ^ xl
  csc_transF( in3, in2, in1, in0,	// in
	      in7, in6, in5, in4 );	// xor-out
  // zr = g(y) ^ xr
  csc_transG( in7, in6, in5, in4,	// in
	      in3, in2, in1, in0 );	// xor-out  
  // zl = f(zr) ^ y
  csc_transF( in3, in2, in1, in0,	// in
 	      in7, in6, in5, in4 );	// xor-out
  // output
  out7 = in7; out6 = in6; out5 = in5; out4 = in4;
  out3 = in3; out2 = in2; out1 = in1; out0 = in0;
}
#undef CSC_TRANSP_CLASS
#endif

#endif /* ifndef __CSC_COMMON_H */

// ------------------------------------------------------------------
