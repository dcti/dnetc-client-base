// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: csc-common.h,v $
// Revision 1.1.2.1  1999/10/07 18:41:14  cyp
// sync'd from head
//
// Revision 1.1  1999/07/23 02:43:06  fordbr
// CSC cores added
//
//

#ifndef __CSC_COMMON_H
#define __CSC_COMMON_H "@(#)$Id: csc-common.h,v 1.1.2.1 1999/10/07 18:41:14 cyp Exp $"

#include <stdlib.h>
#include <string.h>
#include "cputypes.h"

#if defined( BIT_32 ) || defined( MMX_BITSLICER )
  #define CSC_BIT_32
#elif defined( BIT_64 )
  #define CSC_BIT_64
#else
  #error define either CSC_BIT_32 or CSC_BIT_64
#endif

#if (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32) && \
    defined(CSC_BIT_64) && (_MSC_VER >= 11) // VC++ >= 5.0
  typedef unsigned __int64 ulong;
#else
  typedef unsigned long ulong;
#endif

#define _0 ( (ulong)0)
#define _1 (~(_0))

#define PASTE( xx, yy ) PASTE1( xx, yy )
#define PASTE1( xx, yy ) xx##yy

// ------------------------------------------------------------------
// bitslice version of c0..c8
extern const ulong csc_tabc[9][64];

// bitslice version of e and e'
extern const ulong csc_tabe[2][64];

// table-lookup implementation of transP()
extern const u8 csc_tabp[256];

// ------------------------------------------------------------------
inline void transF( ulong t00, ulong t01, ulong t02, ulong t03,
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
inline void transG( ulong t00, ulong t01, ulong t02, ulong t03,
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
#if defined( INLINE_TRANSP ) && !defined( __IN_CSC_COMMON_CPP )
#define transP transPi

inline void transP( ulong in7, ulong in6, ulong in5, ulong in4, 
		    ulong in3, ulong in2, ulong in1, ulong in0,
		    ulong &out7, ulong &out6, ulong &out5, ulong &out4, 
		    ulong &out3, ulong &out2, ulong &out1, ulong &out0 ) 
{
  // y = f(xr) ^ xl
  transF( in3, in2, in1, in0,	// in
	  in7, in6, in5, in4 );	// xor-out
  // zr = g(y) ^ xr
  transG( in7, in6, in5, in4,	// in
	  in3, in2, in1, in0 );	// xor-out  
  // zl = f(zr) ^ y
  transF( in3, in2, in1, in0,	// in
	  in7, in6, in5, in4 );	// xor-out
  // output
  out7 = in7; out6 = in6; out5 = in5; out4 = in4;
  out3 = in3; out2 = in2; out1 = in1; out0 = in0;
}
#else
void transP( ulong in7, ulong in6, ulong in5, ulong in4, 
	     ulong in3, ulong in2, ulong in1, ulong in0,
	     ulong &out7, ulong &out6, ulong &out5, ulong &out4, 
	     ulong &out3, ulong &out2, ulong &out1, ulong &out0 );
#endif

#endif
