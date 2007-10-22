// Copyright distributed.net 1997-2003 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// $Id: rotate.h,v 1.7 2007/10/22 16:48:35 jlawson Exp $
//

#ifndef __ROTATE_H__
#define __ROTATE_H__

#if !defined(__GNUC__)
#define __inline__ inline
#endif

//-------------------------------------------------------------------

#if ((CLIENT_CPU == CPU_SPARC) && !defined(__SUNPRO_CC)) || (CLIENT_CPU == CPU_ARM)

#define SHL(x, s) ((u32) ((x) << (s) ))
#define SHR(x, s) ((u32) ((x) >> (32 - (s)) ))

#else

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))

#endif


//-------------------------------------------------------------------

#if (defined(USE_ROTL_MACRO))

// this is faster than the inline assembler on x86/gcc
// because gcc translates it into a roll instruction by itself

#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)

#elif (CLIENT_CPU == CPU_PA_RISC) && defined(__GNUC__)

static __inline__ u32 ROTL(u32 x, u32 y)
{
	register u32 res;

	__asm__ __volatile(
		"mtsar %2\n\tvshd %1,%1,%0\n\t"
		: "=r" (res)
		: "r" (x), "r" (32 - y)
		);
	return res;
}

static __inline__ u32 ROTL3(u32 x)
{
	register u32 res;

	__asm__ __volatile(
		"shd %1,%1,29,%0\n\t"
		: "=r" (res)
		: "r" (x));
	return res;
}

#elif (CLIENT_CPU == CPU_POWERPC) && defined(__GNUC__)

static __inline__ u32 ROTL(u32 x, u32 y)
{
        register u32 res;

        __asm__ __volatile(
                "rlwnm %0,%1,%2,0,31\n\t"
                :"=r" (res)
                :"r" (x), "r" (y));

        return res;
}

static __inline__ u32 ROTL3(u32 x)
{
        register u32 res;

        __asm__ __volatile(
                "rlwinm %0,%1,3,0,31\n\t"
                :"=r" (res)
                :"r" (x));

        return res;
}


#elif (CLIENT_CPU == CPU_POWER) && defined(__GNUC__)

static __inline__ int ROTL(u32 x, u32 y)
{
  register u32 res;

  __asm(
       "rlmi %0, %1, %2, 0, 31"
       :"=r" (res)
       :"r" (x), "r" (y));
  return res;
}

static __inline__ u32 ROTL3(u32 x)
{
	register u32 res;

  __asm(
        "rlimi %0, %1, 3, 0, 31"
        :"=r" (res)
        :"r" (x));
  return res;
}


#elif (CLIENT_CPU == CPU_68K) && defined(__GNUC__)


static __inline__ u32 ROTL(u32 y, u32 x)
{
        __asm__ __volatile(
                "roll %2,%0\n\t"
                :"=d" (y)
                :"0" (y), "d" (x) );
        return y;
}
static __inline__ u32 ROTL3(u32 x)
{
        register u32 res;

        __asm__ __volatile(
                "roll #3,%0\n\t"
                :"=d" (res)
                :"0" (x));

        return res;
}

#elif (CLIENT_CPU == CPU_MIPS) && defined(__GNUC__)

static __inline__ u32 ROTL(u32 x, u32 y)
{
  register u32 res;
  __asm(
        "rol %0, %1, %2"
       :"=r" (res)
       :"r" (x), "r" (y));
  return res;
}

static __inline__ u32 ROTL3(u32 x)
{
  register u32 res;
  __asm(
        "rol %0, %1, 3"
       :"=r" (res)
       :"r" (x));
  return res;
}

#elif (CLIENT_CPU == CPU_S390) && defined(S390_Z_ARCH) && defined(__GNUC__)
   /* In contrast to gcc 2.95.3 the gcc 3.2 does produces faster code    */
   /* with the standard defines for ROTL and ROTL3 on Linux for zSeries. */
   /* There is only a reasonable optimization for 31 bit linux on        */
   /* z/Architecture at the moment.                                      */
   /* z/Architecture has the RLL Operation for 31 Bit OSses as well      */
   /* gcc only translates this with '-Wa,-Aesame' and adequate binutils  */
   /* Looking forward for a -march=z900 which recognizes the rotation    */

static __inline__ u32 ROTL(u32 x, u32 y)
{
  register u32 res;
  __asm__ (
        "rll	%0,%1,0(%2)"
       :"=r" (res)
       :"r" (x), "a" (y));
  return res;
}

static __inline__ u32 ROTL3(u32 x)
{
  register u32 res;
  __asm__ (
        "rll	%0,%1,3"
       :"=r" (res)
       :"r" (x));
  return res;
}

#elif (CLIENT_CPU == CPU_X86) && defined(_MSC_VER)

#pragma warning(push)
#pragma warning(disable:4035)   // no return value (value already in eax).
static __forceinline u32 ROTL(u32 x, u32 y)
{
  __asm mov eax,[x]
  __asm mov ecx,[y]
  __asm rol eax,cl
}
#pragma warning(pop)

#define ROTL3(x)  ROTL(x,3)

#elif (CLIENT_CPU == CPU_X86) && defined(__GNUC__)

static __inline__ u32 ROTL(u32 x, u32 y)
{
	register u32 res;

	__asm__ __volatile(
		"roll %%cl,%0\n\t"
		:"=g" (res)
		:"0" (x), "c" (y));

	return res;
}

static __inline__ u32 ROTL3(u32 x)
{
	register u32 res;

	__asm__ __volatile(
		"roll $3,%0\n\t"
		:"=g" (res)
		:"0" (x));

	return res;
}

#elif (CLIENT_CPU == CPU_ALPHA || CLIENT_CPU == CPU_IA64)

//// This is based on the post on the rc5 list by micha (mbruck@ins-coin.de)
//// It'll work on any DEC Alpha platform and maybe others
//#define ROTL(v, n) (((u32)((v) << ((n) & 31)) + 
//	(u32)((v) >> (32 - ((n) & 31)))) & 0xFFFFFFFF)
//	
//// This is based on the post on the rc5 list by Joao Miguel Neves
////   (rsacrack@camoes.rnl.ist.utl.pt)
//// It'll also work on any DEC Alpha platform and maybe others
//#define ROTL3(x) (((x) << 3) | ((x) >> 29))
//
// This is from Frank Horowitz <frank@ned.dem.csiro.au>, and is reportedly
// 10% faster on alphas (posted to rc5-coders@llamas.net Oct 12/97
//
static __inline__ u32 ROTL(u32 x, u32 s)
{
  register union {unsigned long long a;
    struct{unsigned int hi;unsigned int lo;}b;}temp;
  temp.a = ((unsigned long long) (x) << ((s) & 31));
  return        (temp.b.hi + temp.b.lo );
}

static __inline__ u32 ROTL3(u32 x)
{
  register union {unsigned long long a;
    struct{unsigned int hi;unsigned int lo;}b;}temp;
  temp.a = ((unsigned long long) (x) << 3);
  return        (temp.b.hi + temp.b.lo );
}

#else
	
#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)

#endif

//-------------------------------------------------------------------

#endif

