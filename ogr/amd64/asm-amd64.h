/*
 * Copyright distributed.net 1999-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_AMD64_H__
#define __ASM_AMD64_H__ "@(#)$Id: asm-amd64.h,v 1.12 2009/09/17 20:15:59 andreasb Exp $"


#if defined(__ICC)
  #if (SCALAR_BITS != 32)
  #error Only supported for 32-bit SCALAR typedef.
  #endif
  static inline int __CNTLZ__(register SCALAR i)
  {
    _asm mov eax,i
    _asm not eax
    _asm mov edx,20h
    _asm bsr eax,eax
    _asm sub edx,eax
    _asm mov i,edx
    return i;
  }
  #define __CNTLZ(x) (__CNTLZ__(x))

#elif defined(__WATCOMC__)
  #if (SCALAR_BITS != 32)
  #error Only supported for 32-bit SCALAR typedef.
  #endif

  int __CNTLZ__(SCALAR);
  #pragma aux __CNTLZ__ =  \
          "not  eax"     \
          "mov  edx,20h" \
          "bsr  eax,eax" \
          "sub  edx,eax" \
          value [edx] parm [eax] modify exact [eax edx] nomemory;
  #define __CNTLZ(x) (__CNTLZ__(x))

#elif defined(__GNUC__)

  #if (SCALAR_BITS == 32)
  static __inline__ int __CNTLZ__(register SCALAR input)
  {
     register unsigned int result;
     __asm__("notl %k1\n\t"     \
             "movl $32,%k0\n\t" \
             "bsrl %k1,%k1\n\t"  \
             "subl %k1,%k0\n\t"  \
             :"=R"(result), "=R"(input) : "1"(input) : "cc" );
    return result;
  }
  #define __CNTLZ(x) __CNTLZ__(x)

  #elif (SCALAR_BITS == 64)
  static __inline__ int __CNTLZ__(register SCALAR input)
  {
    register SCALAR result;
    __asm__("notq  %1\n\t"
            "movq  $64,%0\n\t"
            "bsrq  %1,%1\n\t"
            "subq  %1,%0\n\t"
            :"=r"(result), "=r"(input) : "1"(input) : "cc" );
    return (int) result;
  }
  #define __CNTLZ(x) __CNTLZ__(x)

  #else
  #error Unsupport bitsize for SCALAR typedef.
  #endif

#elif defined(_MSC_VER)

  #if defined(_M_AMD64)
  extern "C" unsigned char _BitScanReverse(unsigned long* index, unsigned long mask);
  extern "C" unsigned char _BitScanReverse64(unsigned long * Index, unsigned __int64 Mask);
  #pragma intrinsic(_BitScanReverse)
  #pragma intrinsic(_BitScanReverse64)
  #endif

  static __forceinline int __CNTLZ__(register SCALAR i)
  {
#if defined(_M_AMD64)
	  // unfortunately VC.NET 2003 x64 does not allow inline assembly at all!
      // use a VC.NET intrinsic instead of the BSR instruction.
  #if (SCALAR_BITS == 32)
      register unsigned long tmp;
      _BitScanReverse(&tmp, ~i);
      return (0x20 - tmp);
  #elif (SCALAR_BITS == 64)
      register unsigned long tmp;
      _BitScanReverse64(&tmp, ~i);
      return (0x40 - tmp);
  #endif
#else
  #if (SCALAR_BITS != 32)
  #error Only supported for 32-bit SCALAR typedef.
  #endif

	  __asm {
        mov ecx,i
        not ecx
        mov eax,20h
        bsr ecx,ecx
        sub eax,ecx
      }
      // return value in eax
#endif
  }
  #define __CNTLZ(x) (__CNTLZ__(x))

#endif  /* compiler */

#endif  /* __ASM_AMD64__ */
