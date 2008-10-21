/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_AMD64_H__
#define __ASM_AMD64_H__ "@(#)$Id: asm-amd64.h,v 1.6 2008/10/21 03:39:07 jlawson Exp $"

#if defined(__ICC)
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
  #define __CNTLZ(x) __CNTLZ__(x)

#elif defined(__WATCOMC__)

  int __CNTLZ__(SCALAR);
  #pragma aux __CNTLZ__ =  \
          "not  eax"     \
          "mov  edx,20h" \
          "bsr  eax,eax" \
          "sub  edx,eax" \
          value [edx] parm [eax] modify exact [eax edx] nomemory;
  #define __CNTLZ(x) __CNTLZ__(x)

#elif defined(__GNUC__)

  static __inline__ int __CNTLZ__(register SCALAR input)
  {
     register unsigned int result;
     __asm__("notl %k1\n\t"     \
             "movl $33,%k0\n\t" \
             "bsrl %k1,%k1\n\t"  \
             "jz   0f\n\t"     \
             "subl %k1,%k0\n\t"  \
             "decl %k0\n\t"     \
             "0:"              \
             :"=R"(result), "=R"(input) : "1"(input) : "cc" );
    return result;
  }
  #define __CNTLZ(x) __CNTLZ__(x)

#elif defined(_MSC_VER)

  static __forceinline int __CNTLZ__(register SCALAR i)
  {
      __asm {
        mov ecx,i
        not ecx
        mov eax,20h
        bsr ecx,ecx
        sub eax,ecx
      }
      // return value in eax
  }
  #define __CNTLZ(x) __CNTLZ__(x)

#endif  /* compiler */

#endif  /* __ASM_AMD64__ */
