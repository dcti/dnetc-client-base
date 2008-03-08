/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_AMD64_H__
#define __ASM_AMD64_H__ "@(#)$Id: asm-amd64.h,v 1.4 2008/03/08 20:18:29 kakace Exp $"

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
     __asm__("notl %1\n\t"     \
             "movl $32,%0\n\t" \
             "bsrl %1,%1\n\t"  \
             "subl %1,%0\n\t"  \
             "0:"              \
             :"=r"(result), "=r"(input) : "1"(input) : "cc" );
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
