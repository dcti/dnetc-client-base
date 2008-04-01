/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_X86_H__
#define __ASM_X86_H__ "@(#)$Id: asm-x86.h,v 1.8 2008/04/01 14:42:19 stream Exp $"

#if (SCALAR_BITS == 32)
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

    int __CNTLONEREV__(SCALAR);
    #pragma aux __CNTLONEREV__ =  \
            "bsr  eax,eax"        \
            value [eax] parm [eax] modify [eax] nomemory;
    #define __CNTLZ(x) (32-__CNTLONEREV__(~(x)))

  #elif defined(__GNUC__)

    static __inline__ int __CNTLZ__(register SCALAR input)
    {
       register SCALAR result;
       __asm__("notl %1\n\t"
               "movl $33,%0\n\t"
               "bsrl %1,%1\n\t"
               "jz   0f\n\t"
               "subl %1,%0\n\t"
               "decl %0\n\t"
               "0:"
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
#else   /*------------------- assume SCALAR_BITS == 64 ---------------------*/
  #if defined(__GNUC__)
    static __inline__ int __CNTLZ__(register SCALAR input)
    {
      register SCALAR result;
      __asm__("notq  %1\n\t"
              "movq  $65,%0\n\t"
              "bsrq  %1,%1\n\t"
              "jz   0f\n\t"
              "subq  %1,%0\n\t"
              "decq  %0\n\t"
              "0:"
              :"=r"(result), "=r"(input) : "1"(input) : "cc" );
      return (int) result;
    }
    #define __CNTLZ(x) __CNTLZ__(x)
  #endif  /* compiler */
#endif  /* SCALAR_BITS */

#endif  /* __ASM_X86__ */
