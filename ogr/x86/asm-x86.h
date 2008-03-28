/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_X86_H__
#define __ASM_X86_H__ "@(#)$Id: asm-x86.h,v 1.6 2008/03/28 22:21:34 kakace Exp $"

/*
 * Macro to check assertions at compile-time (e.g. sizeof(foo) == something)
 */
#define STATIC_ASSERT(cond)  { typedef int safoo[(cond) ? 1 : -1]; }

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

    #if (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
      #include <stddef.h>   /* offsetof */
      #include "ansi/ogrp2_corestate.h"

      void LIST_RIGHT_xx(void *levlist, int s, int newbit);
      void COMP_LEFT_xx(void *levcomp, int s);
      #pragma aux LIST_RIGHT_xx =  \
        "mov  edx,[edi+12]"                  \
        "shrd [edi+16],edx,cl"               \
        "mov  edx,[edi+8]"                   \
        "shrd [edi+12],edx,cl"               \
        "mov  edx,[edi+4]"                   \
        "shrd [edi+8],edx,cl"                \
        "mov edx,[edi+0]"                    \
        "shrd [edi+4],edx,cl"                \
        "shrd edx,esi,cl"                    \
        "mov  [edi+0],edx"                   \
        parm [edi] [ecx] [esi] modify exact [edx];

      #pragma aux COMP_LEFT_xx =  \
        "mov  eax,[edi+4]"        \
        "mov  edx,[edi+8]"        \
        "shld [edi+0],eax,cl"     \
        "shld eax,edx,cl"         \
        "mov  [edi+4],eax"        \
        "mov  eax,[edi+12]"       \
        "shld edx,eax,cl"         \
        "mov  [edi+8],edx"        \
        "mov  edx,[edi+16]"       \
        "shld eax,edx,cl"         \
        "mov  [edi+12],eax"       \
        "shl  edx,cl"             \
        "mov  [edi+16],edx"       \
        parm [edi] [ecx] modify exact [eax edx ebx];

      #define COMP_LEFT_LIST_RIGHT(lev,s)        \
        STATIC_ASSERT( offsetof(struct Level, list) == 0  );   \
        STATIC_ASSERT( offsetof(struct Level, comp) == 40 );   \
        STATIC_ASSERT( sizeof(lev->list) == 20 );              \
        STATIC_ASSERT( sizeof(lev->comp) == 20 );              \
        LIST_RIGHT_xx(&(lev->list[0]),s,newbit); \
        COMP_LEFT_xx(&(lev->comp[0]),s);         \
        newbit = 0;                              \
        comp0  = lev->comp[0];                   

    #endif

    int __CNTLONEREV__(SCALAR);
    #pragma aux __CNTLONEREV__ =  \
            "bsr  eax,eax"      \
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
