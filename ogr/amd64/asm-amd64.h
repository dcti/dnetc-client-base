/*
 * Copyright distributed.net 1999-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_AMD64_H__
#define __ASM_AMD64_H__ "@(#)$Id: asm-amd64.h,v 1.1.2.1 2004/08/10 18:20:56 jlawson Exp $"

/* If we were to cover the whole range of 0x00000000 ... 0xffffffff
   we would need ...
   static __inline__ int __CNTLZ__(register unsigned int input)
   {
      register unsigned int result;
      __asm__("notl %1\n\t"     \
              "movl $33,%0\n\t" \
              "bsrl %1,%1\n\t"  \
              "jz   0f\n\t"     \
              "subl %1,%0\n\t"  \
              "decl %0\n\t"     \
              "0:"              \
              :"=r"(result), "=r"(input) : "1"(input) : "cc" );
      return result;
   }
   but since the function is only executed for (comp0 < 0xfffffffe),
   we can optimize it to...
*/

#if defined(__ICC)
  static inline int __CNTLZ__(register unsigned int i)
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

  #if (OGROPT_ALTERNATE_CYCLE == 0) && (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
    #error fixme: No longer compatible with existing code
    // need to shift-in "newbit"
    void COMP_LEFT_LIST_RIGHT_xx(U *levcomp, U *levlist, int s);
    #pragma aux COMP_LEFT_LIST_RIGHT_xx =  \
      "mov eax,[edi+4]"                   \
      "mov edx,[esi+12]"                  \
      "shld [edi+0],eax,cl"               \
      "shrd [esi+16],edx,cl"              \
      "mov eax,[edi+8]"                   \
      "mov edx,[esi+8]"                   \
      "shld [edi+4],eax,cl"               \
      "shrd [esi+12],edx,cl"              \
      "mov eax,[edi+12]"                  \
      "mov edx,[esi+4]"                   \
      "shld [edi+8],eax,cl"               \
      "shrd [esi+8],edx,cl"               \
      "mov eax,[edi+16]"                  \
      "mov edx,[esi+0]"                   \
      "shld [edi+12],eax,cl"              \
      "shrd [esi+4],edx,cl"               \
      "shl eax,cl"                        \
      "shr edx,cl"                        \
      "mov [edi+16],eax"                  \
      "mov [esi+0],edx"                   \
      parm [edi] [esi] [ecx] modify exact [edx eax];

    #define COMP_LEFT_LIST_RIGHT(lev,s) \
      COMP_LEFT_LIST_RIGHT_xx(&(lev->comp[0]),&(lev->list[0]),s)
  #endif

  int __CNTLZ__(unsigned int);
  #pragma aux __CNTLZ__ =  \
          "not  eax"     \
          "mov  edx,20h" \
          "bsr  eax,eax" \
          "sub  edx,eax" \
          value [edx] parm [eax] modify exact [eax edx] nomemory;
  #define __CNTLZ(x) __CNTLZ__(x)

#elif defined(__GNUC__)

  #if (OGROPT_ALTERNATE_CYCLE == 0) && (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
    #error fixme: No longer compatible with existing code
    // need to shift-in "newbit"
    #define COMP_LEFT_LIST_RIGHT(lev,s) { \
      asm(                                \
        "movl  4(%0),%%eax\n\t"           \
        "movl  12(%1),%%edx\n\t"          \
                                          \
        "shldl %%cl,%%eax,(%0)\n\t"       \
        "movl  8(%0),%%eax\n\t"           \
                                          \
        "shrdl %%cl,%%edx,16(%1)\n\t"     \
        "movl  8(%1),%%edx\n\t"           \
                                          \
        "shldl %%cl,%%eax,4(%0)\n\t"      \
        "movl  12(%0),%%eax\n\t"          \
                                          \
        "shrdl %%cl,%%edx,12(%1)\n\t"     \
        "movl  4(%1),%%edx\n\t"           \
                                          \
        "shldl %%cl,%%eax,8(%0)\n\t"      \
        "movl  16(%0),%%eax\n\t"          \
                                          \
        "shrdl %%cl,%%edx,8(%1)\n\t"      \
        "movl  (%1),%%edx\n\t"            \
                                          \
        "shldl %%cl,%%eax,12(%0)\n\t"     \
        "shrdl %%cl,%%edx,4(%1)\n\t"      \
                                          \
        "shll  %%cl,16(%0)\n\t"           \
        "shrl  %%cl,(%1)\n\t"             \
                                          \
        : /* no output */                 \
        : "D" (&(lev->comp)), "S" (&(lev->list)), "c" (s) /* get s in ecx*/ \
        : "memory", "cc", "eax", "edx"    \
      );                                  \
    }
  #endif

  static __inline__ int __CNTLZ__(register unsigned int input)
  {
     register unsigned int result;
     __asm__("notl %1\n\t"     \
             "movl $32,%0\n\t" \
             "bsrl %1,%1\n\t"  \
             "subl %1,%0\n\t"  \
             :"=r"(result), "=r"(input) : "1"(input) : "cc" );
     return result;
  }
  #define __CNTLZ(x) __CNTLZ__(x)

#elif defined(_MSC_VER)

  static __forceinline int __CNTLZ__(register unsigned int i)
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
