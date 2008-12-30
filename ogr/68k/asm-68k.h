/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#ifndef __ASM_68K_H__
#define __ASM_68K_H__ "@(#)$Id: asm-68k.h,v 1.4 2008/12/30 20:58:43 andreasb Exp $"

#if defined(ASM_68K) && defined(__GNUC__)

  #if (__NeXT__)

    #if defined(mc68020) || defined(mc68030) || defined(mc68040) || defined(mc68060)
      static __inline__ int __CNTLZ__(register SCALAR i) {
        __asm__ ("bfffo %0{#0:#0},%0" : "=d" (i) : "0" (i));
        return i;
      }
      #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
    #else
      static __inline__ int __CNTLZ__(register SCALAR input,
                                      const char bitarray[])
      {
        register int result;

        /* gcc-2.5.8 on NeXT needs (&ogr_first_blank_8bit[0]) for
         * address register parameters. Otherwise it will give:
         * ogr/ansi/ogr.cpp:2172: inconsistent operand constraints in an `asm'
        */
        __asm__ ("   cmpl    #0xffff0000,%1\n"
                 "   bcs     0f\n"
                 "   moveq   #16,%0\n"
                 "   bra     1f\n"
                 "0: swap    %1\n"
                 "   moveq   #0,%0\n"
                 "1: cmpw    #0xff00,%1\n"
                 "   bcs     2f\n"
                 "   lslw    #8,%1\n"
                 "   addql   #8,%0\n"
                 "2: lsrw    #8,%1\n"
                 "   addb    %3@(0,%1:w),%0"
                 :"=d" (result), "=d" (input)
                 : "1" (input), "a" (&bitarray[0]));
        return result;
      }
      #define __CNTLZ_ARRAY_BASED(x,y) __CNTLZ__(x,y)
    #endif

  #else /* !__NeXT__ */

    #if defined(mc68020) || defined(mc68030) || defined(mc68040) || defined(mc68060)
      static __inline__ int __CNTLZ__(register SCALAR i) {
        __asm__ ("bfffo %0,0,0,%0" : "=d" (i) : "0" (i));
        return i;
      }
      #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
    #else
      static __inline__ int __CNTLZ__(register SCALAR input,
                                      const char bitarray[])
      {
        register int result;
        __asm__ ("   cmp.l   #0xffff0000,%1\n"
                 "   bcs.b   0f\n"
                 "   moveq   #16,%0\n"
                 "   bra.b   1f\n"
                 "0: swap    %1\n"
                 "   moveq   #0,%0\n"
                 "1: cmp.w   #0xff00,%1\n"
                 "   bcs.b   2f\n"
                 "   lsl.w   #8,%1\n"
                 "   addq    #8,%0\n"
                 "2: lsr.w   #8,%1\n"
                 "   add.b   0(%3,%1.w),%0"
                 :"=d" (result), "=d" (input)
                 : "1" (input), "a" (bitarray));
        return result;
      }
      #define __CNTLZ_ARRAY_BASED(x,y) __CNTLZ__(x,y)
    #endif

    #if defined(mc68040)
      /* align to 8-byte boundary - pad with nops */
      #define __BALIGN __asm__ __volatile__ (".balignw 8,0x4e71" : : )
    #elif defined(mc68060) || defined(mc68030) || defined(mc68020)
      /* align to 4-byte boundary - pad with nops */
      #define __BALIGN __asm__ __volatile__ (".balignw 4,0x4e71" : : )
    #endif

  #endif  /* !__NeXT__ */
#endif /* ASM_68K && __GNUC__ */
#endif __ASM_68K_H__
