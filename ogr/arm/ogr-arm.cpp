/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

#include "ansi/ogrp2-32.h"


const char *ogr_arm_cpp(void) {
return "@(#)$Id: ogr-arm.cpp,v 1.4 2008/12/30 20:58:43 andreasb Exp $"; }

#if defined(ASM_ARM)

  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* 0-2 - partial support */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             1 /* 0-2 - 'yes'           */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */

  #if defined(__GNUC__)
    #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 1)
      static __inline__ int __CNTLZ__(register SCALAR input,
                                      const char bitarray[])
      {
        register int temp, result;
        __asm__ ("mov     %0,#0\n\t"             \
                 "cmp     %1,#0xffff0000\n\t"    \
                 "movcs   %1,%1,lsl#16\n\t"      \
                 "addcs   %0,%0,#16\n\t"         \
                 "cmp     %1,#0xff000000\n\t"    \
                 "movcs   %1,%1,lsl#8\n\t"       \
                 "ldrb    %1,[%3,%1,lsr#24]\n\t" \
                 "addcs   %0,%0,#8\n\t"          \
                 "add     %0,%0,%1"              \
                 : "=r" (result), "=r" (temp)
                 : "1" (input), "r" ((unsigned int)bitarray));
        return result;
      }
      #define __CNTLZ_ARRAY_BASED(x) __CNTLZ__(x)
    #endif
  #endif  /* __GNUC__ */

#endif  /* ASM_ARM */

#include "ansi/ogrp2_codebase.cpp"
