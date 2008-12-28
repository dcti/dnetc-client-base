/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Derived from ansi/ogrng-32.cpp
 *
 * $Id: ogrng-arm1.cpp,v 1.1 2008/12/28 23:20:52 teichp Exp $
*/

#include "ansi/ogrng-32.h"

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* 0-2 - 'no'  (default) */
#define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */

#if defined(__GNUC__)
  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 1)
    static __inline__ int __CNTLZ__(register unsigned int input,
                                    const char bitarray[])
    {
      register int result=0;
      __asm__ ("cmp     %2,#0xffff0000\n\t"    \
               "movcs   %2,%2,lsl#16\n\t"      \
               "addcs   %0,%0,#16\n\t"         \
	       "cmp     %2,#0xff000000\n\t"    \
               "movcs   %2,%2,lsl#8\n\t"       \
               "ldrb    %2,[%3,%2,lsr#24]\n\t" \
               "addcs   %0,%0,#8\n\t"          \
               "add     %0,%0,%2"              \
               : "=r" (result)
               : "0" (result), "r" (input), "r" ((unsigned int)bitarray));   
      return result;
    }
    #define __CNTLZ_ARRAY_BASED(x, y) __CNTLZ__(x, y)
  #endif  /* OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM */
#endif  /* __GNUC__ */

#define OGR_NG_GET_DISPATCH_TABLE_FXN    ogrng_get_dispatch_table_arm1

#include "ansi/ogrng_codebase.cpp"
