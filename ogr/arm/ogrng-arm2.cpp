/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Derived from ansi/ogrng-32.cpp
 *
 * $Id: ogrng-arm2.cpp,v 1.1 2008/12/28 23:20:52 teichp Exp $
*/

#include "ansi/ogrng-32.h"

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - 'no'  (default) */
#define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */

#if defined(__GNUC__)
  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2)
    static __inline__ int __CNTLZ__(register unsigned int input)
    {
      register int result;
      __asm__ ("clz     %0,%1"              \
               : "=r" (result)
               : "r" (input));   
      return result;
    }
    #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
  #endif  /* OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM */
#endif  /* __GNUC__ */

#define OGR_NG_GET_DISPATCH_TABLE_FXN    ogrng_get_dispatch_table_arm2

#include "ansi/ogrng_codebase.cpp"
