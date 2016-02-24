/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of eight 32-bit scalars.
 *
 * $Id: ogrng-32.cpp,v 1.3 2008/10/27 10:14:11 oliver Exp $
*/

#include "ansi/ogrng-32.h"

//------------------------ PLATFORM-SPECIFIC SETTINGS ------------------------

#if defined(__PPC__) || defined(__POWERPC__) || (CLIENT_CPU == CPU_PPC)
  #include "ppc/asm-ppc.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && defined(__CNTLZ__)
    #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
  #endif
#elif (CLIENT_CPU == CPU_X86)
  #include "x86/asm-x86.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#elif (CLIENT_CPU == CPU_AMD64)
  #include "amd64/asm-amd64.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#elif (CLIENT_CPU == CPU_68K)
  #include "68k/asm-68k.h"
  #if defined(__CNTLZ)
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 2 /* 0-2 - '100% asm       */
  #elif defined(__CNTLZ_ARRAY_BASED)
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 1 /* 0-2 - 'Partial asm    */
  #else
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0 /* 0-2 - 'no'  (default) */
  #endif
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#elif (CLIENT_CPU == CPU_ARM)
  #define __CNTLZ(x) (1+__builtin_clzl(~(x)))
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - memory-based    */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#else
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#endif


/*
** Define the name of the dispatch table.
** Each core shall define a unique name.
*/
#if !defined(OGR_NG_GET_DISPATCH_TABLE_FXN)
  #define OGR_NG_GET_DISPATCH_TABLE_FXN    ogrng_get_dispatch_table
#endif


#include "ansi/ogrng_codebase.cpp"
