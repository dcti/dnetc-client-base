/*
 * Copyright distributed.net 2001-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr.cpp for all m68k processors
*/

#if defined(mc68060)
const char *ogr_68060_cpp(void)
#elif defined(mc68040)
const char *ogr_68040_cpp(void)
#elif defined(mc68030)
const char *ogr_68030_cpp(void)
#elif defined(mc68020)
const char *ogr_68020_cpp(void)
#else
const char *ogr_68000_cpp(void)
#endif
{ return "@(#)$Id: ogr-68k.cpp,v 1.2 2007/10/22 16:48:28 jlawson Exp $"; }

/*
** The following macro is defined on the command line for each target CPU
** #define OGR_GET_DISPATCH_TABLE_FXN
*/

// For reference purpose :
// #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
// #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
// #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */

#define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
#define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
#define OGROPT_ALTERNATE_CYCLE                0 /* 0-2 - 'no'  (default) */
#define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

#if defined(ASM_68K)
  #if !defined(mc68040)   /* 68000/020/030/060 */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0  /* GCC is better */
  #endif
#endif

#include "asm-68k.h"

#if defined(__BALIGN)
  #define OGROPT_CYCLE_CACHE_ALIGN                1   /* balign       */
#else
  #define OGROPT_CYCLE_CACHE_ALIGN                0   /* no balign    */
#endif

#if defined(__CNTLZ)
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     2  /* Full asm support    */
#elif defined(__CNTLZ_ARRAY_BASED)
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     1  /* Partial asm support */
#endif

#include "ansi/ogr.cpp"
