/*
 * Copyright distributed.net 2002-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

const char *ogr_ev4_64_cpp(void) {
return "@(#)$Id: ev4-64.cpp,v 1.2 2007/10/22 16:48:28 jlawson Exp $"; }

#if defined(__GNUC__)
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - register based */
#else /* Compaq CC */
  #include <c_asm.h>
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - register based */
#endif

#define OVERWRITE_DEFAULT_OPTIMIZATIONS
#define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table_ev4_64

#include "baseincs.h" //for endian detection
#include "alpha/alpha-asm.h"
#include "ansi/ogr-64.cpp"
