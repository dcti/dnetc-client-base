/*
 * Copyright distributed.net 2002-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ev4.cpp,v 1.1.2.2 2003/02/25 12:30:23 snake Exp $"; }

#if defined(__GNUC__)
  #define OGROPT_BITOFLIST_DIRECT_BIT           1 /* seems to be a win */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* nope              */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* nope              */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* we have no ctlz   */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* nope              */
#else /* Compaq CC */
  #include <c_asm.h>
  #define OGROPT_BITOFLIST_DIRECT_BIT           1 /* yep               */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        1 /* yep               */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 1 /* yep               */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* we have no ctlz   */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* nope              */
#endif

#define ALPHA_CIX
#define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_cix
#define OVERWRITE_DEFAULT_OPTIMIZATIONS

#include "ansi/ogr.cpp"
