/*
 * Copyright distributed.net 2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ev67.cpp,v 1.2 2002/09/02 00:35:47 andreasb Exp $"; }

#if defined(__GNUC__)
  #define OGROPT_BITOFLIST_DIRECT_BIT           1 /* seems to be a win */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* nope              */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* nope              */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have ctlz      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* nope              */
#else /* Compaq CC */
  #include <c_asm.h>
  #define OGROPT_BITOFLIST_DIRECT_BIT           1 /* yep               */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        1 /* yep               */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 1 /* yep               */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have ctlz      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* nope              */
#endif

#define ALPHA_CIX
#define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_cix
#define OVERWRITE_DEFAULT_OPTIMIZATIONS

#include "ansi/ogr.cpp"
