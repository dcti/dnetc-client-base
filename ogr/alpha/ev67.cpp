/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ev67.cpp,v 1.1.2.1 2002/03/29 08:51:54 sampo Exp $"; }

#if defined(__GCC__) || defined(__GNUC__)
  #define ALPHA_CIX  /* to turn on CTLZ optimization in ogr.cpp */

  #define OGROPT_BITOFLIST_DIRECT_BIT           1 /* 'no' irrelevant  */
  #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
  #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /*  we have ctlz    */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /*  does benefit ?  */

  #define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_cix
  #define OVERWRITE_DEFAULT_OPTIMIZATIONS
  #include "ansi/ogr.cpp"
#else //__GCC__
  #error depends on gcc inline assembly.
#endif //
