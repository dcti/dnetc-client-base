/*
 * Copyright distributed.net 1999-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ogr-ppc.cpp,v 1.5 2003/11/01 14:20:15 mweiser Exp $"; }

#if defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)
  #if (__MWERKS__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* MWC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* ppc optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
  #elif (__MRC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* MrC is better    */
    #define OGROPT_ALTERNATE_CYCLE                1 /* ppc optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* MrC is better    */
  #elif (__APPLE_CC__)//GCC with exclusive ppc, mach-o and ObjC extensions
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* ACC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* ppc optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* ACC is better    */
  #elif (__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* GCC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                1 /* ppc optimized    */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* GCC is better    */
  #else
    #error play with the settins to find out optimal settings for your compiler
  #endif
  #define OGR_GET_DISPATCH_TABLE_FXN ppc_ogr_get_dispatch_table
  #define OVERWRITE_DEFAULT_OPTIMIZATIONS
  #include "ansi/ogr.cpp"
#else
  #error use this only with ppc since it may contain ppc assembly
#endif
