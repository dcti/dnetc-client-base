/*
 * Copyright distributed.net 1999-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ogr-vec.cpp,v 1.3.4.3 2003/04/04 14:31:52 gavin Exp $"; }

#if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */
  #if (__MWERKS__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* MWC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                2 /* AltiVec support! */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #elif (__MRC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* MrC is better    */
    #define OGROPT_ALTERNATE_CYCLE                2 /* AltiVec support! */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* MrC is better    */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #elif (__APPLE_CC__)//GCC with exclusive ppc, mach-o and ObjC extensions
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* ACC does benefit */
    #define OGROPT_ALTERNATE_CYCLE                2 /* AltiVec support! */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* ACC is better    */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* no balignl       */
  #elif (__GNUC__)
    #define OGROPT_BITOFLIST_DIRECT_BIT           0 /* 'no' irrelevant  */
    #define OGROPT_COPY_LIST_SET_BIT_JUMPS        0 /* 'no' irrelevant  */
    #define OGROPT_FOUND_ONE_FOR_SMALL_DATA_CACHE 0 /* 'no' irrelevant  */
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   1 /* we have cntlzw   */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* GCC is better    */
    #define OGROPT_ALTERNATE_CYCLE                2 /* AltiVec support! */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 2 /* use switch_asm   */
    #define OGROPT_CYCLE_CACHE_ALIGN              1
  #else
    #error play with the settins to find out optimal settings for your compiler
  #endif
  #define OGR_GET_DISPATCH_TABLE_FXN vec_ogr_get_dispatch_table
  #define OVERWRITE_DEFAULT_OPTIMIZATIONS
  #include "ansi/ogr.cpp"
#else //__VEC__
  #error do you really want to use AltiVec without compiler support?
#endif //__VEC__
