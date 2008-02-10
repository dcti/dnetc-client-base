/*
 * Copyright distributed.net 1999-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

const char *ogr_ppc_cpp(void) {
return "@(#)$Id: ogr-ppc.cpp,v 1.7 2008/02/10 00:26:19 kakace Exp $"; }

#if defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)

  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     2 /* 0-2 - '100% asm'      */

  #if defined(HAVE_KOGE_PPC_CORES)
    /*
    ** ASM-optimized OGR cores. Set options that are relevant for ogr_create().
    */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - irrelevant      */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - irrelevant      */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             2 /* 0-2 - '100% asm'      */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - irrelevant      */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - ** MUST BE 1 ** */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

  #elif defined(__MWERKS__)

    #if (__MWERKS__ >= 0x2400)
      #define OGROPT_STRENGTH_REDUCE_CHOOSE       0 /* 0/1 - MWC is better   */
    #else
      #define OGROPT_STRENGTH_REDUCE_CHOOSE       1 /* 0/1 - MWC benefits    */
    #endif
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - 'yes'           */

  #elif defined(__MRC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - MrC is better   */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - MrC is better   */

  #elif defined(__APPLE_CC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - GCC is better   */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - 'yes'           */
    #define OGROPT_NO_FUNCTION_INLINE             1 /* 0/1 - 'yes'           */

  #elif defined(__GNUC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - GCC is better   */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              1 /* 0/1 - 'yes'           */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std'           */
    #if (__GNUC__ >= 3)
      #define OGROPT_NO_FUNCTION_INLINE           1 /* for found_one()       */
    #endif

  #elif defined(__xlC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

  #else
    #error play with the settings to find out optimal settings for your compiler
  #endif


  /*========================================================================*/

  #include "asm-ppc.h"

  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2)
    #if !defined(__CNTLZ__)
      #warning OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM reset to 0
      #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
      #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
    #else
      #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
    #endif
  #endif


  #if (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
    #define COMP_LEFT_LIST_RIGHT(lev, s) {  \
      U temp1, temp2, temp3;                \
      int ss = 32 - (s);                    \
      comp0 <<= s;                          \
      temp1 = newbit << ss;                 \
      temp2 = list0 << ss;                  \
      list0 >>= s;                          \
      temp3 = list1 << ss;                  \
      list1 >>= s;                          \
      list0 |= temp1;                       \
      temp1 = list2 << ss;                  \
      list2 >>= s;                          \
      list1 |= temp2;                       \
      temp2 = list3 << ss;                  \
      list3 >>= s;                          \
      list2 |= temp3;                       \
      temp3 = comp1 >> ss;                  \
      list4 >>= s;                          \
      list3 |= temp1;                       \
      temp1 = comp2 >> ss;                  \
      comp1 <<= s;                          \
      list4 |= temp2;                       \
      temp2 = comp3 >> ss;                  \
      comp2 <<= s;                          \
      comp0 |= temp3;                       \
      temp3 = comp4 >> ss;                  \
      comp3 <<= s;                          \
      comp1 |= temp1;                       \
      comp2 |= temp2;                       \
      comp4 <<= s;                          \
      comp3 |= temp3;                       \
      newbit = 0;                           \
    }
  #endif


  #include "ansi/ogr.cpp"

  #if !defined(OGR_BITMAPS_LENGTH) || (OGR_BITMAPS_LENGTH != 160)
  #error OGR_BITMAPS_LENGTH must be 160 !!!
  #endif

  /*
  ** Check the settings again since we have to make sure ogr_create()
  ** produces compatible datas.
  */
  #if defined(HAVE_KOGE_PPC_CORES) && (OGROPT_HAVE_OGR_CYCLE_ASM == 2) \
    && (OGROPT_ALTERNATE_CYCLE == 1)

    #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
      #error KOGE core is not time-constrained
    #endif

    #ifdef __cplusplus
    extern "C" {
    #endif
    int cycle_ppc_scalar(void *state, int *pnodes, const unsigned char *choose,
                         const int *OGR);
    #ifdef __cplusplus
    }
    #endif

    static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
    {
      with_time_constraints = with_time_constraints;
      return cycle_ppc_scalar(state, pnodes, &choose(0,0), OGR);
    }
  #endif

#else
  #error use this only with ppc since it may contain ppc assembly
#endif
