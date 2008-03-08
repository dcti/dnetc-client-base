/*
 * Copyright distributed.net 1999-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

#include "ansi/ogrp2-32.h"

const char *ogr_ppc_cpp(void) {
return "@(#)$Id: ogr-ppc.cpp,v 1.9 2008/03/08 21:11:38 kakace Exp $"; }

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
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - 'std' (default) */

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
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - 'yes'           */

  #elif defined(__MRC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - MrC is better   */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - MrC is better   */

  #elif defined(__APPLE_CC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - GCC is better   */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - 'yes'           */
    #define OGROPT_NO_FUNCTION_INLINE             1 /* 0/1 - 'yes'           */

  #elif defined(__GNUC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - GCC is better   */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              1 /* 0/1 - 'yes'           */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - 'std'           */
    #if (__GNUC__ >= 3)
      #define OGROPT_NO_FUNCTION_INLINE           1 /* for found_one()       */
    #endif

  #elif defined(__xlC__)

    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0-2 - 'yes'           */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - 'std' (default) */

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


  #if (OGROPT_ALTERNATE_CYCLE == 1) 
    /*
     ** Initialize top state
     */
      #undef  SETUP_TOP_STATE
      #define SETUP_TOP_STATE(lev)  \
        SCALAR comp0 = lev->comp[0],     \
               comp1 = lev->comp[1],     \
               comp2 = lev->comp[2],     \
               comp3 = lev->comp[3],     \
               comp4 = lev->comp[4];     \
        SCALAR list0 = lev->list[0],     \
               list1 = lev->list[1],     \
               list2 = lev->list[2],     \
               list3 = lev->list[3],     \
               list4 = lev->list[4];     \
        SCALAR dist0 = lev->dist[0],     \
               dist1 = lev->dist[1],     \
               dist2 = lev->dist[2],     \
               dist3 = lev->dist[3],     \
               dist4 = lev->dist[4];     \
        int newbit = 1;


    /*
     ** Shift COMP and LIST bitmaps (default implementation)
     */
    #undef COMP_LEFT_LIST_RIGHT
    #if (PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 1)
      #define COMP_LEFT_LIST_RIGHT(lev, s) {  \
        SCALAR temp1, temp2, temp3;           \
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
    #else
      #define COMP_LEFT_LIST_RIGHT(lev, s) {  \
        SCALAR temp1, temp2;                  \
        int ss = 32 - (s);                    \
        comp0 <<= s;                          \
        temp1 = newbit << ss;                 \
        temp2 = list0 << ss;                  \
        list0 = (list0 >> (s)) | temp1;       \
        temp1 = list1 << ss;                  \
        list1 = (list1 >> (s)) | temp2;       \
        temp2 = list2 << ss;                  \
        list2 = (list2 >> (s)) | temp1;       \
        temp1 = list3 << ss;                  \
        list3 = (list3 >> (s)) | temp2;       \
        temp2 = comp1 >> ss;                  \
        list4 = (list4 >> (s)) | temp1;       \
        temp1 = comp2 >> ss;                  \
        comp0 |= temp2;                       \
        temp2 = comp3 >> ss;                  \
        comp1 = (comp1 << (s)) | temp1;       \
        temp1 = comp4 >> ss;                  \
        comp2 = (comp2 << (s)) | temp2;       \
        comp4 = comp4 << (s);                 \
        comp3 = (comp3 << (s)) | temp1;       \
        newbit = 0;                           \
      }
    #endif

    /*
     ** Shift COMP and LIST bitmaps by 32 (default implementation)
     */
    #undef  COMP_LEFT_LIST_RIGHT_WORD
    #define COMP_LEFT_LIST_RIGHT_WORD(lev)  \
      list4 = list3;                        \
      list3 = list2;                        \
      list2 = list1;                        \
      list1 = list0;                        \
      list0 = newbit;                       \
      comp0 = comp1;                        \
      comp1 = comp2;                        \
      comp2 = comp3;                        \
      comp3 = comp4;                        \
      comp4 = 0;                            \
      newbit = 0;

    /*
     ** Push level state. Update LIST and COMP bitmaps
     */
    #undef  PUSH_LEVEL_UPDATE_STATE
    #define PUSH_LEVEL_UPDATE_STATE(lev)    \
      lev->list[0] = list0; dist0 |= list0; \
      lev->list[1] = list1; dist1 |= list1; \
      lev->list[2] = list2; dist2 |= list2; \
      lev->list[3] = list3; dist3 |= list3; \
      lev->list[4] = list4; dist4 |= list4; \
      lev->comp[0] = comp0; comp0 |= dist0; \
      lev->comp[1] = comp1; comp1 |= dist1; \
      lev->comp[2] = comp2; comp2 |= dist2; \
      lev->comp[3] = comp3; comp3 |= dist3; \
      lev->comp[4] = comp4; comp4 |= dist4; \
      newbit = 1;

    /*
     ** Pop level state (all bitmaps).
     */
    #undef  POP_LEVEL
    #define POP_LEVEL(lev)  \
      list0 = lev->list[0]; \
      list1 = lev->list[1]; \
      list2 = lev->list[2]; \
      list3 = lev->list[3]; \
      list4 = lev->list[4]; \
      dist0 &= ~list0;      \
      comp0 = lev->comp[0]; \
      dist1 &= ~list1;      \
      comp1 = lev->comp[1]; \
      dist2 &= ~list2;      \
      comp2 = lev->comp[2]; \
      dist3 &= ~list3;      \
      comp3 = lev->comp[3]; \
      dist4 &= ~list4;      \
      comp4 = lev->comp[4]; \
      newbit = 0;

    /*
     ** Save final state (all bitmaps)
     */
    #undef  SAVE_FINAL_STATE
    #define SAVE_FINAL_STATE(lev) \
      lev->list[0] = list0;       \
      lev->list[1] = list1;       \
      lev->list[2] = list2;       \
      lev->list[3] = list3;       \
      lev->list[4] = list4;       \
      lev->dist[0] = dist0;       \
      lev->dist[1] = dist1;       \
      lev->dist[2] = dist2;       \
      lev->dist[3] = dist3;       \
      lev->dist[4] = dist4;       \
      lev->comp[0] = comp0;       \
      lev->comp[1] = comp1;       \
      lev->comp[2] = comp2;       \
      lev->comp[3] = comp3;       \
      lev->comp[4] = comp4;

  #endif  /* OGROPT_ALTERNATE_CYCLE */


  /*
  ** Define the name of the dispatch table.
  */
  #define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table

  #include "ansi/ogrp2_codebase.cpp"


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
