/*
 * Copyright distributed.net 1999-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the "list", "dist" and "comp" bitmap is made of
 *  one 32-bit scalar part (left side), and one 128-bit vector part, thus the
 *  "hybrid" name.
*/

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ogr-vec.cpp,v 1.3.4.12 2004/08/18 18:37:47 kakace Exp $"; }

#if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */

  #define OGROPT_ALTERNATE_CYCLE                  2 /* 0-2 - ** MUST BE 2 ** */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     2 /* 0-2 - '100% asm'      */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT   0 /* 0/1 - irrelevant      */

  #if defined(HAVE_KOGE_PPC_CORES)
    /*
    ** ASM-optimized OGR cores. Set options that are relevant for ogr_create().
    */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             2 /* 0-2 - "100% asm"      */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - irrelevant      */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - irrelevant      */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - irrelevant      */

  #elif defined(__MWERKS__)

    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'            */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - irrelevant      */

  #elif defined(__MRC__)

    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'            */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - irrelevant      */

  #elif defined(__APPLE_CC__)

    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'            */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes'           */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'            */
    #define OGROPT_NO_FUNCTION_INLINE             1 /* 0/1 - 'yes'           */

  #elif defined(__GNUC__)

    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'            */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
    #define OGROPT_CYCLE_CACHE_ALIGN              1 /* 0/1 - 'yes'           */
    #if (__GNUC__ >= 3)
      #define OGROPT_NO_FUNCTION_INLINE           1 /* for found_one()       */
    #endif

  #else
    #error play with the settings to find out optimal settings for your compiler
  #endif

  #define OGR_GET_DISPATCH_TABLE_FXN    vec_ogr_get_dispatch_table


  /*========================================================================*/

  #include <stddef.h>       /* offsetof() */
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


  #if (OGROPT_ALTERNATE_CYCLE == 2)

    #define OGROPT_OGR_CYCLE_ALTIVEC 1    /* For use in ogr.h */

    #if defined (__GNUC__) && !defined(__APPLE_CC__) && (__GNUC__ >= 3)
      #include <altivec.h>
      #define ONES_DECL   (vector unsigned int) {~0u}
    #else
      #define ONES_DECL   (vector unsigned int) (~0u)
    #endif

    typedef vector unsigned char v8_t;
    typedef vector unsigned int v32_t;


    /* define the local variables used for the top recursion state */
    #define SETUP_TOP_STATE(lev)                            \
      v8_t Varray[32];                                      \
      U comp0, dist0, list0;                                \
      vector unsigned int compV, listV, distV;              \
      vector unsigned int zeroes = vec_splat_u32(0);        \
      vector unsigned int ones = ONES_DECL;                 \
      listV = lev->listV.v;                                 \
      distV = lev->distV.v;                                 \
      compV = lev->compV.v;                                 \
      list0 = lev->list0;                                   \
      dist0 = lev->dist0;                                   \
      comp0 = lev->comp0;                                   \
      size_t listOff = offsetof(struct Level, list0);       \
      if ((listOff&15) != 12) return CORE_E_INTERNAL;       \
      int newbit = 1;                                       \
      { /* Initialize Varray[] */                           \
        vector unsigned char val = vec_splat_u8(0);         \
        vector unsigned char one = vec_splat_u8(1);         \
        int i;                                              \
        for (i = 0; i < 32; i++) {                          \
          Varray[i] = val;                                  \
          val = vec_add(val, one);                          \
        }                                                   \
      }
          
    /* shift the list to add or extend the first mark */
    #define COMP_LEFT_LIST_RIGHT(lev, s)                    \
    {                                                       \
      U comp1;                                              \
      v32_t Vs, Vss, listV1, bmV;                           \
      int ss = 32 - (s);                                    \
      Vs = (v32_t) Varray[s];                               \
      list0 >>= s;                                          \
      newbit <<= ss;                                        \
      listV1 = vec_lde(listOff, (U *)lev);                  \
      list0 |= newbit;                                      \
      comp1 = lev->compV.u[0];                              \
      comp0 <<= s;                                          \
      lev->list0 = list0;                                   \
      compV = vec_slo(compV, (v8_t)Vs);                     \
      Vss = vec_sub(zeroes, Vs);                            \
      comp1 >>= ss;                                         \
      bmV = vec_sl(ones, Vs);                               \
      listV1 = vec_sld(listV1, listV, 12);                  \
      comp0 |= comp1;                                       \
      compV = vec_sll(compV, (v8_t)Vs);                     \
      listV = vec_sel(listV1, listV, bmV);                  \
      listV = vec_rl(listV, Vss);                           \
      lev->compV.v = compV;                                 \
      newbit = 0;                                           \
    }

    /*
    ** shift by word size
    */
    #define COMP_LEFT_LIST_RIGHT_32(lev)                    \
      list0 = newbit;                                       \
      v32_t listV1 = vec_lde(listOff, (U *)lev);            \
      lev->list0 = newbit;                                  \
      compV = vec_sld(compV, zeroes, 4);                    \
      comp0 = lev->compV.u[0];                              \
      listV = vec_sld(listV1, listV, 12);                   \
      lev->compV.v = compV;                                 \
      newbit = 0;

    /* set the current mark and push a level to start a new mark */
    #define PUSH_LEVEL_UPDATE_STATE(lev)                    \
      (lev+1)->list0 = list0;                               \
      distV = vec_or(distV, listV);                         \
      lev->comp0 = comp0;                                   \
      dist0 |= list0;                                       \
      compV = vec_or(compV, distV);                         \
      lev->listV.v = listV;                                 \
      newbit = 1;                                           \
      (lev+1)->compV.v = compV;                             \
      comp0 |= dist0;

    /* pop a level to continue work on previous mark */
    #define POP_LEVEL(lev)                                  \
      listV = lev->listV.v;                                 \
      list0 = lev->list0;                                   \
      comp0 = lev->comp0;                                   \
      distV = vec_andc(distV, listV);                       \
      dist0 &= ~list0;                                      \
      compV = lev->compV.v;                                 \
      newbit = 0;

    /* save the local state variables */
    #define SAVE_FINAL_STATE(lev)                           \
      lev->listV.v = listV;                                 \
      lev->distV.v = distV;                                 \
      lev->compV.v = compV;                                 \
      lev->list0 = list0;                                   \
      lev->dist0 = dist0;                                   \
      lev->comp0 = comp0;

  #endif  /* ALTERNATE_CYCLE == 2 */


  #include "ansi/ogr.cpp"

  #if !defined(BITMAPS_LENGTH) || (BITMAPS_LENGTH != 160)
  #error BITMAPS_LENGTH must be 160 !!!
  #endif


  /*
  ** Check the settings again since we have to make sure ogr_create()
  ** produces compatible datas.
  */
  #if defined(HAVE_KOGE_PPC_CORES) && (OGROPT_HAVE_OGR_CYCLE_ASM == 2) \
    && (OGROPT_ALTERNATE_CYCLE == 2)

    #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
      #error KOGE core is not time-constrained
    #endif

    #ifdef __cplusplus
    extern "C" {
    #endif
    int cycle_ppc_hybrid(void *state, int *pnodes, const unsigned char *choose,
                         const int *OGR);
    #ifdef __cplusplus
    }
    #endif

    static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
    {
      with_time_constraints = with_time_constraints;
      return cycle_ppc_hybrid(state, pnodes, &choose(0,0), OGR);
    }
  #endif

  /*========================================================================*/

#else //__VEC__
  #error do you really want to use AltiVec without compiler support?
#endif //__VEC__
