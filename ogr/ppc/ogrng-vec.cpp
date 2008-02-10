/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the "list", "dist" and "comp" bitmap is made of
 *  one 32-bit scalar part (left side), and two 128-bit vector parts, thus the
 *  "hybrid" name.
*/

const char *ogrng_vec_cpp(void) {
return "@(#)$Id: ogrng-vec.cpp,v 1.1 2008/02/10 18:10:43 kakace Exp $"; }

#if defined(__VEC__) || defined(__ALTIVEC__) /* compiler supports AltiVec */

  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     2 /* 0-2 - '100% asm'      */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT   1 /* 0/1 - ** MUST BE 1 ** */


  #if defined(HAVE_FLEGE_PPC_CORES)
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0/1 - ** MUST BE 1 ** */
  #endif

  #define OGR_NG_GET_DISPATCH_TABLE_FXN    vec_ogrng_get_dispatch_table


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


  #define OGROPT_OGR_CYCLE_ALTIVEC 1    /* For use in ogr-ng.h */

  #if defined (__GNUC__) && !defined(__APPLE_CC__) && (__GNUC__ >= 3)
    #include <altivec.h>
  #endif

  typedef vector unsigned char v8_t;
  typedef vector unsigned int v32_t;


  /* define the local variables used for the top recursion state */
  #define SETUP_TOP_STATE(lev)                            \
    v8_t vecShift[32];                                    \
    U comp0, dist0;                                       \
    v32_t compV0, compV1;                                 \
    v32_t listV0, listV1;                                 \
    v32_t distV0, distV1;                                 \
    v32_t V_ZERO = vec_splat_u32(0);                      \
    v32_t V_ONES = vec_splat_s32(-1);                     \
    v32_t newbit = vec_splat_u32(1);                      \
    listV0 = lev->listV0.v;                               \
    listV1 = lev->listV1.v;                               \
    distV0 = lev->distV0.v;                               \
    distV1 = lev->distV1.v;                               \
    compV0 = lev->compV0.v;                               \
    compV1 = lev->compV1.v;                               \
    dist0 = lev->distV0.u[0];                             \
    comp0 = lev->compV0.u[0];                             \
    { /* Initialize vecShift[] */                         \
      vector unsigned char val = vec_splat_u8(0);         \
      vector unsigned char one = vec_splat_u8(1);         \
      int i;                                              \
      for (i = 0; i < 32; i++) {                          \
        vecShift[i] = val;                                \
        val = vec_add(val, one);                          \
      }                                                   \
    }
          
  /* shift the list to add or extend the first mark */
  #define COMP_LEFT_LIST_RIGHT(lev, s)                    \
  {                                                       \
    v32_t shift_l = (v32_t) vecShift[s];                  \
    v32_t shift_r = vec_sub(V_ZERO, shift_l);             \
    v32_t mask_l  = vec_sr(V_ONES, shift_l);              \
    v32_t mask_r  = vec_sl(V_ONES, shift_l);              \
    v32_t temp1, temp2;                                   \
    temp1 = vec_sld(compV0, compV1, 4);                   \
    temp2 = vec_sld(listV0, listV1, 12);                  \
    temp1 = vec_sel(temp1, compV0, mask_l);               \
    temp2 = vec_sel(temp2, listV1, mask_r);               \
    compV0 = vec_rl(temp1, shift_l);                      \
    listV1 = vec_rl(temp2, shift_r);                      \
    lev->compV0.v = compV0;                               \
    temp2 = vec_sld(newbit, listV0, 12);                  \
    temp1 = vec_slo(compV1, (v8_t) shift_l);              \
    temp2 = vec_sel(temp2, listV0, mask_r);               \
    compV1 = vec_sll(temp1, (v8_t) shift_l);              \
    listV0 = vec_rl(temp2, shift_r);                      \
    newbit = V_ZERO;                                      \
    comp0 = lev->compV0.u[0];                             \
  }

  /*
  ** shift by word size
  */
  #define COMP_LEFT_LIST_RIGHT_32(lev)    \
    compV0 = vec_sld(compV0, compV1, 4);  \
    lev->compV0.v = compV0;               \
    compV1 = vec_sld(compV1, V_ZERO, 4);  \
    listV1 = vec_sld(listV0, listV1, 12); \
    listV0 = vec_sld(newbit, listV0, 12); \
    comp0  = lev->compV0.u[0];            \
    newbit = V_ZERO;

  /* set the current mark and push a level to start a new mark */
  #define PUSH_LEVEL_UPDATE_STATE(lev)    \
    lev->listV0.v = listV0;               \
    distV0 = vec_or(distV0, listV0);      \
    lev->distV0.v = distV0;               \
    compV0 = vec_or(compV0, distV0);      \
    lev->listV1.v = listV1;               \
    distV1 = vec_or(distV1, listV1);      \
    lev->compV1.v = compV1;               \
    compV1 = vec_or(compV1, distV1);      \
    dist0 = lev->distV0.u[0];             \
    newbit = vec_splat_u32(1);            \
    comp0 |= dist0;

  /* pop a level to continue work on previous mark */
  #define POP_LEVEL(lev)                  \
    listV0 = lev->listV0.v;               \
    listV1 = lev->listV1.v;               \
    comp0  = lev->compV0.u[0];            \
    distV0 = vec_andc(distV0, listV0);    \
    distV1 = vec_andc(distV1, listV1);    \
    compV0 = lev->compV0.v;               \
    compV1 = lev->compV1.v;               \
    newbit = V_ZERO;

  /* save the local state variables */
  #define SAVE_FINAL_STATE(lev)           \
    lev->listV0.v = listV0;               \
    lev->distV0.v = distV0;               \
    lev->compV0.v = compV0;               \
    lev->listV1.v = listV1;               \
    lev->distV1.v = distV1;               \
    lev->compV1.v = compV1;


  #include "ansi/ogr-ng.cpp"

  #if !defined(OGRNG_BITMAPS_LENGTH) || (OGRNG_BITMAPS_LENGTH != 256)
  #error OGRNG_BITMAPS_LENGTH must be 256 !!!
  #endif


  /*
  ** Check the settings again since we have to make sure ogr_create()
  ** produces compatible datas.
  */
  #if defined(HAVE_FLEGE_PPC_CORES) && (OGROPT_ALTERNATE_CYCLE == 1)

    #if !defined(OGROPT_IGNORE_TIME_CONSTRAINT_ARG)
      #error FLEGE core is not time-constrained
    #endif

    #ifdef __cplusplus
    extern "C" {
    #endif
    int cycle_ppc_hybrid_256(struct OgrNgState *state, int *pnodes, const u16 *choose);
    #ifdef __cplusplus
    }
    #endif

    static int ogr_cycle_256(struct OgrNgState *oState, int *pnodes,
                             const u16* pchoose, int with_time_constraints)
    {
      with_time_constraints = with_time_constraints;
      return cycle_ppc_hybrid_256(oState, pnodes, pchoose);
    }
  #endif

  /*========================================================================*/

#else //__VEC__
  #error do you really want to use AltiVec without compiler support?
#endif //__VEC__
