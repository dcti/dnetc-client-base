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
return "@(#)$Id: ogr-vec.cpp,v 1.3.4.11 2004/08/18 10:49:01 piru Exp $"; }

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
      #define BYTE_VECTOR_DEF(x) (vector unsigned char) {x}
      #define ONES_DECL   (vector unsigned int) {~0u}
      
    #else
      #define BYTE_VECTOR_DEF(x) (vector unsigned char) (x)
      #define ONES_DECL   (vector unsigned int) (~0u)
    #endif

    typedef vector unsigned char v8_t;
    typedef vector unsigned int v32_t;

    /*
    ** Shift counts.
    ** It's much faster to load the corresponding vector (Varray[s]) from
    ** memory (data cache) than to convert an integer to a splatted vector.
    */
    #if !defined (__GNUC__) || (__GNUC__ >= 3)
    static const v8_t Varray[32] = {
      BYTE_VECTOR_DEF(0),
      BYTE_VECTOR_DEF(1),
      BYTE_VECTOR_DEF(2),
      BYTE_VECTOR_DEF(3),
      BYTE_VECTOR_DEF(4),
      BYTE_VECTOR_DEF(5),
      BYTE_VECTOR_DEF(6),
      BYTE_VECTOR_DEF(7),
      BYTE_VECTOR_DEF(8),
      BYTE_VECTOR_DEF(9),
      BYTE_VECTOR_DEF(10),
      BYTE_VECTOR_DEF(11),
      BYTE_VECTOR_DEF(12),
      BYTE_VECTOR_DEF(13),
      BYTE_VECTOR_DEF(14),
      BYTE_VECTOR_DEF(15),
      BYTE_VECTOR_DEF(16),
      BYTE_VECTOR_DEF(17),
      BYTE_VECTOR_DEF(18),
      BYTE_VECTOR_DEF(19),
      BYTE_VECTOR_DEF(20),
      BYTE_VECTOR_DEF(21),
      BYTE_VECTOR_DEF(22),
      BYTE_VECTOR_DEF(23),
      BYTE_VECTOR_DEF(24),
      BYTE_VECTOR_DEF(25),
      BYTE_VECTOR_DEF(26),
      BYTE_VECTOR_DEF(27),
      BYTE_VECTOR_DEF(28),
      BYTE_VECTOR_DEF(29),
      BYTE_VECTOR_DEF(30),
      BYTE_VECTOR_DEF(31)
    };
    #else
    /*
    ** gcc 2.95.3-altivec doesn't seem to like this, and uses altivec insts
    ** in static constructor breaking things if there's no altivec or altivec
    ** is disabled. Workaround with having the data in aligned char array.
    */
    static const unsigned char Varray_data[32 * 16] __attribute__ ((aligned (16))) = {
      #define UCHAR_VECTOR_DEF(x) x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x
      UCHAR_VECTOR_DEF(0),
      UCHAR_VECTOR_DEF(1),
      UCHAR_VECTOR_DEF(2),
      UCHAR_VECTOR_DEF(3),
      UCHAR_VECTOR_DEF(4),
      UCHAR_VECTOR_DEF(5),
      UCHAR_VECTOR_DEF(6),
      UCHAR_VECTOR_DEF(7),
      UCHAR_VECTOR_DEF(8),
      UCHAR_VECTOR_DEF(9),
      UCHAR_VECTOR_DEF(10),
      UCHAR_VECTOR_DEF(11),
      UCHAR_VECTOR_DEF(12),
      UCHAR_VECTOR_DEF(13),
      UCHAR_VECTOR_DEF(14),
      UCHAR_VECTOR_DEF(15),
      UCHAR_VECTOR_DEF(16),
      UCHAR_VECTOR_DEF(17),
      UCHAR_VECTOR_DEF(18),
      UCHAR_VECTOR_DEF(19),
      UCHAR_VECTOR_DEF(20),
      UCHAR_VECTOR_DEF(21),
      UCHAR_VECTOR_DEF(22),
      UCHAR_VECTOR_DEF(23),
      UCHAR_VECTOR_DEF(24),
      UCHAR_VECTOR_DEF(25),
      UCHAR_VECTOR_DEF(26),
      UCHAR_VECTOR_DEF(27),
      UCHAR_VECTOR_DEF(28),
      UCHAR_VECTOR_DEF(29),
      UCHAR_VECTOR_DEF(30),
      UCHAR_VECTOR_DEF(31)
    };
    const v8_t *Varray = (const v8_t *) Varray_data;
    #endif

    /* define the local variables used for the top recursion state */
    #define SETUP_TOP_STATE(lev)                            \
      U comp0, dist0, list0;                                \
      vector unsigned int compV, listV, distV;              \
      vector unsigned int zeroes = (v32_t)Varray[0];        \
      vector unsigned int ones = ONES_DECL;                 \
      listV = lev->listV.v;                                 \
      distV = lev->distV.v;                                 \
      compV = lev->compV.v;                                 \
      list0 = lev->list0;                                   \
      dist0 = lev->dist0;                                   \
      comp0 = lev->comp0;                                   \
      size_t listOff = offsetof(struct Level, list0);       \
      if ((listOff&15) != 12) return CORE_E_INTERNAL;       \
      int newbit = 1;

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
                         const int *OGR, const vector unsigned char *array);
    #ifdef __cplusplus
    }
    #endif

    static int ogr_cycle(void *state, int *pnodes, int with_time_constraints)
    {
      with_time_constraints = with_time_constraints;
      return cycle_ppc_hybrid(state, pnodes, &choose(0,0), OGR,
                              (const vector unsigned char *)Varray);
    }
  #endif

  /*========================================================================*/

#else //__VEC__
  #error do you really want to use AltiVec without compiler support?
#endif //__VEC__
