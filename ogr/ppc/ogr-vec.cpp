/*
 * Copyright distributed.net 1999-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

const char *ogr_vec_cpp(void) {
return "@(#)$Id: ogr-vec.cpp,v 1.3.4.8 2004/07/13 22:18:35 kakace Exp $"; }

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
  #define OGR_P2_GET_DISPATCH_TABLE_FXN vec_ogr_p2_get_dispatch_table
  #define OVERWRITE_DEFAULT_OPTIMIZATIONS

  #if (defined(__GNUC__) && !defined(__APPLE_CC__) && (__GNUC__ >= 3))
    #include <altivec.h>
  #endif

  typedef vector unsigned char V8_t;
  typedef vector unsigned int  V32_t;

  #if (defined(__GNUC__) && !defined(__APPLE_CC__) && (__GNUC__ >= 3))
    #define ZEROBIT_DECL     {0, 0, 1, 0}
    #define SHIFT_BY_1_DECL  {1}
    #define SHIFT_BY_2_DECL  {2}
    #define SHIFT_BY_3_DECL  {3}
    #define SHIFT_BY_4_DECL  {4}
    #define SHIFT_BY_5_DECL  {5}
    #define SHIFT_BY_6_DECL  {6}
    #define SHIFT_BY_7_DECL  {7}
    #define SHIFT_BY_8_DECL  {8}
    #define SHIFT_BY_16_DECL {16}
    #define SHIFT_BY_24_DECL {24}
    #define SHIFT_BY_32_DECL {32}
    #define ZEROS_DECL       {0}
  #else
    #define ZEROBIT_DECL     (0, 0, 1, 0)
    #define SHIFT_BY_1_DECL  (1)
    #define SHIFT_BY_2_DECL  (2)
    #define SHIFT_BY_3_DECL  (3)
    #define SHIFT_BY_4_DECL  (4)
    #define SHIFT_BY_5_DECL  (5)
    #define SHIFT_BY_6_DECL  (6)
    #define SHIFT_BY_7_DECL  (7)
    #define SHIFT_BY_8_DECL  (8)
    #define SHIFT_BY_16_DECL (16)
    #define SHIFT_BY_24_DECL (24)
    #define SHIFT_BY_32_DECL (32)
    #define ZEROS_DECL       (0)
  #endif

   /* define the local variables used for the top recursion state */
  #define SETUP_TOP_STATE(state,lev)                                  \
     vector unsigned int compV, listV0, listV1, distV;                \
     vector unsigned char VSHIFT_B1 =  (V8_t) SHIFT_BY_1_DECL;        \
     vector unsigned char VSHIFT_B2 =  (V8_t) SHIFT_BY_2_DECL;        \
     vector unsigned char VSHIFT_B3 =  (V8_t) SHIFT_BY_3_DECL;        \
     vector unsigned char VSHIFT_B4 =  (V8_t) SHIFT_BY_4_DECL;        \
     vector unsigned char VSHIFT_B5 =  (V8_t) SHIFT_BY_5_DECL;        \
     vector unsigned char VSHIFT_B6 =  (V8_t) SHIFT_BY_6_DECL;        \
     vector unsigned char VSHIFT_B7 =  (V8_t) SHIFT_BY_7_DECL;        \
     vector unsigned int VSHIFT_L1  = (V32_t) SHIFT_BY_1_DECL;        \
     vector unsigned int VSHIFT_L2  = (V32_t) SHIFT_BY_2_DECL;        \
     vector unsigned int VSHIFT_L3  = (V32_t) SHIFT_BY_3_DECL;        \
     vector unsigned int VSHIFT_L4  = (V32_t) SHIFT_BY_4_DECL;        \
     vector unsigned int VSHIFT_L5  = (V32_t) SHIFT_BY_5_DECL;        \
     vector unsigned int VSHIFT_L6  = (V32_t) SHIFT_BY_6_DECL;        \
     vector unsigned int VSHIFT_L7  = (V32_t) SHIFT_BY_7_DECL;        \
     vector unsigned int VSHIFT_L8  = (V32_t) SHIFT_BY_8_DECL;        \
     vector unsigned int VSHIFT_L16 = (V32_t) SHIFT_BY_16_DECL;       \
     vector unsigned int VSHIFT_L24 = (V32_t) SHIFT_BY_24_DECL;       \
     vector unsigned int VSHIFT_L32 = (V32_t) SHIFT_BY_32_DECL;       \
     U comp0, dist0, list0;                                           \
     vector unsigned int ZEROBIT = (V32_t) ZEROBIT_DECL;              \
     vector unsigned int ZEROS = (V32_t) ZEROS_DECL;                  \
     vector unsigned int ONES = vec_nor(ZEROS, ZEROS);                \
     distV  = state->distV.v;                                         \
     dist0  = state->dist0;                                           \
     compV  = lev->compV.v;                                           \
     listV0 = vec_or(lev->listV0.v, ZEROBIT);                         \
     listV1 = lev->listV1.v;                                          \
     comp0  = lev->comp0;                                             \
     list0  = lev->list0;                                             \
     int cnt2 = lev->cnt2;                                            \
     int newbit = 1;                                                  \
     int limit;

  #if defined(__GNUC__) && (defined(__PPC__) || defined(__POWERPC__))
  # define __rlwinm(Rs,SH,MB,ME) ({ \
     int Ra; \
     __asm__ volatile ("rlwinm %0,%1,%2,%3,%4" : \
     "=r" (Ra) : "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); Ra; })

  # define __rlwnm(Rs,rB,MB,ME) ({ \
     int Ra; \
     __asm__ volatile ("rlwnm %0,%1,%2,%3,%4" : \
     "=r" (Ra) : "r" (Rs), "r" (rB), "n" (MB), "n" (ME)); Ra; })

  # define __rlwimi(Ra,Rs,SH,MB,ME) ({ \
     __asm__ volatile ("rlwimi %0,%2,%3,%4,%5" : \
     "=r" (Ra) : "0" (Ra), "r" (Rs), "n" (SH), "n" (MB), "n" (ME)); Ra; })

  #define __nop ({ __asm__ volatile ("nop");})

  /*
  ** Use "__asm__ volatile" equivalent functions instead of vec_xxx()
  ** built-in functions to prevent the compiler from moving instructions
  ** around (gcc suxx)
  */

  // vec_sld
  #define __vsldoi(Va, Vb, NB) ({                         \
    vector unsigned int Vt;                               \
    __asm__ volatile ("vsldoi %0,%1,%2,%3" : "=v" (Vt) :  \
      "v" (Va), "v" (Vb), "n" (NB));                      \
      Vt;                                                 \
    })

  // vec_sll
  #define __vsl(Va, Vb) ({                                \
    vector unsigned int Vt;                               \
    __asm__ volatile ("vsl %0,%1,%2" : "=v" (Vt) :        \
      "v" (Va), "v" (Vb));                                \
      Vt;                                                 \
    })

  // vec_srl
  #define __vsr(Va, Vb) ({                          \
    vector unsigned int Vt;                         \
    __asm__ volatile ("vsr %0,%1,%2" : "=v" (Vt) :  \
      "v" (Va), "v" (Vb));                          \
      Vt;                                           \
    })

  // vec_slo
  #define __vslo(Va, Vb) ({                         \
    vector unsigned int Vt;                         \
    __asm__ volatile ("vslo %0,%1,%2" : "=v" (Vt) : \
      "v" (Va), "v" (Vb));                          \
      Vt;                                           \
    })

  // vec_sro
  #define __vsro(Va, Vb) ({                         \
    vector unsigned int Vt;                         \
    __asm__ volatile ("vsro %0,%1,%2" : "=v" (Vt) : \
      "v" (Va), "v" (Vb));                          \
      Vt;                                           \
    })

  // vec_sl
  #define __vslw(Va, Vb) ({                         \
    vector unsigned int Vt;                         \
    __asm__ volatile ("vslw %0,%1,%2" : "=v" (Vt) : \
      "v" (Va), "v" (Vb));                          \
      Vt;                                           \
    })

  // vec_rl
  #define __vrlw(Va, Vb) ({                         \
    vector unsigned int Vt;                         \
    __asm__ volatile ("vrlw %0,%1,%2" : "=v" (Vt) : \
      "v" (Va), "v" (Vb));                          \
      Vt;                                           \
    })

  // vec_sel
  #define __vsel(Va, Vb, Vc) ({                         \
    vector unsigned int Vt;                             \
    __asm__ volatile ("vsel %0,%1,%2,%3" : "=v" (Vt) :  \
      "v" (Va), "v" (Vb), "v" (Vc));                    \
      Vt;                                               \
    })

  // vec_add
  #define __vadduwm(Va, Vb) ({                          \
    vector unsigned int Vt;                             \
    __asm__ volatile ("vadduwm %0,%1,%2" : "=v" (Vt) :  \
      "v" (Va), "v" (Vb));                              \
      Vt;                                               \
    })

  // vec_sub
  #define __vsubuwm(Va, Vb) ({                          \
    vector unsigned int Vt;                             \
    __asm__ volatile ("vsubuwm %0,%1,%2" : "=v" (Vt) :  \
      "v" (Va), "v" (Vb));                              \
      Vt;                                               \
    })

  // vec_or
  #define __vor(Va, Vb) ({                          \
    vector unsigned int Vt;                         \
    __asm__ volatile ("vor %0,%1,%2" : "=v" (Vt) :  \
      "v" (Va), "v" (Vb));                          \
      Vt;                                           \
    })

  #define __stvx(Va, addr) ({                       \
    __asm__ volatile ("stvx %0,0,%1" : : "v" (Va), "r" (addr));  \
  })

  #endif

  /*
  ** shift the list to add or extend the first mark
  ** Each basic bloc should have a length that is an even multiple of 16.
  ** NOP instructions are used to pad uneven blocks and to align them
  ** since GCC is unable to do it right...
  */

  #define COMP_LEFT_LIST_RIGHT(lev, s)                                \
  {                                                                   \
     vector unsigned int tempV;                                       \
     vector unsigned int maskV;                                       \
     vector unsigned int shiftV;                                      \
     U comp1;                                                         \
     void *p;                                                         \
     comp1 = lev->compV.u[0];                                         \
     comp0 = __rlwnm(comp0, s, 0, 31);                                \
     p = &lev->compV.v;                                               \
     switch (s)                                                       \
     {                                                                \
        case 0:                                                       \
          __nop;                                                      \
          __nop;                                                      \
          break;                                                      \
        case 1:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L1);                           \
          compV  = __vsl(compV, VSHIFT_B1);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L1);                       \
          list0  = __rlwinm(list0, 32-1, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B1);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 1, 32-1, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-1, 0, 1-1);             \
          break;                                                      \
        case 2:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L2);                           \
          compV  = __vsl(compV, VSHIFT_B2);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L2);                       \
          list0  = __rlwinm(list0, 32-2, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B2);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 2, 32-2, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-2, 0, 2-1);             \
          break;                                                      \
        case 3:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L3);                           \
          compV  = __vsl(compV, VSHIFT_B3);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L3);                       \
          list0  = __rlwinm(list0, 32-3, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B3);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 3, 32-3, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-3, 0, 3-1);             \
          break;                                                      \
        case 4:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L4);                           \
          compV  = __vsl(compV, VSHIFT_B4);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L4);                       \
          list0  = __rlwinm(list0, 32-4, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B4);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 4, 32-4, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-4, 0, 4-1);             \
          break;                                                      \
        case 5:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L5);                           \
          compV  = __vsl(compV, VSHIFT_B5);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L5);                       \
          list0  = __rlwinm(list0, 32-5, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B5);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 5, 32-5, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-5, 0, 5-1);             \
          break;                                                      \
        case 6:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L6);                           \
          compV  = __vsl(compV, VSHIFT_B6);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L6);                       \
          list0  = __rlwinm(list0, 32-6, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B6);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 6, 32-6, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-6, 0, 6-1);             \
          break;                                                      \
        case 7:                                                       \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, VSHIFT_L7);                           \
          compV  = __vsl(compV, VSHIFT_B7);                           \
          shiftV = __vsubuwm(ZEROS, VSHIFT_L7);                       \
          list0  = __rlwinm(list0, 32-7, 0, 31);                      \
          listV0 = __vsr(listV0, VSHIFT_B7);                          \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          comp0  = __rlwimi(comp0, comp1, 7, 32-7, 31);               \
          __stvx(compV, p);                                           \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-7, 0, 7-1);             \
          break;                                                      \
        case 8:                                                       \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          list0  = __rlwinm(list0, 24, 0, 31);                        \
          listV1 = __vsldoi(listV0, listV1, 15);                      \
          comp0  = __rlwimi(comp0, comp1, 8, 24, 31);                 \
          __stvx(compV, p);                                           \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          list0  = __rlwimi(list0, newbit, 24, 0, 7);                 \
          break;                                                      \
        case 9:                                                       \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L1);                   \
          list0  = __rlwinm(list0, 32-(8+1), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+1), 32-(8+1), 31);       \
          compV  = __vsl(compV, VSHIFT_B1);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B1);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+1), 0, (8+1)-1);     \
          break;                                                      \
        case 10:                                                      \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L2);                   \
          list0  = __rlwinm(list0, 32-(8+2), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+2), 32-(8+2), 31);       \
          compV  = __vsl(compV, VSHIFT_B2);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B2);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+2), 0, (8+2)-1);     \
          break;                                                      \
        case 11:                                                      \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L3);                   \
          list0  = __rlwinm(list0, 32-(8+3), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+3), 32-(8+3), 31);       \
          compV  = __vsl(compV, VSHIFT_B3);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B3);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+3), 0, (8+3)-1);     \
          break;                                                      \
        case 12:                                                      \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L4);                   \
          list0  = __rlwinm(list0, 32-(8+4), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+4), 32-(8+4), 31);       \
          compV  = __vsl(compV, VSHIFT_B4);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B4);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+4), 0, (8+4)-1);     \
          break;                                                      \
        case 13:                                                      \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L5);                   \
          list0  = __rlwinm(list0, 32-(8+5), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+5), 32-(8+5), 31);       \
          compV  = __vsl(compV, VSHIFT_B5);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B5);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+5), 0, (8+5)-1);     \
          break;                                                      \
        case 14:                                                      \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L6);                   \
          list0  = __rlwinm(list0, 32-(8+6), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+6), 32-(8+6), 31);       \
          compV  = __vsl(compV, VSHIFT_B6);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B6);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+6), 0, (8+6)-1);     \
          break;                                                      \
        case 15:                                                      \
          compV  = __vslo(compV, VSHIFT_L8);                          \
          shiftV = __vadduwm(VSHIFT_L8, VSHIFT_L7);                   \
          list0  = __rlwinm(list0, 32-(8+7), 0, 31);                  \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (8+7), 32-(8+7), 31);       \
          compV  = __vsl(compV, VSHIFT_B7);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L8);                         \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B7);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(8+7), 0, (8+7)-1);     \
          break;                                                      \
        case 16:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          list0  = __rlwinm(list0, 16, 0, 31);                        \
          listV1 = __vsldoi(listV0, listV1, 14);                      \
          comp0  = __rlwimi(comp0, comp1, 16, 16, 31);                \
          __stvx(compV, p);                                           \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          list0  = __rlwimi(list0, newbit, 16, 0, 15);                \
          break;                                                      \
        case 17:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L1);                  \
          list0  = __rlwinm(list0, 32-(16+1), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+1), 32-(16+1), 31);     \
          compV  = __vsl(compV, VSHIFT_B1);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B1);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+1), 0, (16+1)-1);   \
          break;                                                      \
        case 18:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L2);                  \
          list0  = __rlwinm(list0, 32-(16+2), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+2), 32-(16+2), 31);     \
          compV  = __vsl(compV, VSHIFT_B2);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B2);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+2), 0, (16+2)-1);   \
          break;                                                      \
        case 19:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L3);                  \
          list0  = __rlwinm(list0, 32-(16+3), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+3), 32-(16+3), 31);     \
          compV  = __vsl(compV, VSHIFT_B3);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B3);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+3), 0, (16+3)-1);   \
          break;                                                      \
        case 20:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L4);                  \
          list0  = __rlwinm(list0, 32-(16+4), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+4), 32-(16+4), 31);     \
          compV  = __vsl(compV, VSHIFT_B4);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B4);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+4), 0, (16+4)-1);   \
          break;                                                      \
        case 21:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L5);                  \
          list0  = __rlwinm(list0, 32-(16+5), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+5), 32-(16+5), 31);     \
          compV  = __vsl(compV, VSHIFT_B5);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B5);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+5), 0, (16+5)-1);   \
          break;                                                      \
        case 22:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L6);                  \
          list0  = __rlwinm(list0, 32-(16+6), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+6), 32-(16+6), 31);     \
          compV  = __vsl(compV, VSHIFT_B6);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B6);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+6), 0, (16+6)-1);   \
          break;                                                      \
        case 23:                                                      \
          compV  = __vslo(compV, VSHIFT_L16);                         \
          shiftV = __vadduwm(VSHIFT_L16, VSHIFT_L7);                  \
          list0  = __rlwinm(list0, 32-(16+7), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (16+7), 32-(16+7), 31);     \
          compV  = __vsl(compV, VSHIFT_B7);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L16);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B7);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(16+7), 0, (16+7)-1);   \
          break;                                                      \
        case 24:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          list0  = __rlwinm(list0, 8, 0, 31);                         \
          listV1 = __vsldoi(listV0, listV1, 13);                      \
          comp0  = __rlwimi(comp0, comp1, 24, 8, 31);                 \
          __stvx(compV, p);                                           \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          list0  = __rlwimi(list0, newbit, 8, 0, 23);                 \
          break;                                                      \
        case 25:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L1);                  \
          list0  = __rlwinm(list0, 32-(24+1), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+1), 32-(24+1), 31);     \
          compV  = __vsl(compV, VSHIFT_B1);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B1);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+1), 0, (24+1)-1);   \
          break;                                                      \
        case 26:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L2);                  \
          list0  = __rlwinm(list0, 32-(24+2), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+2), 32-(24+2), 31);     \
          compV  = __vsl(compV, VSHIFT_B2);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B2);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+2), 0, (24+2)-1);   \
          break;                                                      \
        case 27:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L3);                  \
          list0  = __rlwinm(list0, 32-(24+3), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+3), 32-(24+3), 31);     \
          compV  = __vsl(compV, VSHIFT_B3);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B3);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+3), 0, (24+3)-1);   \
          break;                                                      \
        case 28:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L4);                  \
          list0  = __rlwinm(list0, 32-(24+4), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+4), 32-(24+4), 31);     \
          compV  = __vsl(compV, VSHIFT_B4);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B4);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+4), 0, (24+4)-1);   \
          break;                                                      \
        case 29:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L5);                  \
          list0  = __rlwinm(list0, 32-(24+5), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+5), 32-(24+5), 31);     \
          compV  = __vsl(compV, VSHIFT_B5);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B5);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+5), 0, (24+5)-1);   \
          break;                                                      \
        case 30:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L6);                  \
          list0  = __rlwinm(list0, 32-(24+6), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+6), 32-(24+6), 31);     \
          compV  = __vsl(compV, VSHIFT_B6);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B6);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+6), 0, (24+6)-1);   \
          break;                                                      \
        case 31:                                                      \
          compV  = __vslo(compV, VSHIFT_L24);                         \
          shiftV = __vadduwm(VSHIFT_L24, VSHIFT_L7);                  \
          list0  = __rlwinm(list0, 32-(24+7), 0, 31);                 \
          tempV  = __vsldoi(listV0, listV1, 12);                      \
          maskV  = __vslw(ONES, shiftV);                              \
          comp0  = __rlwimi(comp0, comp1, (24+7), 32-(24+7), 31);     \
          compV  = __vsl(compV, VSHIFT_B7);                           \
          shiftV = __vsubuwm(ZEROS, shiftV);                          \
          __nop;                                                      \
          listV0 = __vsro(listV0, VSHIFT_L24);                        \
          listV1 = __vsel(tempV, listV1, maskV);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsr(listV0, VSHIFT_B7);                          \
          listV1 = __vrlw(listV1, shiftV);                            \
          list0  = __rlwimi(list0, newbit, 32-(24+7), 0, (24+7)-1);   \
          break;                                                      \
        case 32:                                                      \
          comp0  = lev->compV.u[0];                                   \
          compV  = __vslo(compV, VSHIFT_L32);                         \
          list0  = newbit;                                            \
          listV1 = __vsldoi(listV0, listV1, 12);                      \
          __stvx(compV, p);                                           \
          listV0 = __vsldoi(ZEROS, listV0, 12);                       \
          break;                                                      \
     }                                                                \
     newbit = 0;                                                      \
  }

  /* shift by word size */
  #define COMP_LEFT_LIST_RIGHT_32(lev)                                \
     comp0  = lev->compV.u[0];                                        \
     compV  = __vslo(compV, VSHIFT_L32);                              \
     list0  = newbit;                                                 \
     listV1 = __vsldoi(listV0, listV1, 12);                           \
     newbit = 0;                                                      \
     lev->compV.v = compV;                                            \
     listV0 = __vsldoi(ZEROS, listV0, 12);

  /* set the current mark and push a level to start a new mark */
  #define PUSH_LEVEL_UPDATE_STATE(lev)                                \
     lev->listV0.v = listV0;                                          \
     distV = __vor(distV, listV1);                                    \
     lev->listV1.v = listV1;                                          \
     compV = __vor(compV, distV);                                     \
     lev->list0 = list0;                                              \
     dist0 |= list0;                                                  \
     (lev+1)->compV.v = compV;                                        \
     listV0 = __vor(listV0, ZEROBIT);                                 \
     lev->comp0 = comp0;                                              \
     comp0 |= dist0;                                                  \
     lev->cnt2 = cnt2;                                                \
     newbit = 1;                                                      \
     lev->limit = limit;

  /* pop a level to continue work on previous mark */
  #define POP_LEVEL(lev)                                              \
     listV1 = lev->listV1.v;                                          \
     comp0 = lev->comp0;                                              \
     newbit = 0;                                                      \
     list0 = lev->list0;                                              \
     compV = lev->compV.v;                                            \
     listV0 = lev->listV0.v;                                          \
     distV = vec_andc(distV, listV1);                                 \
     limit = lev->limit;                                              \
     cnt2 = lev->cnt2;                                                \
     dist0 = dist0 & ~list0;

  /* save the local state variables */
  #define SAVE_FINAL_STATE(state,lev)                                 \
     lev->listV0.v = listV0;                                          \
     lev->listV1.v = listV1;                                          \
     state->distV.v = distV;                                          \
     lev->compV.v = compV;                                            \
     lev->list0 = list0;                                              \
     state->dist0 = dist0;                                            \
     lev->comp0 = comp0;                                              \
     lev->cnt2 = cnt2;

  #include "ansi/ogr.cpp"
#else //__VEC__
  #error do you really want to use AltiVec without compiler support?
#endif //__VEC__
