/*
 * Copyright distributed.net 2002-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of one 32-bit scalar (left side), and two 64-bit scalars, so that the
 * bitmaps precision strictly matches the regular 32-bit core. The leftmost
 * 32 bits (dist[0] and comp[0]) are don't care bits. However, list0 is
 * handled as a 64-bit datatype, which helps to store the otherwise implied
 * bit that determines the position of the mark being worked on. Shifting
 * list0 then achieves the same result as or'ing "newbit" as this is done in
 * other cores.
 * Beside, the OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT setting selects a
 * memory based implementation (0), or a register based implementation (1).
*/

const char *ogr64_cpp(void) {
return "@(#)$Id: ogr-64.cpp,v 1.1.2.4 2004/08/10 18:38:05 jlawson Exp $"; }

#include <stddef.h>
#include "cputypes.h"

#ifndef HAVE_I64
#error fixme: your compiler does not appear to support 64-bit datatypes
#endif


#define OGROPT_ALTERNATE_CYCLE                  2 /* 0-2 - ** MUST BE 2 ** */

#if defined(__PPC__) || defined(__POWERPC__) || (CLIENT_CPU == CPU_PPC)
  #include "ppc/asm-ppc.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - register-based  */

  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2)
    #if !defined(__CNTLZ__)
      #warning OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM reset to 0
      #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
      #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
    #else
      #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
    #endif
  #endif
#elif (CLIENT_CPU == CPU_X86)
  // this 64-bit version does not actually seem to be a benefit on x86.
  #include "x86/asm-x86.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
    #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
    #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
  #endif
#elif (CLIENT_CPU == CPU_AMD64)
  // this 64-bit version does not actually seem to be a benefit on x86.
  #include "amd64/asm-amd64.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - register based */

  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
    #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
    #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
  #endif
#else
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */
#endif


#define OGR_GET_DISPATCH_TABLE_FXN    ogr64_get_dispatch_table


#if (OGROPT_ALTERNATE_CYCLE == 2)
  
  #define OGROPT_64BIT_IMPLEMENTATION 1   /* for use in ogr.h */

  #if (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
    /*
    ** Initialize top state
    ** comp0 and dist0 must be 32-bit
    */
    #define SETUP_TOP_STATE(lev)            \
      U comp1, comp2;                       \
      U list0, list1, list2;                \
      U dist1, dist2;                       \
      u32 comp0 = (u32) lev->comp[0];       \
      u32 dist0 = (u32) lev->dist[0];       \
      comp1 = lev->comp[1];                 \
      comp2 = lev->comp[2];                 \
      list0 = lev->list[0] | ((U)1 << 32);  \
      list1 = lev->list[1];                 \
      list2 = lev->list[2];                 \
      dist1 = lev->dist[1];                 \
      dist2 = lev->dist[2];

    /*
    ** Shift COMP and LIST bitmaps
    ** comp0 and dist0 are 32-bit values
    */
    #define COMP_LEFT_LIST_RIGHT(lev, s) {  \
      U temp1, temp2;                       \
      register int ss = 64 - (s);           \
      temp2 = list0 << ss;                  \
      list0 = list0 >> (s);                 \
      temp1 = list1 << ss;                  \
      list1 = (list1 >> (s)) | temp2;       \
      temp2 = comp1 >> ss;                  \
      list2 = (list2 >> (s)) | temp1;       \
      temp1 = comp2 >> ss;                  \
      comp0 = (u32) ((comp0 << (s)) | temp2);       \
      comp1 = (comp1 << (s)) | temp1;       \
      comp2 = (comp2 << (s));               \
    }

    /*
    ** Shift COMP and LIST bitmaps by 32
    ** comp0 and dist0 are 32-bit values
    */
    #define COMP_LEFT_LIST_RIGHT_32(lev)      \
      list2 = (list2 >> 32) | (list1 << 32);  \
      list1 = (list1 >> 32) | (list0 << 32);  \
      list0 >>= 32;                           \
      comp0 = (u32) (comp1 >> 32);            \
      comp1 = (comp1 << 32) | (comp2 >> 32);  \
      comp2 <<= 32;

    /*
    ** Update state then go deeper
    ** comp0 and dist0 are 32-bit values
    */
    #define PUSH_LEVEL_UPDATE_STATE(lev)    \
      lev->list[0] = list0; dist0 |= list0; \
      lev->list[1] = list1; dist1 |= list1; \
      lev->list[2] = list2; dist2 |= list2; \
      lev->comp[0] = comp0; comp0 |= dist0; \
      lev->comp[1] = comp1; comp1 |= dist1; \
      lev->comp[2] = comp2; comp2 |= dist2; \
      list0 |= ((U)1 << 32);

    /*
    ** Pop level state (all bitmaps).
    ** comp0 is a 32-bit value
    */
    #define POP_LEVEL(lev)    \
      list0 = lev->list[0];   \
      list1 = lev->list[1];   \
      list2 = lev->list[2];   \
      dist0 &= ~list0;        \
      dist1 &= ~list1;        \
      dist2 &= ~list2;        \
      comp0 = (u32) lev->comp[0];   \
      comp1 = lev->comp[1];   \
      comp2 = lev->comp[2];

    /*
    ** Save final state (all bitmaps)
    */
    #define SAVE_FINAL_STATE(lev)   \
      lev->list[0] = list0;         \
      lev->list[1] = list1;         \
      lev->list[2] = list2;         \
      lev->dist[0] = dist0;         \
      lev->dist[1] = dist1;         \
      lev->dist[2] = dist2;         \
      lev->comp[0] = comp0;         \
      lev->comp[1] = comp1;         \
      lev->comp[2] = comp2;

  #else /* OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0 */
    /*
    ** Initialize top state
    ** comp0 and dist0 must be 32-bit
    */
    #define SETUP_TOP_STATE(lev)    \
      u32 comp0 = (u32) lev->comp[0];     \
      u32 dist0 = (u32) lev->dist[0];     \
      lev->list[0] |= ((U)1 << 32);

    /*
    ** Shift COMP and LIST bitmaps
    ** comp0 and dist0 are 32-bit values
    */
    #define __COMP_LEFT_LIST_RIGHT(lev, s, ss) {              \
      U temp1, temp2;                                         \
      temp2 = lev->list[0] << (ss);                           \
      lev->list[0] = (lev->list[0] >> (s));                   \
      temp1 = lev->list[1] << (ss);                           \
      lev->list[1] = (lev->list[1] >> (s)) | temp2;           \
      temp2 = lev->comp[1] >> (ss);                           \
      lev->list[2] = (lev->list[2] >> (s)) | temp1;           \
      temp1 = lev->comp[2] >> (ss);                           \
      comp0 = (u32) (lev->comp[0] = (lev->comp[0] << (s)) | temp2); \
      lev->comp[1] = (lev->comp[1] << (s)) | temp1;           \
      lev->comp[2] = lev->comp[2] << (s);                     \
    }

    #define COMP_LEFT_LIST_RIGHT(lev, s) {    \
      register int ss = 64 - (s);             \
      __COMP_LEFT_LIST_RIGHT(lev, s, ss);     \
    }

    /*
    ** Shift COMP and LIST bitmaps by 32
    ** comp0 and dist0 are 32-bit values
    */
    #define COMP_LEFT_LIST_RIGHT_32(lev)    \
      __COMP_LEFT_LIST_RIGHT(lev, 32, 32);

    /*
    ** Update state then go deeper
    ** comp0 and dist0 are 32-bit values
    */
    #define PUSH_LEVEL_UPDATE_STATE(lev) {            \
      struct Level *lev2 = lev + 1;                   \
      U temp = lev->list[0];                          \
      lev2->list[0] = temp | ((U)1 << 32);            \
      dist0 = (u32) (lev2->dist[0] = lev->dist[0] | temp);  \
      lev2->comp[0] = (comp0 |= dist0);               \
      temp = (lev2->list[1] = lev->list[1]);          \
      temp = (lev2->dist[1] = lev->dist[1] | temp);   \
      lev2->comp[1] = lev->comp[1] | temp;            \
      temp = (lev2->list[2] = lev->list[2]);          \
      temp = (lev2->dist[2] = lev->dist[2] | temp);   \
      lev2->comp[2] = lev->comp[2] | temp;            \
    }

    /*
    ** Pop level state (all bitmaps).
    ** comp0 is a 32-bit value
    */
    #define POP_LEVEL(lev) comp0 = (u32) lev->comp[0];

    /*
    ** Save final state (all bitmaps)
    */
    #define SAVE_FINAL_STATE(lev)          /* nothing */

  #endif  /* OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 0 */
#endif    /* OGROPT_ALTERNATE_CYCLE == 2 */


#include "ansi/ogr.cpp"

#if !defined(BITMAPS_LENGTH) || (BITMAPS_LENGTH != 160)
#error BITMAPS_LENGTH must be 160 !!!
#endif
