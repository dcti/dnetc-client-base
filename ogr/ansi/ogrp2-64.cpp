/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
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
 * Beside, the PRIVATE_ALT_COMP_LEFT_LIST_RIGHT setting selects a
 * memory based implementation (0), or a register based implementation (1).
 *
 * $Id: ogrp2-64.cpp,v 1.1 2008/03/08 20:07:14 kakace Exp $
*/


#include "ansi/ogrp2-64.h"

#ifndef HAVE_I64
#error fixme: your compiler does not appear to support 64-bit datatypes
#endif


#if defined(__PPC__) || defined(__POWERPC__) || (CLIENT_CPU == CPU_PPC)
  #include "ppc/asm-ppc.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */

  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && defined(__CNTLZ__)
    #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
  #endif
#elif (CLIENT_CPU == CPU_X86)
  // this 64-bit version does not actually seem to be a benefit on x86.
  #include "x86/asm-x86.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - 'std' (default) */
#elif (CLIENT_CPU == CPU_SPARC)
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - 'register-based' */
#elif (CLIENT_CPU == CPU_AMD64)
  #include "amd64/asm-amd64.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register based */
#elif (CLIENT_CPU == CPU_ALPHA)
  #include "baseincs.h"     // For endianness detection.
  #if defined(__GNUC__)
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register based */
  #else /* Compaq CC */
    #include <c_asm.h>
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
    #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
    #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
    #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register based */
  #endif
#else
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         1 /* 0/1 - 'yes' (default) */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - 'std' (default) */
#endif


//----------------------------------------------------------------------------

#if (PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 1)
 /*
 ** Initialize top state
 ** comp0 and dist0 must be 32-bit
 */
 #undef  SETUP_TOP_STATE
 #define SETUP_TOP_STATE(lev)             \
   BMAP comp1, comp2;                     \
   BMAP list0, list1, list2;              \
   BMAP dist1, dist2;                     \
   SCALAR comp0 = (u32) lev->comp[0];     \
   SCALAR dist0 = (u32) lev->dist[0];     \
   comp1 = lev->comp[1];                  \
   comp2 = lev->comp[2];                  \
   list0 = lev->list[0] | ((BMAP)1 << 32);\
   list1 = lev->list[1];                  \
   list2 = lev->list[2];                  \
   dist1 = lev->dist[1];                  \
   dist2 = lev->dist[2];

 /*
 ** Shift COMP and LIST bitmaps
 ** comp0 and dist0 are 32-bit values
 */
 #undef  COMP_LEFT_LIST_RIGHT
 #define COMP_LEFT_LIST_RIGHT(lev, s) {      \
   BMAP temp1, temp2;                        \
   register int ss = 64 - (s);               \
   temp2 = list0 << ss;                      \
   list0 = list0 >> (s);                     \
   temp1 = list1 << ss;                      \
   list1 = (list1 >> (s)) | temp2;           \
   temp2 = comp1 >> ss;                      \
   list2 = (list2 >> (s)) | temp1;           \
   temp1 = comp2 >> ss;                      \
   comp0 = (SCALAR) ((comp0 << (s)) | temp2);\
   comp1 = (comp1 << (s)) | temp1;           \
   comp2 = (comp2 << (s));                   \
 }

 /*
 ** Shift COMP and LIST bitmaps by 32
 ** comp0 and dist0 are 32-bit values
 */
 #undef  COMP_LEFT_LIST_RIGHT_WORD
 #define COMP_LEFT_LIST_RIGHT_WORD(lev)    \
   list2 = (list2 >> 32) | (list1 << 32);  \
   list1 = (list1 >> 32) | (list0 << 32);  \
   list0 >>= 32;                           \
   comp0 = (SCALAR) (comp1 >> 32);         \
   comp1 = (comp1 << 32) | (comp2 >> 32);  \
   comp2 <<= 32;

 /*
 ** Update state then go deeper
 ** comp0 and dist0 are 32-bit values
 */
 #undef  PUSH_LEVEL_UPDATE_STATE
 #define PUSH_LEVEL_UPDATE_STATE(lev)    \
   lev->list[0] = list0; dist0 |= list0; \
   lev->list[1] = list1; dist1 |= list1; \
   lev->list[2] = list2; dist2 |= list2; \
   lev->comp[0] = comp0; comp0 |= dist0; \
   lev->comp[1] = comp1; comp1 |= dist1; \
   lev->comp[2] = comp2; comp2 |= dist2; \
   list0 |= ((BMAP)1 << 32);

 /*
 ** Pop level state (all bitmaps).
 ** comp0 is a 32-bit value
 */
 #undef  POP_LEVEL
 #define POP_LEVEL(lev)             \
   list0 = lev->list[0];            \
   list1 = lev->list[1];            \
   list2 = lev->list[2];            \
   dist0 &= ~list0;                 \
   dist1 &= ~list1;                 \
   dist2 &= ~list2;                 \
   comp0 = (SCALAR) lev->comp[0];   \
   comp1 = lev->comp[1];            \
   comp2 = lev->comp[2];

 /*
 ** Save final state (all bitmaps)
 */
 #undef  SAVE_FINAL_STATE
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

#endif  /* PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 1 */

//----------------------------------------------------------------------------

/*
 ** Define the name of the dispatch table.
 ** Each core shall define a unique name.
 */
#if !defined(OGR_GET_DISPATCH_TABLE_FXN)
#define OGR_GET_DISPATCH_TABLE_FXN    ogr64_get_dispatch_table
#endif


#include "ansi/ogrp2_codebase.cpp"
