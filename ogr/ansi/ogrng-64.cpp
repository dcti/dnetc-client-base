/*
 * Copyright distributed.net 2002-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: ogrng-64.cpp,v 1.8 2009/09/17 20:16:00 andreasb Exp $
*/

#include "ansi/ogrng-64.h"


//------------------------ PLATFORM-SPECIFIC SETTINGS ------------------------

#if defined(__PPC__) || defined(__POWERPC__) || (CLIENT_CPU == CPU_PPC)
  //#include "ppc/asm-ppc.h"  /* <== FIXME : Doesn't work in 64-bit mode */
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */

  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && defined(__CNTLZ__)
    #define __CNTLZ(x) (__CNTLZ__(~(x))+1)
  #endif
#elif (CLIENT_CPU == CPU_X86)
  #include "x86/asm-x86.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - default         */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#elif (CLIENT_CPU == CPU_SPARC)
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#elif (CLIENT_CPU == CPU_AMD64)
  #include "amd64/asm-amd64.h"
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#else
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - memory-based    */
  #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'default'       */
#endif


//----------------------------------------------------------------------------
/*
** Fixup __CNTLZ
** Assembly code doesn't work with 64-bit datatypes implemented by using two or
** more registers. If we're not compiling for a native 64-bit architecture,
** then the __CNTLZ macro is discarded.
*/
#if !defined(__x86_64__) && !defined(__amd64__) && !defined(_M_AMD64) && \
	!defined(__ppc64__) && !defined(__alpha__) && !defined(__sparc_v9__)
  #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2)
    #undef  __CNTLZ
    #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM 0
    #warning 32-bit architectures are not supported - OGROPT_HAVE_FFZ reset to 0.
  #endif
#endif


//----------------------------------------------------------------------------

#if (PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 1)
  /*
  ** Initialize top state
  */
  #undef  SETUP_TOP_STATE
  #define SETUP_TOP_STATE(lev)                          \
    register BMAP comp0, comp1, comp2, comp3;           \
    register BMAP list0, list1, list2, list3;           \
    BMAP dist0, dist1, dist2, dist3;                    \
    BMAP newbit = (depth < oState->maxdepthm1) ? 1 : 0; \
    comp0 = lev->comp[0];                               \
    comp1 = lev->comp[1];                               \
    comp2 = lev->comp[2];                               \
    comp3 = lev->comp[3];                               \
    list0 = lev->list[0];                               \
    list1 = lev->list[1];                               \
    list2 = lev->list[2];                               \
    list3 = lev->list[3];                               \
    dist0 = lev->dist[0];                               \
    dist1 = lev->dist[1];                               \
    dist2 = lev->dist[2];                               \
    dist3 = lev->dist[3];

  /*
  ** Shift COMP and LIST bitmaps
  */
  #undef  COMP_LEFT_LIST_RIGHT
  #define COMP_LEFT_LIST_RIGHT(lev, s) {          \
    BMAP temp1, temp2;                            \
    register int ss = SCALAR_BITS - (s);          \
    comp0 <<= (s);                                \
    temp1 = newbit << (ss);                       \
    temp2 = list0 << (ss);                        \
    list0 = (list0 >> (s)) | temp1;               \
    temp1 = list1 << (ss);                        \
    list1 = (list1 >> (s)) | temp2;               \
    temp2 = list2 << (ss);                        \
    list2 = (list2 >> (s)) | temp1;               \
    temp1 = comp1 >> (ss);                        \
    list3 = (list3 >> (s)) | temp2;               \
    temp2 = comp2 >> (ss);                        \
    comp0 |= temp1;                               \
    temp1 = comp3 >> (ss);                        \
    comp1 = (comp1 << (s)) | temp2;               \
    comp3 = comp3 << (s);                         \
    comp2 = (comp2 << (s)) | temp1;               \
    newbit = 0;                                   \
    }

  /*
  ** Shift COMP and LIST bitmaps by 32
  */
  #undef  COMP_LEFT_LIST_RIGHT_WORD
  #define COMP_LEFT_LIST_RIGHT_WORD(lev)  \
    comp0 = comp1;                        \
    comp1 = comp2;                        \
    comp2 = comp3;                        \
    comp3 = 0;                            \
    list3 = list2;                        \
    list2 = list1;                        \
    list1 = list0;                        \
    list0 = newbit;                       \
    newbit = 0;

  /*
  ** Update state then go deeper
  */
  #undef  PUSH_LEVEL_UPDATE_STATE
  #define PUSH_LEVEL_UPDATE_STATE(lev)    \
    lev->list[0] = list0; dist0 |= list0; \
    lev->list[1] = list1; dist1 |= list1; \
    lev->list[2] = list2; dist2 |= list2; \
    lev->list[3] = list3; dist3 |= list3; \
    lev->comp[0] = comp0; comp0 |= dist0; \
    lev->comp[1] = comp1; comp1 |= dist1; \
    lev->comp[2] = comp2; comp2 |= dist2; \
    lev->comp[3] = comp3; comp3 |= dist3; \
    newbit = 1;

  /*
  ** Pop level state (all bitmaps).
  */
  #undef  POP_LEVEL
  #define POP_LEVEL(lev)          \
    list0 = lev->list[0];         \
    list1 = lev->list[1];         \
    list2 = lev->list[2];         \
    list3 = lev->list[3];         \
    dist0 &= ~list0;              \
    dist1 &= ~list1;              \
    dist2 &= ~list2;              \
    dist3 &= ~list3;              \
    comp0 = lev->comp[0];         \
    comp1 = lev->comp[1];         \
    comp2 = lev->comp[2];         \
    comp3 = lev->comp[3];         \
    newbit = 0;

  /*
  ** Save final state (all bitmaps)
  */
  #undef  SAVE_FINAL_STATE
  #define SAVE_FINAL_STATE(lev)   \
    lev->list[0] = list0;         \
    lev->list[1] = list1;         \
    lev->list[2] = list2;         \
    lev->list[3] = list3;         \
    lev->dist[0] = dist0;         \
    lev->dist[1] = dist1;         \
    lev->dist[2] = dist2;         \
    lev->dist[3] = dist3;         \
    lev->comp[0] = comp0;         \
    lev->comp[1] = comp1;         \
    lev->comp[2] = comp2;         \
    lev->comp[3] = comp3;

#endif  /* PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 1 */


//----------------------------------------------------------------------------

/*
** Define the name of the dispatch table.
** Each core shall define a unique name.
*/
#if !defined(OGR_NG_GET_DISPATCH_TABLE_FXN)
  #define OGR_NG_GET_DISPATCH_TABLE_FXN    ogrng64_get_dispatch_table
#endif


#include "ansi/ogrng_codebase.cpp"
