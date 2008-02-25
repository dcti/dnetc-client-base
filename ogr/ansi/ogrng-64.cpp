/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of four 64-bit scalars, so that the bitmaps precision strictly matches
 * the regular 32-bit core.
 * Beside, the OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT setting selects a
 * memory based implementation (0), or a register based implementation (1).
*/


#include <stddef.h>
#include "cputypes.h"

#ifndef HAVE_I64
#error fixme: your compiler does not appear to support 64-bit datatypes
#endif


#define OGROPT_ALTERNATE_CYCLE                  0 /* 0/1 - 'no' (default)  */
#define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT   1 /* 0/1 - ** MUST BE 1 ** */

#define OGROPT_64BIT_IMPLEMENTATION 1   /* for use in ogr-ng.h */


#if !defined(OVERWRITE_DEFAULT_OPTIMIZATIONS) /*should move to arch-specific*/
  #if defined(__PPC__) || defined(__POWERPC__) || (CLIENT_CPU == CPU_PPC)
    #include "ppc/asm-ppc.h"
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */

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
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - memory-based    */

    #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
      #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
      #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
      #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
    #endif
  #elif (CLIENT_CPU == CPU_SPARC)
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */
  #elif (CLIENT_CPU == CPU_AMD64)
    #include "amd64/asm-amd64.h"
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   2 /* 0-2 - '100% asm'      */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      1 /* 0/1 - register-based  */

    #if (OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM == 2) && !defined(__CNTLZ)
      #warning Macro __CNTLZ not defined. OGROPT_FFZ reset to 0.
      #undef  OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM
      #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0
    #endif
  #else
    #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define PRIVATE_ALT_COMP_LEFT_LIST_RIGHT      0 /* 0/1 - memory-based    */
  #endif
#endif


#if !defined(OGR_NG_GET_DISPATCH_TABLE_FXN)
#define OGR_NG_GET_DISPATCH_TABLE_FXN    ogrng64_get_dispatch_table
#endif


#if (OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1)
  
  #if (PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 1)
    /*
    ** Initialize top state
    ** compA and distA must be 32-bit
    */
    #define SETUP_TOP_STATE(lev)                      \
      U compA, compB, compC, compD;                   \
      U listA, listB, listC, listD;                   \
      U distA, distB, distC, distD;                   \
      u32 comp0, dist0;                               \
      U newbit = (depth < oState->maxdepthm1) ? 1 : 0;\
      compA = lev->comp[0];                           \
      compB = lev->comp[1];                           \
      compC = lev->comp[2];                           \
      compD = lev->comp[3];                           \
      listA = lev->list[0];                           \
      listB = lev->list[1];                           \
      listC = lev->list[2];                           \
      listD = lev->list[3];                           \
      distA = lev->dist[0];                           \
      distB = lev->dist[1];                           \
      distC = lev->dist[2];                           \
      distD = lev->dist[3];                           \
      comp0 = (u32) (compA >> 32);                    \
      dist0 = (u32) (distA >> 32);

    /*
    ** Shift COMP and LIST bitmaps
    ** compA and distA are 32-bit values
    */
    #define __COMP_LEFT_LIST_RIGHT(lev, s, ss) {   \
      U temp1, temp2;                              \
      compA <<= (s);                               \
      temp1 = newbit << (ss);                      \
      temp2 = listA << (ss);                       \
      listA = (listA >> (s)) | temp1;              \
      temp1 = listB << (ss);                       \
      listB = (listB >> (s)) | temp2;              \
      temp2 = listC << (ss);                       \
      listC = (listC >> (s)) | temp1;              \
      temp1 = compB >> (ss);                       \
      listD = (listD >> (s)) | temp2;              \
      temp2 = compC >> (ss);                       \
      compA |= temp1;                              \
      temp1 = compD >> (ss);                       \
      compB = (compB << (s)) | temp2;              \
      compD = compD << (s);                        \
      compC = (compC << (s)) | temp1;              \
      newbit = 0;                                  \
      comp0 = (u32) (compA >> 32);                 \
    }

    #define COMP_LEFT_LIST_RIGHT(lev, s) { \
      register int ss = 64 - (s);          \
      __COMP_LEFT_LIST_RIGHT(lev, s, ss)   \
    }

    /*
    ** Shift COMP and LIST bitmaps by 32
    ** compA and distA are 32-bit values
    */
    #define COMP_LEFT_LIST_RIGHT_32(lev)      \
      __COMP_LEFT_LIST_RIGHT(lev, 32, 32)

    /*
    ** Update state then go deeper
    ** compA and distA are 32-bit values
    */
    #define PUSH_LEVEL_UPDATE_STATE(lev)    \
      dist0 |= (u32) (listA >> 32);         \
      lev->list[0] = listA; distA |= listA; \
      lev->list[1] = listB; distB |= listB; \
      lev->list[2] = listC; distC |= listC; \
      lev->list[3] = listD; distD |= listD; \
      lev->comp[0] = compA; compA |= distA; \
      lev->comp[1] = compB; compB |= distB; \
      lev->comp[2] = compC; compC |= distC; \
      lev->comp[3] = compD; compD |= distD; \
      comp0 |= dist0;                       \
      newbit = 1;

    /*
    ** Pop level state (all bitmaps).
    ** compA is a 32-bit value
    */
    #define POP_LEVEL(lev)          \
      listA = lev->list[0];         \
      listB = lev->list[1];         \
      listC = lev->list[2];         \
      listD = lev->list[3];         \
      distA &= ~listA;              \
      distB &= ~listB;              \
      distC &= ~listC;              \
      distD &= ~listD;              \
      compA = lev->comp[0];         \
      compB = lev->comp[1];         \
      compC = lev->comp[2];         \
      compD = lev->comp[3];         \
      dist0 = (u32) (distA >> 32);  \
      comp0 = (u32) (compA >> 32);  \
      newbit = 0;

    /*
    ** Save final state (all bitmaps)
    */
    #define SAVE_FINAL_STATE(lev)   \
      lev->list[0] = listA;         \
      lev->list[1] = listB;         \
      lev->list[2] = listC;         \
      lev->list[3] = listD;         \
      lev->dist[0] = distA;         \
      lev->dist[1] = distB;         \
      lev->dist[2] = distC;         \
      lev->dist[3] = distD;         \
      lev->comp[0] = compA;         \
      lev->comp[1] = compB;         \
      lev->comp[2] = compC;         \
      lev->comp[3] = compD;

  #else /* PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 0 */
    /*
    ** Initialize top state
    ** compA and distA must be 32-bit
    */
    #define SETUP_TOP_STATE(lev)                \
      u32 comp0 = (u32) (lev->comp[0] >> 32);   \
      u32 dist0 = (u32) (lev->dist[0] >> 32);   \
      U newbit = (depth < oState->maxdepthm1) ? 1 : 0;

    /*
    ** Shift COMP and LIST bitmaps
    ** compA and distA are 32-bit values
    */
    #define __COMP_LEFT_LIST_RIGHT(lev, s, ss) {      \
      U temp1, temp2;                                 \
      temp1 = newbit << (ss);                         \
      temp2 = lev->list[0] << (ss);                   \
      lev->list[0] = (lev->list[0] >> (s)) | temp1;   \
      temp1 = lev->list[1] << (ss);                   \
      lev->list[1] = (lev->list[1] >> (s)) | temp2;   \
      temp2 = lev->list[2] << (ss);                   \
      lev->list[2] = (lev->list[2] >> (s)) | temp1;   \
      temp1 = lev->comp[1] >> (ss);                   \
      lev->list[3] = (lev->list[3] >> (s)) | temp2;   \
      temp2 = lev->comp[2] >> (ss);                   \
      lev->comp[0] = (lev->comp[0] << (s)) | temp1;   \
      temp1 = lev->comp[3] >> (ss);                   \
      lev->comp[1] = (lev->comp[1] << (s)) | temp2;   \
      lev->comp[3] = lev->comp[3] << (s);             \
      lev->comp[2] = (lev->comp[2] << (s)) | temp1;   \
      comp0 = (u32) (lev->comp[0] >> 32);             \
      newbit = 0;                                     \
    }

    #define COMP_LEFT_LIST_RIGHT(lev, s) { \
      register int ss = 64 - (s);          \
      __COMP_LEFT_LIST_RIGHT(lev, s, ss);  \
    }

    /*
    ** Shift COMP and LIST bitmaps by 32
    ** compA and distA are 32-bit values
    */
    #define COMP_LEFT_LIST_RIGHT_32(lev)    \
      __COMP_LEFT_LIST_RIGHT(lev, 32, 32);

    /*
    ** Update state then go deeper
    ** compA and distA are 32-bit values
    */
    #define PUSH_LEVEL_UPDATE_STATE(lev) {            \
      struct OgrLevel *lev2 = lev + 1;                \
      U temp;                                         \
      temp = (lev2->list[0] = lev->list[0]);          \
      temp = (lev2->dist[0] = lev->dist[0] | temp);   \
      dist0 |= (u32) (temp >> 32);                    \
      lev2->comp[0] = (lev->comp[0] | temp);          \
      comp0 |= dist0;                                 \
      temp  = (lev2->list[1] = lev->list[1]);         \
      temp  = (lev2->dist[1] = lev->dist[1] | temp);  \
      lev2->comp[1] = lev->comp[1] | temp;            \
      temp  = (lev2->list[2] = lev->list[2]);         \
      temp  = (lev2->dist[2] = lev->dist[2] | temp);  \
      lev2->comp[2] = lev->comp[2] | temp;            \
      temp  = (lev2->list[3] = lev->list[3]);         \
      temp  = (lev2->dist[3] = lev->dist[3] | temp);  \
      lev2->comp[3] = lev->comp[3] | temp;            \
      newbit = 1;                                     \
    }

    /*
    ** Pop level state (all bitmaps).
    ** compA is a 32-bit value
    */
    #define POP_LEVEL(lev)                \
      comp0 = (u32) (lev->comp[0] >> 32); \
      dist0 = (u32) (lev->dist[0] >> 32); \
      newbit = 0;

    /*
    ** Save final state (all bitmaps)
    */
    #define SAVE_FINAL_STATE(lev)          /* nothing */

  #endif  /* PRIVATE_ALT_COMP_LEFT_LIST_RIGHT == 0 */
#endif    /* OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT == 1 */


#include "ansi/ogr-ng.cpp"

#if !defined(OGRNG_BITMAPS_LENGTH) || (OGRNG_BITMAPS_LENGTH != 256)
#error OGRNG_BITMAPS_LENGTH must be 256 !!!
#endif
