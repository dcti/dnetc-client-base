/*
 * Copyright distributed.net 1999-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/

const char *ogrng_ppc_cpp(void) {
return "@(#)$Id: ogrng-ppc.cpp,v 1.1 2008/02/10 00:11:28 kakace Exp $"; }

#if defined(ASM_PPC) || defined(__PPC__) || defined(__POWERPC__)

  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM     2 /* 0-2 - '100% asm'      */

  #if defined(HAVE_FLEGE_PPC_CORES)
    /*
    ** ASM-optimized OGR cores. Set options that are relevant for ogr_create().
    */
    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - irrelevant      */
    #define OGROPT_ALTERNATE_CYCLE                1 /* 0/1 - '100% asm'      */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */

  #elif defined(__MWERKS__)

    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'no'            */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std'           */

  #elif defined(__MRC__)

    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'no'            */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - MrC is better   */

  #elif defined(__APPLE_CC__)

    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'no'            */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std'           */

  #elif defined(__GNUC__)

    #define OGROPT_CYCLE_CACHE_ALIGN              1 /* 0/1 - 'yes'           */
    #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'no'            */
    #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std'           */

  #elif defined(__xlC__)

    #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
    #define OGROPT_ALTERNATE_CYCLE                0 /* 0/1 - 'no'            */
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
    int cycle_ppc_scalar_256(struct OgrNgState *state, int *pnodes, const u16 *choose);
    #ifdef __cplusplus
    }
    #endif

    static int ogr_cycle_256(struct OgrNgState *oState, int *pnodes,
                             const u16* pchoose, int with_time_constraints)
    {
      with_time_constraints = with_time_constraints;
      return cycle_ppc_scalar_256(state, pnodes, pchoose);
    }
  #endif

#else
  #error use this only with ppc since it may contain ppc assembly
#endif
