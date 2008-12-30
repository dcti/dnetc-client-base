/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

#include "ansi/ogrp2-64.h"

/* Baseline implementation.
 * Due to changes in OGR-P2 code design, the implementations provided by this
 * file and the baseline ogr-64.cpp (renamed to ogrp2-64.cpp) have been swapped.
 */

const char *ogr_ev4_64_cpp(void) {
return "@(#)$Id: ev4-64.cpp,v 1.4 2008/12/30 20:58:43 andreasb Exp $"; }

#if defined(__GNUC__)
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
#else /* Compaq CC */
  #include <c_asm.h>
  #define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE         0 /* 0/1 - 'no'            */
  #define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
  #define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
  #define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */
#endif

#define OGR_GET_DISPATCH_TABLE_FXN    ogr64_get_dispatch_table_ev4

#include "baseincs.h" //for endian detection
#include "alpha/alpha-asm.h"
#include "ansi/ogrp2_codebase.cpp"
