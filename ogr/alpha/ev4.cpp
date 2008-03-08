/*
 * Copyright distributed.net 2002-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 */

#include "ansi/ogrp2-32.h"

#error Fixme: Obsolete file. Merged into ogrp2-32.cpp


const char *ogr_ev4_cpp(void) {
return "@(#)$Id: ev4.cpp,v 1.4 2008/03/08 20:18:28 kakace Exp $"; }

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

#define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table_ev4

#include "baseincs.h" //for endian detection
#include "alpha/alpha-asm.h"
#include "ansi/ogrp2_codebase.cpp"
