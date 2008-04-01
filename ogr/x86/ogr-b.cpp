/*
 * Copyright distributed.net 2001-2004 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr.cpp for all processor WITHOUT a fast bsr instruction.
 * (ie, 386, 486, Pentium, P4, K5, K6, K7, Cyrix(all), etc)
 *
 * $Id: ogr-b.cpp,v 1.7 2008/04/01 14:42:19 stream Exp $
*/

#define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table_nobsr

#define OGROPT_HAVE_FIND_FIRST_ZERO_BIT_ASM   0 /* 0-2 - 'no'  (default) */
#ifdef __WATCOMC__
  /* "imul reg,constant" look faster */
  #define OGROPT_STRENGTH_REDUCE_CHOOSE       0 /* 0/1 - 'yes' (default) */
#else
  #define OGROPT_STRENGTH_REDUCE_CHOOSE       1 /* 0/1 - 'yes' (default) */
#endif

#define OGROPT_NO_FUNCTION_INLINE             0 /* 0/1 - 'no'  (default) */
#define OGROPT_HAVE_OGR_CYCLE_ASM             0 /* 0-2 - 'no'  (default) */
#define OGROPT_CYCLE_CACHE_ALIGN              0 /* 0/1 - 'no'  (default) */

#ifdef __WATCOMC__
  /* Have assembly routine */
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 1 /* 0/1 - 'std' (default) */
#else
  #define OGROPT_ALTERNATE_COMP_LEFT_LIST_RIGHT 0 /* 0/1 - 'std' (default) */
#endif

#include "asm-x86-p2.h"
#include "ansi/ogrp2-32.h"
#include "ansi/ogrp2_codebase.cpp"
