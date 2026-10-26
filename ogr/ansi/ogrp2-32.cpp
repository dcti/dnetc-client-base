/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of five 32-bit scalars.
 *
 * $Id: ogrp2-32.cpp,v 1.1 2008/03/08 20:07:14 kakace Exp $
*/

#include "ansi/ogrp2-32.h"

#if (CLIENT_CPU == CPU_ALPHA)
  #includ "baseincs.h"    // For endianness detection.
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
#endif

/*
** Define the name of the dispatch table.
** Each core shall define a unique name.
*/
#if !defined(OGR_GET_DISPATCH_TABLE_FXN)
  #define OGR_GET_DISPATCH_TABLE_FXN    ogr_get_dispatch_table
#endif


#include "ansi/ogrp2_codebase.cpp"
