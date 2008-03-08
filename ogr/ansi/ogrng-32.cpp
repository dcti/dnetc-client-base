/*
 * Copyright distributed.net 2002-2008 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Implementation : Each of the three bitmaps "list", "dist" and "comp" is
 * made of eight 32-bit scalars.
 *
 * $Id: ogrng-32.cpp,v 1.1 2008/03/08 20:07:14 kakace Exp $
*/

#include "ansi/ogrng-32.h"


/*
** Define the name of the dispatch table.
** Each core shall define a unique name.
*/
#if !defined(OGR_NG_GET_DISPATCH_TABLE_FXN)
  #define OGR_NG_GET_DISPATCH_TABLE_FXN    ogrng_get_dispatch_table
#endif


#include "ansi/ogrng_codebase.cpp"
