/*
 * Copyright distributed.net 2001-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr.cpp for all processor WITHOUT a fast bsr instruction.
 * (ie, 386, 486, Pentium, P4, K5, K6, K7, Cyrix(all), etc)
 *
 * $Id: ogr-b.cpp,v 1.4 2003/11/01 14:20:15 mweiser Exp $
*/
#define OGR_NOFFZ
#define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_nobsr

#include "ansi/ogr.cpp"
