/*
 * Copyright distributed.net 2001-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Wrapper around ogr.cpp for all processor WITH a fast bsr instruction.
 * (ie, PPro, PII, PIII)
 *
 * $Id: ogr-a.cpp,v 1.4 2003/11/01 14:20:15 mweiser Exp $
*/
#undef OGR_NOFFZ
/* fine OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table */

#include "ansi/ogr.cpp"
