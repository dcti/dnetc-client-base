/*
 * Wrapper around ogr.cpp for all processor WITH a fast bsr instruction.
 * (ie, PPro, PII, PIII)
 *
 * $Id: ogr-a.cpp,v 1.1.2.1 2001/04/06 16:19:24 cyp Exp $
*/
#undef OGR_NOFFZ
/* fine OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table */

#include "ansi/ogr.cpp"