/*
 * Wrapper around ogr.cpp for all processor WITHOUT a fast bsr instruction.
 * (ie, 386, 486, Pentium, P4, K5, K6, K7, Cyrix(all), etc)
 *
 * $Id: ogr-b.cpp,v 1.1.2.1 2001/04/06 16:19:24 cyp Exp $
*/
#define OGR_NOFFZ
#define OGR_GET_DISPATCH_TABLE_FXN ogr_get_dispatch_table_nobsr

#include "ansi/ogr.cpp"