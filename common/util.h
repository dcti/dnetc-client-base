/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Log: util.h,v $
 * Revision 1.1  1999/03/18 03:51:18  cyp
 * Created.
 *
 *
*/

#ifndef __CLIENT_UTIL_H__ 
#define __CLIENT_UTIL_H__ 


const char *projectmap_expand( const char *map );
const char *projectmap_build( char *buf, const char *strtomap );
const char *ogr_stubstr(const struct Stub *stub);
//char *strfproj( char *buffer, const char *fmt, FileEntry *data );

#endif /* __CLIENT_UTIL_H__ */

