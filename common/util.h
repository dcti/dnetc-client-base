/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ****************** THIS IS WORLD-READABLE SOURCE *********************
 *
 * $Log: util.h,v $
 * Revision 1.2  1999/03/20 07:32:42  cyp
 * moved IsFilenameValid() and DoesFileExist() to utils.cpp
 *
 * Revision 1.1  1999/03/18 03:51:18  cyp
 * Created.
 *
 *
*/

#ifndef __CLIENT_UTIL_H__ 
#define __CLIENT_UTIL_H__ 

extern const char *projectmap_expand( const char *map );
extern const char *projectmap_build( char *buf, const char *strtomap );
//char *strfproj( char *buffer, const char *fmt, WorkRecord *work );

extern const char *ogr_stubstr(const struct Stub *stub);
extern unsigned long ogr_nodecount( const struct Stub *stub );

extern int IsFilenameValid( const char *filename );
extern int DoesFileExist( const char *filename );

#endif /* __CLIENT_UTIL_H__ */

