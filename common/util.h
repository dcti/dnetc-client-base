/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ****************** THIS IS WORLD-READABLE SOURCE *********************
 *
 * $Log: util.h,v $
 * Revision 1.3  1999/03/31 11:43:09  cyp
 * created BufferGetDefaultFilename
 *
 * Revision 1.2  1999/03/20 07:32:42  cyp
 * moved IsFilenameValid() and DoesFileExist() to utils.cpp
 *
 * Revision 1.1  1999/03/18 03:51:18  cyp
 * Created.
 *
*/

#ifndef __CLIENT_UTIL_H__ 
#define __CLIENT_UTIL_H__ 

const char *projectmap_expand( const char *map );
const char *projectmap_build( char *buf, const char *strtomap );
//char *strfproj( char *buffer, const char *fmt, WorkRecord *work );

const char *ogr_stubstr(const struct Stub *stub);
unsigned long ogr_nodecount( const struct Stub *stub );

int IsFilenameValid( const char *filename );
int DoesFileExist( const char *filename );

/* convert a basename to a real buffer file name */
const char *BufferGetDefaultFilename( unsigned int proj, int is_out_type,
                                                    const char *basename );
#endif /* __CLIENT_UTIL_H__ */

