/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------
 * misc functions that don't fit anywhere else
 * ----------------------------------------------------------------
*/ 
#ifndef __CLIENT_UTIL_H__ 
#define __CLIENT_UTIL_H__ "@(#)$Id: util.h,v 1.4 1999/04/05 13:56:41 cyp Exp $"

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

