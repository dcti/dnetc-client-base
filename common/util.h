/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------
 * misc functions that don't fit anywhere else
 * ----------------------------------------------------------------
*/ 
#ifndef __UTIL_H__ 
#define __UTIL_H__ "@(#)$Id: util.h,v 1.6.2.6 2000/02/04 08:29:59 cyp Exp $"

void trace_out( int indlevel, const char *fmt, ... );
#ifdef TRACE
#define TRACE_OUT(x) trace_out x
#else
#define TRACE_OUT(x) 
#endif

const char *projectmap_expand( const char *map );
const char *projectmap_build( char *buf, const char *strtomap );
//char *strfproj( char *buffer, const char *fmt, WorkRecord *work );

int utilGatherOptionArraysToList( char *opsize, unsigned int maxsize,
                                  const int *table1, const int *table2 );
int utilScatterOptionListToArraysEx( const char *oplist, 
                                  int *table1, int *table2, 
                                  const int *defaults1, const int *defaults2 );
int utilScatterOptionListToArrays( const char *oplist, 
                                  int *table1, int *table2, 
                                  int defaultval );

unsigned int __iter2norm( u32 iterlo, u32 iterhi );

const char *ogr_stubstr(const struct Stub *stub);
unsigned long ogr_nodecount( const struct Stub *stub );

int IsFilenameValid( const char *filename );
int DoesFileExist( const char *filename );

/* convert a basename to a real buffer file name */
const char *BufferGetDefaultFilename( unsigned int proj, int is_out_type,
                                                    const char *basename );

const char *utilGetAppName(void); /* "rc5 des" or "dnetc" or whatever */
const char *utilSetAppName(const char *newname); /* shouldn't be needed */

int utilCheckIfBetaExpired(int print_msg); /* prints message if appropriate */

/* get list of pid's for procname. if procname has a path, then search 
   for exactly that, else search for basename. if pidlist or maxnumpids
   is null/0, then return found count, else return number of pids now
   in list. On error return < 0
*/
int utilGetPIDList( const char *procname, long *pidlist, int maxnumpids );


#endif /* __UTIL_H__ */

