/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------
 * misc functions that don't fit anywhere else. 
 *
 * Note that this file may be included from anywhere, and should
 * therefore not have any non-standard dependancies (eg, 'u32', 'Stub' etc)
 * ----------------------------------------------------------------
*/ 
#ifndef __UTIL_H__ 
#define __UTIL_H__ "@(#)$Id: util.h,v 1.6.2.10 2000/10/13 21:50:23 cyp Exp $"

void trace_out( int indlevel, const char *fmt, ... );
void trace_setsrc( const char *src_filename );
#ifdef TRACE
#define TRACE_OUT(x) trace_setsrc(__FILE__); trace_out x
#else
#define TRACE_OUT(x) 
#endif

const char *projectmap_expand( const char *map );
const char *projectmap_build( char *buf, const char *strtomap );

int utilGatherOptionArraysToList( char *opsize, unsigned int maxsize,
                                  const int *table1, const int *table2 );
int utilScatterOptionListToArraysEx( const char *oplist, 
                                  int *table1, int *table2, 
                                  const int *defaults1, const int *defaults2 );
int utilScatterOptionListToArrays( const char *oplist, 
                                  int *table1, int *table2, 
                                  int defaultval );

/* Whats the name of this application? Used for thread-name, banners etc */
const char *utilGetAppName(void); /* "rc5 des" or "dnetc" or whatever */
const char *utilSetAppName(const char *newname); /* shouldn't be needed */

/* prints message if appropriate */
int utilCheckIfBetaExpired(int print_msg);

/* get list of pid's for procname. if procname has a path, then search 
   for exactly that, else search for basename. if pidlist or maxnumpids
   is null/0, then return found count, else return number of pids now
   in list. On error return < 0
*/
int utilGetPIDList( const char *procname, long *pidlist, int maxnumpids );


#endif /* __UTIL_H__ */

