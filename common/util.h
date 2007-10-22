/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
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
#define __UTIL_H__ "@(#)$Id: util.h,v 1.20 2007/10/22 16:48:28 jlawson Exp $"

/* drag in DNETC_UNUSED_* defines so we don't have to do so in all the
** source files explicitly */
#include "unused.h"

#if defined(__GNUC__)
#define __CHKFMT_TRACE_OUT __attribute__((__format__(__printf__,2,3)))
#else
#define __CHKFMT_TRACE_OUT
#endif

void trace_out( int indlevel, const char *fmt, ... ) __CHKFMT_TRACE_OUT;
void trace_setsrc( const char *src_filename );
#ifdef TRACE
#define TRACE_OUT(x) trace_setsrc(__FILE__); trace_out x
#else
#define TRACE_OUT(x)
#endif

#include "problem.h"       // for CONTEST_COUNT

// project order map + project state vec ==> string
const char *projectmap_expand( const int* map, const int* state );

// string or default ==> project order map + project state vec
const int* projectmap_build( int* buf, int* state, const char *strtomap );


int utilGatherOptionArraysToList(char *opsize, unsigned int maxsize,
                                 const int *table1, const int *table2);
int utilScatterOptionListToArraysEx(const char *oplist,
                                    int *table1, int *table2,
                                    const int *defaults1,
                                    const int *defaults2);
int utilScatterOptionListToArrays(const char *oplist,
                                  int *table1, int *table2,
                                  int defaultval);

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
int utilGetPIDList(const char *procname, long *pidlist, int maxnumpids);

/* returns 1 = valid, 0 = invalid */
int utilIsUserIDValid(const char *userid);

/* strncpy with terminating '\0' */
char *strncpyz(char *dest, const char *src, int n);

#endif /* __UTIL_H__ */
