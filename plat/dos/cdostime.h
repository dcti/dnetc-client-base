/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * ------------------------------------------------------------------
 * POSIX gettimeofday() and clock_gettime()
 *
 * gettimeofday caveat: the timezone.tz_dsttime member is not accurate. 
 * There is no way to translate the ANSI timezone variable to the 
 * dsttime DST_* code.
 * ------------------------------------------------------------------
*/
#ifndef __CLIDOS_TIME_H__
#define __CLIDOS_TIME_H__ "@(#)$Id: cdostime.h,v 1.1.2.1 2001/01/21 15:10:23 cyp Exp $"

#pragma pack(1)
struct timeval  { long tv_sec; long tv_usec; }; /* seconds and microsecs */
struct timespec { long tv_sec; long tv_nsec; }; /* seconds and nanosecs */
struct __timezone {int tz_minuteswest, tz_dsttime; };
#pragma pack();

#undef timezone
#define timezone __timezone

#ifdef __cplusplus
extern "C" {
#endif

int gettimeofday( struct timeval *, struct timezone * );
int clock_gettime( int /* actually clockid_t */, struct timespec * );
int getmicrotime( struct timeval * ); /* named in honor of BSD kernel */

#ifdef __cplusplus
}
#endif

#define DST_NONE  0 /* not on dst */
#define DST_USA   1 /* USA style dst */
#define DST_AUST  2 /* Australian style dst */
#define DST_WET   3 /* Western European dst */
#define DST_MET   4 /* Middle European dst */
#define DST_EET   5 /* Eastern European dst */
#define DST_CAN   6 /* Canada */
#define DST_GB    7 /* Great Britain and Eire */
#define DST_RUM   8 /* Rumania */
#define DST_TUR   9 /* Turkey */
#define DST_AUSTALT 10  /* Australian style with shift in 1986 */

typedef int clockid_t;
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME  0 /* supposed to be in <time.h> */
#endif
#define CLOCK_VIRTUAL   1
#define CLOCK_PROF      2
#define CLOCK_MONOTONIC 3

#endif //__CLIDOS_TIME_H__
