/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ANSI/POSIX functions not available for NetWare
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwlcomp.h,v 1.1.2.2 2002/04/12 23:56:49 andreasb Exp $
*/

#ifndef __CNW_LCOMPAT_H_
#define __CNW_LCOMPAT_H_ 

#include <time.h>                       /* time_t and time(time_t *) */
                                        /* note: nw's time_t is *unsigned* */
#pragma pack(1)                         
struct timeb {                          /* from <sys/timeb.h> */
        time_t  time;                   /* seconds since the Epoch */
        unsigned short millitm;         /* + milliseconds since the Epoch */
        short   timezone;               /* minutes west of CUT */
        short   dstflag;                /* DST == non-zero */
};
#pragma pack()

/* ------------------------------------------------------------------- */

#ifdef __cplusplus   /* avoid pulling in *any* non-standard headers */
extern "C" {
#endif

extern int ftime(struct timeb *__tb);    /* ANSI (should be in time.h) */
extern int daylight;                     /* this is really extern */
extern int timezone; /* this is really extern. intentionally 'int' */

extern void usleep(unsigned long);       /* BSD 4.3 (should be unistd.h) */
extern unsigned int sleep(unsigned int); /* POSIX 1 (should be unistd.h) */

#ifdef __cplusplus
}
#endif

#endif /* __CNW_LCOMPAT_H_ */
