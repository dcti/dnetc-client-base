/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __PROBFILL_H__
#define __PROBFILL_H__ "@(#)$Id: probfill.h,v 1.8.2.3 1999/12/19 19:23:24 cyp Exp $"

#define PROBFILL_ANYCHANGED  1
#define PROBFILL_GETBUFFERRS 2
#define PROBFILL_UNLOADALL   3
#define PROBFILL_RESIZETABLE 4

unsigned int LoadSaveProblems(Client *client,
                              unsigned int load_problem_count,int mode);
/* returns number of actually loaded problems */

// --------------------------------------------------------------------------

#define PROBLDR_DISCARD      0x01
#define PROBLDR_FORCEUNLOAD  0x02
extern int SetProblemLoaderFlags( const char *loaderflags_map /* 1 char per contest */ );

// --------------------------------------------------------------------------

#define FILEENTRY_OS         CLIENT_OS
#define FILEENTRY_BUILDHI    ((CLIENT_BUILD_FRAC >> 8) & 0xff)
#define FILEENTRY_BUILDLO    ((CLIENT_BUILD_FRAC     ) & 0xff)
#define FILEENTRY_CPU(_core_cpu,_core_sel) \
                             (((_core_cpu & 0x0f)<<4) | (_core_sel & 0x0f))

// --------------------------------------------------------------------------

#endif /* __PROBFILL_H__ */
