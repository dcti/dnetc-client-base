/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __PROBFILL_H__
#define __PROBFILL_H__ "@(#)$Id: probfill.h,v 1.8.2.2 1999/10/11 18:42:59 cyp Exp $"

#define PROBFILL_ANYCHANGED  1
#define PROBFILL_GETBUFFERRS 2
#define PROBFILL_UNLOADALL   3
#define PROBFILL_RESIZETABLE 4

// --------------------------------------------------------------------------

#define PROBLDR_DISCARD      0x01
#define PROBLDR_FORCEUNLOAD  0x02
extern int SetProblemLoaderFlags( const char *loaderflags_map /* 1 char per contest */ );

// --------------------------------------------------------------------------

#if (CLIENT_CONTEST >= 80)
  #if (( CLIENT_CPU       > 0x01F  /* 0-31 */  ) || \
     ((CLIENT_CONTEST-80) > 0x07   /* 80-87 */ ) || \
     ( CLIENT_BUILD       > 0x0F   /* 0-15 */   ) || \
     ( CLIENT_BUILD_FRAC  > 0x03FF /* 0-1023 */) || \
     ( CLIENT_OS          > 0x3F   /* 0-63 */  ))      /* + cputype 0-15 */
  #error CLIENT_CPU/_OS/_CONTEST/_BUILD are out of range for FileEntry check tags
  #endif
#else
  #if (( CLIENT_CPU         > 0x01F  /* 0-31 */  ) || \
       ((CLIENT_CONTEST-64) > 0x07   /* 64-71 */ ) || \
       ( CLIENT_BUILD       > 0x0F   /* 0-15 */   ) || \
       ( CLIENT_BUILD_FRAC  > 0x03FF /* 0-1023 */) || \
       ( CLIENT_OS          > 0x3F   /* 0-63 */  ))      /* + cputype 0-15 */
  #error CLIENT_CPU/_OS/_CONTEST/_BUILD are out of range for FileEntry check tags
	#endif
#endif    

/* 
   The fileentry cpu macro is used only from within probfill. 
   Keep it that way.
*/
#define FILEENTRY_CPU     ((u8)(((cputype & 0x0F)<<4) | (CLIENT_CPU & 0x0F)))
#define FILEENTRY_OS      ((CLIENT_OS & 0x3F) | ((CLIENT_CPU & 0x10) << 3) | \
                          (((CLIENT_BUILD_FRAC>>8)&2)<<5))
#if (CLIENT_CONTEST >= 80)
#define FILEENTRY_BUILDHI ((((CLIENT_CONTEST-80)&0x07)<<5) | \
                            ((CLIENT_BUILD & 0x0F)<<1) | \
                            ((CLIENT_BUILD_FRAC>>8)&1)) 
#else														
#define FILEENTRY_BUILDHI ((((CLIENT_CONTEST-64)&0x07)<<5) | \
                            ((CLIENT_BUILD & 0x0F)<<1) | \
                            ((CLIENT_BUILD_FRAC>>8)&1)) 
#endif														
#define FILEENTRY_BUILDLO ((CLIENT_BUILD_FRAC) & 0xff)

// --------------------------------------------------------------------------

unsigned int LoadSaveProblems(Client *client,
                              unsigned int load_problem_count,int mode);
/* returns number of actually loaded problems */

#endif /* __PROBFILL_H__ */
