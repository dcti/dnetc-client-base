// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: probfill.h,v $
// Revision 1.5  1999/01/01 02:45:16  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.4  1998/12/01 00:34:04  cyp
// hah! gcc doesn't savvy trailing space in multi-line directives.
//
// Revision 1.3  1998/12/01 15:00:00  cyp
// New 'resize mode' for probfill.
//
// Revision 1.2  1998/11/25 09:23:36  chrisb
// various changes to support x86 coprocessor under RISC OS
//
// Revision 1.1  1998/09/28 01:16:09  cyp
// Spun off from client.cpp
//
// 

#ifndef __PROBFILL_H__
#define __PROBFILL_H__

#define PROBFILL_ANYCHANGED  1
#define PROBFILL_GETBUFFERRS 2
#define PROBFILL_UNLOADALL   3
#define PROBFILL_RESIZETABLE 4

// --------------------------------------------------------------------------
#if (( CLIENT_CPU         > 0x01F  /* 0-31 */  ) || \
     ((CLIENT_CONTEST-64) > 0x0F   /* 64-79 */ ) || \
     ( CLIENT_BUILD       > 0x07   /* 0-7 */   ) || \
     ( CLIENT_BUILD_FRAC  > 0x03FF /* 0-1023 */) || \
     ( CLIENT_OS          > 0x3F   /* 0-63 */  ))      /* + cputype 0-15 */
#error CLIENT_CPU/_OS/_CONTEST/_BUILD are out of range for FileEntry check tags
#endif    

#define FILEENTRY_CPU    ((u8)(((cputype & 0x0F)<<4) | (CLIENT_CPU & 0x0F)))

#if (CLIENT_OS == OS_RISCOS)
#define FILEENTRY_RISCOS_X86_CPU ((u8)(((cputype & 0x0F)<<4) | (CPU_X86 & 0x0F)))
#endif

#define FILEENTRY_OS      ((CLIENT_OS & 0x3F) | ((CLIENT_CPU & 0x10) << 3) | \
                           (((CLIENT_BUILD_FRAC>>8)&2)<<5))
#define FILEENTRY_BUILDHI ((((CLIENT_CONTEST-64)&0x0F)<<4) | \
                            ((CLIENT_BUILD & 0x07)<<1) | \
                            ((CLIENT_BUILD_FRAC>>8)&1)) 
#define FILEENTRY_BUILDLO ((CLIENT_BUILD_FRAC) & 0xff)  

// --------------------------------------------------------------------------

#endif /* __PROBFILL_H__ */
