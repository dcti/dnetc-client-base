// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: os2defs.h,v $
// Revision 1.1.2.1  2001/01/21 15:10:27  cyp
// restructure and discard of obsolete elements
//
// Revision 1.7.2.1  1999/12/24 04:18:26  trevorh
// Add keyboard defines
//
// Revision 1.7  1999/04/15 21:58:33  trevorh
// Include files in correct order
//
// Revision 1.6  1999/02/10 00:51:36  trevorh
// Removed ifdef, ctype.h is needed in Watcom and EMX
//
// Revision 1.5  1999/02/04 22:50:32  trevorh
// Added #defines for INCL_SUB and INCL_DOS to make Vio calls work
//
// Revision 1.4  1999/01/27 19:52:48  patrick
//
// Added some defines for OS2-EMX target
//
// Revision 1.3  1999/01/01 02:45:18  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.2  1998/07/01 08:11:39  ziggyb
// Added new header to support the new connection detection routines on DOD and
// lurk.
//
// Revision 1.1  1998/06/14 11:17:31  ziggyb
// initial add
//
//

#ifndef OS2DEFS_H
#define OS2DEFS_H

#define INCL_DOSPROCESS         /* For Disk functions */
#define INCL_DOSFILEMGR         /* For Dos_Delete */
#define INCL_ERRORS             /* DOS error values */
#define INCL_DOSMISC            /* DosQuerySysInfo() */
#define INCL_WINWORKPLACE       /* Workplace shell objects */
#define INCL_VIO                /* OS/2 text graphics functions */
#define INCL_DOS
#define INCL_SUB
#define INCL_KBD
#include <os2.h>

extern "C" {
#ifndef __WATCOMC__
  // all compilers I have use sys/types.h (patrick)
  #include <sys/types.h>
  #include <string.h>
#else
  #include <types.h>
#endif
// defined isprint, ... for EMX and Watcom
#include <ctype.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/ioctl.h>
#include <sys/stat.h>     // for stat()
#define BSD_SELECT
#include <sys/select.h>
}

#endif
