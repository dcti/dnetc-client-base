/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/

#ifndef __OS2DEFS_H__
#define __OS2DEFS_H__ "@(#)$Id: os2defs.h,v 1.3 2003/09/12 22:29:27 mweiser Exp $"

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
#include <sys/time.h>
#define BSD_SELECT
#include <sys/select.h>
#include <stdlib.h>
}

#ifndef _MAX_PATH
    #define _MAX_PATH 255
#endif

int os2CliUninstallClient(int do_the_uninstall_without_feedback);
int os2CliInstallClient(int do_the_install_without_feedback, const char *exename);
int os2CliSendSignal(int action, const char *exename);

#define DNETC_MSG_RESTART    0x00
#define DNETC_MSG_SHUTDOWN   0x01
#define DNETC_MSG_PAUSE      0x02
#define DNETC_MSG_UNPAUSE    0x03

int os2GetPIDList(const char *procname, long *pidlist, int maxnumpids);

#endif
