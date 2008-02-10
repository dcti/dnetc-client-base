/* -*-C++-*-
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This file should be shared between client and proxynet some day ...
 */
#ifndef __PROJDATA_H__
#define __PROJDATA_H__ "@(#)$Id: projdata.h,v 1.5 2008/02/10 00:24:30 kakace Exp $"

#include "cputypes.h"   // u32 ...

/* ------------------------------------------------------------------------- */

enum {
  OGR_NG,
  RC5_72,   // http://www.rsasecurity.com/rsalabs/challenges/secretkey/
  OGR_P2    // OGR final stage
  // PROJECT_NOT_HANDLED("create your project id here and adjust count")
};
#define PROJECT_COUNT       3  /* OGR_NG,RC5_72,OGR_P2 */
#define CONTEST_COUNT       PROJECT_COUNT

#define MAX_PROJECT_NAME_LEN 6 /* "RC5-72" */

/* ------------------------------------------------------------------------- */
  
/* Project Flags - static, defined at compile time */

#define PROJECT_UNSUPPORTED                       0x00000000
#define PROJECT_OK                                0x00000001
#define PROJECTFLAG_TIME_THRESHOLD                0x00000002
#define PROJECTFLAG_RANDOM_BLOCKS                 0x00000004

/* Project States - may change at runtime */

#define PROJECTSTATE_USER_DISABLED                0x00000001
#define PROJECTSTATE_SUSPENDED                    0x00000002
#define PROJECTSTATE_CLOSED                       0x00000004

/* ------------------------------------------------------------------------- */

/* returns PROJECT_UNSUPPORTED (0) if client has no core for projectid,
   PROJECT_OK | (list of flags for projectid) otherwise */
u32 ProjectGetFlags(int projectid);

const char *ProjectGetName(int projectid);
const char *ProjectGetIniSectionName(int projectid);
const char *ProjectGetUnitName(int projectid);
const char *ProjectGetFileExtension(int projectid);

/* ------------------------------------------------------------------------- */

#endif /* __PROJDATA_H__ */
