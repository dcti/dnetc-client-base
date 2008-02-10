/*
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *projdata_cpp(void) {
return "@(#)$Id: projdata.cpp,v 1.6 2008/02/10 00:24:30 kakace Exp $"; }

#include "cputypes.h"
#include "projdata.h"
#include "baseincs.h"


static const struct ProjectInfo_t
{
  int ProjectID;
  const char *ProjectName;
  const char *FileExtension;    /* length=3, unique */
  const char *IniSectionName;
  const char *UnitName;
  unsigned int Iter2_call_me_something_better_than__Key__Factor; /* by how much must iterations/keysdone
                        be multiplied to get the number of keys checked. */
} ProjectInfoVec[] =
// obsolete projects may be omitted
{
//  ProjectID             Name      filext ini sect. unit
  { OGR_NG, "OGR-NG", "og2", "ogr-ng", "nodes", 1 },
  { RC5_72, "RC5-72", "r72", "rc5-72", "keys",  1 },
  { OGR_P2, "OGR-P2", "ogf", "ogr-p2", "nodes", 1 },
  { -1,                   NULL,     NULL,  NULL,     NULL,    0 }
#if (PROJECT_COUNT != 3)
  #error PROJECT_NOT_HANDLED("ProjectInfo[]: static initializer was last updated for PROJECT_COUNT == 3")
#endif
};

/* ----------------------------------------------------------------------- */

static const struct ProjectInfo_t *__internalGetProjectInfoVector( int projectid )
{
  for (int i = 0; ProjectInfoVec[i].ProjectName != NULL; i++)
  {
    if (ProjectInfoVec[i].ProjectID == projectid)
      return (&ProjectInfoVec[i]);
  }
  return ((const struct ProjectInfo_t *)(NULL));
}

// --------------------------------------------------------------------------

/* returns PROJECT_UNSUPPORTED (0) if client has no core for projectid,
   PROJECT_OK | (list of flags for projectid) otherwise */
u32 ProjectGetFlags(int projectid)
{
  #if defined(HAVE_OGR_CORES)
    #define PROJECT_OK_OGR_NG PROJECT_OK
  #else
    #define PROJECT_OK_OGR_NG PROJECT_UNSUPPORTED
  #endif
  #if defined(HAVE_RC5_72_CORES)
    #define PROJECT_OK_RC5_72 PROJECT_OK
  #else
    #define PROJECT_OK_RC5_72 PROJECT_UNSUPPORTED
  #endif
  #if defined(HAVE_OGR_PASS2)
    #define PROJECT_OK_OGR_P2 PROJECT_OK
  #else
    #define PROJECT_OK_OGR_P2 PROJECT_UNSUPPORTED
  #endif
  static const u32 projectflags[PROJECT_COUNT] = {
    /* OGR_NG */ PROJECT_OK_OGR_NG,
    /* RC5_72 */ PROJECT_OK_RC5_72
        | PROJECTFLAG_TIME_THRESHOLD /* not yet! | PROJECTFLAG_PREFERRED_BLOCKSIZE */
        | PROJECTFLAG_RANDOM_BLOCKS,
    /* OGR_P2 */ PROJECT_OK_OGR_P2,
  };
  #undef PROJECT_OK_OGR_NG
  #undef PROJECT_OK_RC5_72
  #undef PROJECT_OK_OGR_P2

  if ( 0 <= projectid && projectid < PROJECT_COUNT )
    if ((projectflags[projectid] & PROJECT_OK) == PROJECT_OK)
      return projectflags[projectid];

  return PROJECT_UNSUPPORTED;
}

// --------------------------------------------------------------------------

const char *ProjectGetName(int projectid)
{
  const struct ProjectInfo_t *ProjectInfo =
                    __internalGetProjectInfoVector( projectid );
  if (ProjectInfo && ProjectInfo->ProjectName && *ProjectInfo->ProjectName)
    return ProjectInfo->ProjectName;
  return ((const char *)("???"));
}

const char *ProjectGetIniSectionName(int projectid)
{
  const struct ProjectInfo_t *ProjectInfo =
                    __internalGetProjectInfoVector( projectid );
  if (ProjectInfo && ProjectInfo->IniSectionName && *ProjectInfo->IniSectionName)
    return ProjectInfo->IniSectionName;
  return ((const char *)("unknown-project"));
}

const char *ProjectGetUnitName(int projectid)
{
  const struct ProjectInfo_t *ProjectInfo =
                    __internalGetProjectInfoVector( projectid );
  if (ProjectInfo && ProjectInfo->UnitName && *ProjectInfo->UnitName)
    return ProjectInfo->UnitName;
  return ((const char *)("units"));
}

const char *ProjectGetFileExtension(int projectid)
{
  const struct ProjectInfo_t *ProjectInfo =
                    __internalGetProjectInfoVector( projectid );
  if (ProjectInfo && ProjectInfo->FileExtension && *ProjectInfo->FileExtension)
    return ProjectInfo->FileExtension;
  return ((const char *)("zzz"));
}

// --------------------------------------------------------------------------
