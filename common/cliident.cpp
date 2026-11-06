/*
 * Copyright distributed.net 1997-2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Original version by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * with additions by distributed.net contributors.
 *
 * ----------------------------------------------------------------------
 * The file contains:
 * - CliIdentifyModules() which prints the git revisions the client was
 *   built from and when it was built ('-ident'), for bug reports.
 * - CliGetBuildTime() returns the time the client was built. Used for
 *   beta expiry and sanity checks for correct clocks.
 * - CliGetSourceTime() returns the commit time of the newest source
 *   tree the client was built from (or the build time, if unknown).
 * - CliGetFullVersionDescriptor() returns the unique build descriptor
 *   as used in the startup banner.
 * - CliIsDevelVersion() returns non-zero if the client was built from
 *   the devel branch or BETA or BETA_PERIOD is defined
 *
 * The build information comes from BUILDINFO_* macros that the build
 * system passes when compiling this file (see common/buildinfo.sh and
 * plat/win/buildinfo.cmd). All of them are optional: without them the
 * source shows as "Untracked" and the build time falls back to the
 * compiler's __DATE__ and __TIME__.
 * ----------------------------------------------------------------------
*/

#include "cputypes.h"
#include "baseincs.h"
#include "version.h"
#include "cliident.h"
#include "clitime.h"   /* CliGetTimeString() */
#include "console.h"   /* ConIsGUI() */
#include "logstuff.h"  /* LogScreenRaw() */
#include "util.h"      /* utilGetAppName() */

/* The private revision is only reported alongside a public one. */
#if !defined(BUILDINFO_PUB_HASH)
  #undef BUILDINFO_PRIV_HASH
#endif
#if defined(BUILDINFO_PUB_HASH) && !defined(BUILDINFO_PUB_DIRTY)
  #define BUILDINFO_PUB_DIRTY 0
#endif
#if defined(BUILDINFO_PRIV_HASH) && !defined(BUILDINFO_PRIV_DIRTY)
  #define BUILDINFO_PRIV_DIRTY 0
#endif

/* ------------------------------------------------------------------------ */

/*
 * Returns "-<hash>" for each repo the client was built from, with a '*'
 * after a hash whose tree had uncommitted changes, e.g.
 * "-ae2870e4*-92467774". Returns "" for an untracked build.
 */
static const char *get_revisions(void)
{
  static char revisions[2 * sizeof("-01234567*")];
  size_t len = 0;

  revisions[0] = '\0';
  #if defined(BUILDINFO_PUB_HASH)
  snprintf(revisions, sizeof(revisions), "-%08lx%s",
           ((unsigned long)(BUILDINFO_PUB_HASH) & 0xffffffffUL),
           ((BUILDINFO_PUB_DIRTY) ? ("*") : ("")) );
  len = strlen(revisions);
  #endif
  #if defined(BUILDINFO_PRIV_HASH)
  snprintf(revisions + len, sizeof(revisions) - len, "-%08lx%s",
           ((unsigned long)(BUILDINFO_PRIV_HASH) & 0xffffffffUL),
           ((BUILDINFO_PRIV_DIRTY) ? ("*") : ("")) );
  #endif
  (void)len;
  return revisions;
}

/* ------------------------------------------------------------------------ */

//! Time the client was built.
/*!
 * Taken from BUILDINFO_TIME. Without it, falls back to __DATE__ and
 * __TIME__, which are in the build machine's local time, so may be off
 * by the time zone difference. Used, for instance, to 'ensure' that time
 * from the .ini or recvd from a proxy is sane.
 *
 * \return UTC seconds since the epoch, or 0 if unknown.
 */
time_t CliGetBuildTime(void)
{
  #if defined(BUILDINFO_TIME)
  return (time_t)(BUILDINFO_TIME);
  #else
  static time_t buildtime = (time_t)0;
  if (buildtime == (time_t)0)
  {
    static const char monnames[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
    const char *date = __DATE__;   /* "Mmm dd yyyy" */
    const char *tod = __TIME__;    /* "hh:mm:ss" */
    struct tm bd;
    unsigned int mon;

    for (mon = 0; mon < 12; mon++)
    {
      if (memcmp(&monnames[mon * 3], date, 3) == 0)
        break;
    }
    memset((void *)&bd, 0, sizeof(bd));
    bd.tm_mon  = (int)mon;
    bd.tm_mday = atoi(date + 4);
    bd.tm_year = atoi(date + 7) - 1900;
    bd.tm_hour = atoi(tod);
    bd.tm_min  = atoi(tod + 3);
    bd.tm_sec  = atoi(tod + 6);
    bd.tm_isdst = -1;
    if (mon < 12 && bd.tm_mday >= 1 && bd.tm_mday <= 31 && bd.tm_year >= 99)
    {
      time_t t = mktime(&bd);
      if (t != ((time_t)-1))
        buildtime = t;
    }
  }
  return buildtime;
  #endif
}

//! Commit time of the newest source tree the client was built from.
/*!
 * The newer of the public and private repos' HEAD commit times. Falls
 * back to the build time for an untracked build.
 *
 * \return UTC seconds since the epoch, or 0 if unknown.
 */
time_t CliGetSourceTime(void)
{
  time_t newest = (time_t)0;
  #if defined(BUILDINFO_PUB_HASH) && defined(BUILDINFO_PUB_TIME)
  newest = (time_t)(BUILDINFO_PUB_TIME);
  #endif
  #if defined(BUILDINFO_PRIV_HASH) && defined(BUILDINFO_PRIV_TIME)
  if ((time_t)(BUILDINFO_PRIV_TIME) > newest)
    newest = (time_t)(BUILDINFO_PRIV_TIME);
  #endif
  if (newest == (time_t)0)
    newest = CliGetBuildTime();
  return newest;
}

/* ------------------------------------------------------------------------ */

static void print_time_line(const char *label, time_t when)
{
  if (when)
  {
    struct timeval tv;
    tv.tv_sec = when;
    tv.tv_usec = 0;
    LogScreenRaw("%-16s%s UTC\n", label, CliGetTimeString(&tv, 3));
  }
  else
    LogScreenRaw("%-16sunknown\n", label);
}

//! Print the source revisions and build time ('-ident').
void CliIdentifyModules(void)
{
  #if defined(BUILDINFO_PUB_HASH)
  LogScreenRaw("%-16s%08lx%s\n", "Public source:",
               (unsigned long)(BUILDINFO_PUB_HASH),
               ((BUILDINFO_PUB_DIRTY) ? (" (uncommitted changes)") : ("")) );
    #if defined(BUILDINFO_PUB_TIME)
    print_time_line("  committed:", (time_t)(BUILDINFO_PUB_TIME));
    #endif
  #else
  LogScreenRaw("%-16s%s\n", "Public source:", "Untracked");
  #endif
  #if defined(BUILDINFO_PRIV_HASH)
  LogScreenRaw("%-16s%08lx%s\n", "Private source:",
               (unsigned long)(BUILDINFO_PRIV_HASH),
               ((BUILDINFO_PRIV_DIRTY) ? (" (uncommitted changes)") : ("")) );
    #if defined(BUILDINFO_PRIV_TIME)
    print_time_line("  committed:", (time_t)(BUILDINFO_PRIV_TIME));
    #endif
  #endif
  print_time_line("Built:", CliGetBuildTime());
}

/* ------------------------------------------------------------------------ */

//! Determine whether the build is "beta" or "release" build.
/*!
 * Development beta builds have a timebomb that causes them to
 * expire after a few days/weeks.
 *
 * \return Returns zero if stable, non-zero if development.
 */
int CliIsDevelVersion(void)
{
  #if (defined(BETA) || defined(BETA_PERIOD))
  return 1;
  #else
  return 0;
  #endif
}

const char *CliGetFullVersionDescriptor(void)
{
  static char buffer[sizeof("v" CLIENT_VERSIONSTRING "-XXX-*dev* for " CLIENT_OS_NAME_EXTENDED) + 64];

  snprintf( buffer, sizeof(buffer), "%s v" CLIENT_VERSIONSTRING "-"
         "%c"  /* GUI == "G", CLI == "C" */
         #ifdef CLIENT_SUPPORTS_SMP
           #ifdef HAVE_MULTICRUNCH_VIA_FORK
           "F" /* fork() */
           #else
           "T" /* threads */
           #endif
         #else
         "P"   /* polling */
         #endif
         "%c"  /* limited release or dev branch or public release */
         "%s"  /* "-<public hash>[*][-<private hash>[*]]" or "" */
         "%s"  /* "-*dev*" or "" */
         " for " CLIENT_OS_NAME_EXTENDED,
         utilGetAppName(),
         ((ConIsGUI())?('G'):('C')),
         ((CliIsDevelVersion())?('L'):('R')),
         get_revisions(),
         ((CliIsDevelVersion())?("-*dev*"):("")) );
  buffer[sizeof(buffer) - 1] = '\0';
  return buffer;
}
