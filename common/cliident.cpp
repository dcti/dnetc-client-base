/*
 * Copyright distributed.net 1997-2011 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * ----------------------------------------------------------------------
 * The file contains:
 * - CliIdentifyModules() which lists the cvs id strings to stdout.
 *   Users can assist us (when making bug reports) by telling us
 *   exactly which modules were actually in effect when the binary was made.
 *   Currently, starting the client with the '-ident' switch will exec the
 *   function.
 * - CliGetNewestModuleTime() returns time_t of the newest module
 *   in the list - useful for asserting beta expiry, that the
 *   time obtained from proxies is sane etc.
 * - CliGetFullVersionDescriptor() returns the unique build descriptor
 *   as used in the startup banner.
 * - CliIsDevelVersion() returns non-zero if the client was built from
 *   the devel branch or BETA or BETA_PERIOD is defined
 * ----------------------------------------------------------------------
*/
const char *cliident_cpp(void) {
return "@(#)$Id: cliident.cpp,v 1.38 2011/03/31 05:07:27 jlawson Exp $"; }

#include "cputypes.h"
#include "baseincs.h"
#include "base64.h"
#include "bench.h"
#include "client.h" /* client.h needs to before buff*.h */
#include "buffbase.h"
#include "buffupd.h"
#include "ccoreio.h"
#include "checkpt.h"
#include "clicdata.h"
#include "clievent.h"
#include "cliident.h"
#include "clisync.h"
#include "clitime.h"
#include "cmdline.h"
#include "confmenu.h"
#include "confopt.h"
#include "confrwv.h"
#include "console.h"
#include "coremem.h"
#include "cpucheck.h"
#include "disphelp.h"
#include "iniread.h"
#include "logstuff.h"
#include "lurk.h"
#include "mail.h"
//#include "memfile.h"
#include "modereq.h"
#include "netbase.h"
#include "netconn.h"
#include "pack.h"       /* careful: order is important here */
#include "pack1.h"      /* switches on packing */
#include "pack4.h"
#include "pack8.h"
#include "pack0.h"      /* switch packing off again */
#include "pathwork.h"
#include "pollsys.h"
#include "probfill.h"
#include "problem.h"
#include "probman.h"
#include "projdata.h"
#include "random.h"
#include "rsadata.h"
#include "selcore.h"
#include "selftest.h"
#include "setprio.h"
#include "sleepdef.h"
#include "triggers.h"
#include "unused.h"
#include "util.h"
#include "version.h"
#if (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
#include "w32sock.h"
#include "w32cons.h"
#include "w32pre.h"
#include "w32util.h"
#include "w32svc.h"
#endif
#if (CLIENT_OS == OS_NEXTSTEP)
#include "next_sup.h"
#endif
#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
#include "x86id.h"
#endif
#if defined(HAVE_OGR_CORES) || defined(HAVE_OGR_PASS2)
#include "ogr.h"
#endif

static const char *h_ident_table[] =
{
  (const char *)__BASE64_H__,
  (const char *)__BASEINCS_H__,
  (const char *)__BENCH_H__,
  (const char *)__BUFFBASE_H__,
  (const char *)__BUFFUPD_H__,
  (const char *)__CCOREIO_H__,
  (const char *)__CHECKPT_H__,
  (const char *)__CLICDATA_H__,
  (const char *)__CLIENT_H__,
  (const char *)__CLIEVENT_H__,
  (const char *)__CLIIDENT_H__,
  (const char *)__CLISYNC_H__,
  (const char *)__CLITIME_H__,
  (const char *)__CMDLINE_H__,
  (const char *)__CONFMENU_H__,
  (const char *)__CONFOPT_H__,
  (const char *)__CONFRWV_H__,
  (const char *)__CONSOLE_H__,
  (const char *)__COREMEM_H__,
  (const char *)__CPUCHECK_H__,
  (const char *)__CPUTYPES_H__,
  (const char *)__DISPHELP_H__,
  (const char *)__INIREAD_H__,
  (const char *)__LOGSTUFF_H__,
  (const char *)__LURK_H__,
  (const char *)__MAIL_H__,
//(const char *)__MEMFILE_H__,
  (const char *)__MODEREQ_H__,
  (const char *)__NETBASE_H__,
  (const char *)__NETCONN_H__,
  (const char *)__PACK_H__,
  (const char *)__PACK0_H__,
  (const char *)__PACK1_H__,
  (const char *)__PACK4_H__,
  (const char *)__PACK8_H__,
  (const char *)__PATHWORK_H__,
  (const char *)__POLLSYS_H__,
  (const char *)__PROBFILL_H__,
  (const char *)__PROBLEM_H__,
  (const char *)__PROBMAN_H__,
  (const char *)__PROJDATA_H__,
  (const char *)__RANDOM_H__,
  (const char *)__RSADATA_H__,
  (const char *)__SELCORE_H__,
  (const char *)__SELFTEST_H__,
  (const char *)__SETPRIO_H__,
  (const char *)__SLEEPDEF_H__,
  (const char *)__TRIGGERS_H__,
  (const char *)__UNUSED_H__,
  (const char *)__UTIL_H__,
  (const char *)__VERSION_H__,
  #if (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  (const char *)__W32SOCK_H__,
  (const char *)__W32CONS_H__,
  (const char *)__W32PRE_H__,
  (const char *)__W32UTIL_H__,
  (const char *)__W32SVC_H__,
  #endif
  #if (CLIENT_OS == OS_OS2)
  (const char *)__OS2DEFS_H__,
  #endif
  #if (CLIENT_OS == OS_NEXTSTEP)
  (const char *)__NEXT_SUP_H__,
  #endif
#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
  (const char *)__X86ID_H__,
#endif
#if defined(HAVE_RC5_72_CORES)
#endif
#if defined(HAVE_OGR_PASS2)
  (const char *)__OGR_H__,
#endif
#if defined(HAVE_OGR_CORES)
  (const char *)__OGR_NG_H__,
  (const char *)__OGR_INTERFACE_H__,
#endif
  (const char *)0
};

extern const char *base64_cpp(void);
extern const char *bench_cpp(void);
extern const char *buffbase_cpp(void);
extern const char *buffpub_cpp(void);

//disable so public source can compile
//extern const char *buffupd_cpp(void);	
extern const char *checkpt_cpp(void);
extern const char *clicdata_cpp(void);
extern const char *client_cpp(void);
extern const char *clievent_cpp(void);
extern const char *cliident_cpp(void);
extern const char *clirun_cpp(void);
extern const char *clitime_cpp(void);
extern const char *cmdline_cpp(void);
extern const char *confmenu_cpp(void);
extern const char *confopt_cpp(void);
extern const char *confrwv_cpp(void);
extern const char *console_cpp(void);
extern const char *coremem_cpp(void);
extern const char *cpucheck_cpp(void);
extern const char *disphelp_cpp(void);
extern const char *iniread_cpp(void);
extern const char *logstuff_cpp(void);
extern const char *lurk_cpp(void);
extern const char *mail_cpp(void);
extern const char *memfile_cpp(void);
extern const char *modereq_cpp(void);
extern const char *netbase_cpp(void);
extern const char *netconn_cpp(void);
extern const char *pathwork_cpp(void);
extern const char *pollsys_cpp(void);
extern const char *probfill_cpp(void);
extern const char *problem_cpp(void);
extern const char *probman_cpp(void);
extern const char *projdata_cpp(void);
extern const char *random_cpp(void);
extern const char *selcore_cpp(void);
extern const char *selftest_cpp(void);
extern const char *setprio_cpp(void);
extern const char *triggers_cpp(void);
extern const char *util_cpp(void);
#if (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
extern const char *w32sock_cpp(void);
extern const char *w32cons_cpp(void);
extern const char *w32pre_cpp(void);
extern const char *w32util_cpp(void);
extern const char *w32svc_cpp(void);
#endif
#if (CLIENT_OS == OS_OS2)
extern const char *os2inst_cpp(void);
#endif
#if (CLIENT_OS == OS_NEXTSTEP)
extern const char *next_sup_cpp(void);
#endif
#if (CLIENT_OS == OS_LINUX)
extern "C" const char *li_inst_c(void);
extern "C" const char *resolv_c(void);
#endif
#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
extern const char *x86id_cpp(void);
#endif
#if defined(HAVE_RC5_72_CORES)
extern const char *core_r72_cpp(void);
extern const char *stress_r72_cpp(void);
#endif
#if defined(HAVE_OGR_PASS2)
extern const char *core_ogr_cpp(void);
extern const char *ogr_dat_cpp(void);
#endif
#if defined(HAVE_OGR_CORES)
extern const char *core_ogr_ng_cpp(void);
extern const char *ogr_sup_cpp(void);
extern const char *ogrng_dat_cpp(void);
extern const char *ogrng_init_cpp(void);
#endif

static const char * (*ident_table[])(void) =
{
  base64_cpp,
  bench_cpp,
  buffbase_cpp,
  buffpub_cpp,
//  buffupd_cpp,
  checkpt_cpp,
  clicdata_cpp,
  client_cpp,
  clievent_cpp,
  cliident_cpp,
  clirun_cpp,
  clitime_cpp,
  cmdline_cpp,
  confmenu_cpp,
  confopt_cpp,
  confrwv_cpp,
  console_cpp,
  coremem_cpp,
  cpucheck_cpp,
  disphelp_cpp,
  iniread_cpp,
  logstuff_cpp,
  #ifdef LURK
  lurk_cpp,
  #endif
  mail_cpp,
//memfile_cpp,
  modereq_cpp,
  netbase_cpp,
  netconn_cpp,
  pathwork_cpp,
  pollsys_cpp,
  probfill_cpp,
  problem_cpp,
  probman_cpp,
  projdata_cpp,
  random_cpp,
  selcore_cpp,
  selftest_cpp,
  setprio_cpp,
  triggers_cpp,
  util_cpp,
  #if (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
  w32sock_cpp,
  w32cons_cpp,
  w32pre_cpp,
  w32util_cpp,
  w32svc_cpp,
  #endif
  #if (CLIENT_OS == OS_OS2)
  os2inst_cpp,
  #endif
  #if (CLIENT_OS == OS_NEXTSTEP)
  next_sup_cpp,
  #endif
#if (CLIENT_OS == OS_LINUX)
  li_inst_c,
//resolv_c, // only used in some configrations
#endif
#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
  x86id_cpp,
#endif
#if defined(HAVE_RC5_72_CORES)
  core_r72_cpp,
  stress_r72_cpp,
#endif
#if defined(HAVE_OGR_PASS2)
  core_ogr_cpp,
  ogr_sup_cpp,
  ogr_dat_cpp,
#endif
#if defined(HAVE_OGR_CORES)
  core_ogr_ng_cpp,
  ogr_sup_cpp,
  ogrng_dat_cpp,
  ogrng_init_cpp,
#endif
  ((const char * (*)(void))0)
};



static const char *split_line( char *buffer, const char *p1, unsigned int bufsize )
{
  if ( p1 && buffer && bufsize > (20+15+15+15))
  {
    if (strlen(p1)>=45 && p1[5]=='I' && p1[6]=='d' && p1[7]==':' && p1[8]==' ')
    {
      unsigned int pos;
      char *p2 = buffer;
      p1 += 9;
      for ( pos=0; pos<4; pos++)
      {
        unsigned int i, len = 0, colwidth = ((pos == 0)?(20):(15));
        const char *p3;
        while ( *p1 == ' ' )
          p1++;
        p3 = p1;
        while ( *p3 && *p3 != ' ' )
        { p3++; len++; }
        if (!*p3)
          break;
        if (pos == 0 && len>2 && *(p3-1)=='v' && *(p3-2)==',')
          len-=2;
        if (len > colwidth)
          len = colwidth;
        for (i=0;i<len;i++)
          p2[i] = (char)(*p1++);
        p1 = p3;
        while (len<colwidth)
          p2[len++] = ' ';
        p2+=colwidth;
      }
      *p2 = '\0';
      return buffer;
    }
  }
  return (const char *)0;
}



void CliIdentifyModules(void)
{
  const char *p;
  unsigned int pos;
  const unsigned int cppidcount = (sizeof(ident_table)/sizeof(ident_table[0]));
  const unsigned int hidcount = (sizeof(h_ident_table)/sizeof(h_ident_table[0]));
  char buffer[80];

  for (pos = 0; pos < (cppidcount + hidcount); pos++)
  {
    p = ((const char *)0);
    if (pos < cppidcount)
    {
      if (ident_table[pos])
        p = (*ident_table[pos])();
    }
    else
      p = h_ident_table[pos-cppidcount];
    if (split_line( buffer, p, sizeof(buffer)))
      LogScreenRaw( "%s\n", buffer );
  }
  return;
}



/*
 *  Get Date/Time the newest module was committed. Used, for instance, to
 *  'ensure' that time from the .ini or recvd from a proxy is sane.
 */
time_t CliGetNewestModuleTime(void)
{
  static time_t newest = (time_t)0;
  if (newest == (time_t)0)
  {
    char buffer[80];
    const char *p; struct tm bd; unsigned int pos; time_t bdtime;
    unsigned int cppidcount = (sizeof(ident_table)/sizeof(ident_table[0]));
    unsigned int hidcount = (sizeof(h_ident_table)/sizeof(h_ident_table[0]));

    for (pos = 0; pos < (cppidcount + hidcount); pos++)
    {
      p = ((const char *)0);
      if (pos < cppidcount)
      {
        if (ident_table[pos])
          p = (*ident_table[pos])();
      }
      else
        p = h_ident_table[pos-cppidcount];
      if (split_line( buffer, p, sizeof(buffer)))
      {
        #define BD_DATEPOS 35
        #define BD_TIMEPOS (BD_DATEPOS+15)
        // 0         1         2         3         4         5
        // 012345678901234567890123456789012345678901234567890123456789
        // cliident.cpp,v      1.14           1999/04/05     17:56:51
        // cliident.h,v        1.4            1999/04/06     10:20:47
        memset((void *)&bd,-1,sizeof(bd));
        bd.tm_year = atoi(&buffer[BD_DATEPOS+0])-1900;
        bd.tm_mon  = atoi(&buffer[BD_DATEPOS+5])-1;
        bd.tm_mday = atoi(&buffer[BD_DATEPOS+8]);
        bd.tm_hour = atoi(&buffer[BD_TIMEPOS+0]);
        bd.tm_min  = atoi(&buffer[BD_TIMEPOS+3]);
        bd.tm_sec  = atoi(&buffer[BD_TIMEPOS+6]);
        #undef BD_DATEPOS
        #undef BD_TIMEPOS
        if (bd.tm_mon >= 0  && bd.tm_mon  <= 11 &&
           bd.tm_mday >= 1  && bd.tm_mday <= 31 &&
           bd.tm_year >= 99 && bd.tm_year <= 132 &&
           bd.tm_hour >= 0  && bd.tm_hour <= 23 &&
           bd.tm_min  >= 0  && bd.tm_min  <= 59 &&
           bd.tm_sec  >= 0  && bd.tm_sec  <= 61 )
        {
          bd.tm_isdst = -1;
          bdtime = mktime( &bd );
          if (bdtime != ((time_t)-1))
          {
            if (bdtime > newest)
              newest = bdtime;
          }
        }
      }
    }
    #ifdef __DATE__
    if (!newest)
    {
      p = __DATE__;
      for (pos=0;p[3]==' ' && pos<12;pos++)
      {
        const char *monnames="JanFebMarAprMayJunJulAugSepOctNovDec";
        if (memcmp(&monnames[pos*3],p,3)==0)
        {
          memset((void *)&bd,0,sizeof(bd));
          bd.tm_mon = pos;
          bd.tm_mday = atoi(p+4);
          bd.tm_year = (atoi(p+7) - 1900);
          #ifdef __TIME__
          p = __TIME__;
          bd.tm_hour = atoi(p);
          bd.tm_min = atoi(p+3);
          bd.tm_sec = atoi(p+6);
          if (bd.tm_hour<0 || bd.tm_hour>23 || bd.tm_min<0 || bd.tm_min>59)
            bd.tm_hour = bd.tm_min = bd.tm_sec = 0;
          else if (bd.tm_sec < 0 || bd.tm_sec > 61)
            bd.tm_sec = 0;
          #endif
          if (bd.tm_mon >= 0 && bd.tm_mon <= 11 &&
              bd.tm_mday >= 1 && bd.tm_mday <= 31 &&
              bd.tm_year >= 99 && bd.tm_year <= 132 )
          {
            bd.tm_isdst = -1;
            bdtime = mktime( &bd );
            if (bdtime != ((time_t)-1))
              newest = bdtime;
          }
          break;
        }
      }
    }
    #endif
    if (newest)
    {
      time_t zudiff = time(NULL);
      struct tm *tmP = localtime( (const time_t *) &zudiff);
      if (tmP)
      {
        memcpy( &bd, tmP, sizeof( struct tm ));
        tmP = gmtime( (const time_t *) &zudiff);
        if (tmP)
        {
          struct tm utctime;
          memcpy( &utctime, tmP, sizeof( struct tm ));
          zudiff =  ((bd.tm_min  - utctime.tm_min)*60 )
                   +((bd.tm_hour - utctime.tm_hour)*3600 );
          /* last two are when the time is on a year boundary */
          if      (bd.tm_yday == utctime.tm_yday)     { /* no change */ ; }
          else if (bd.tm_yday == utctime.tm_yday + 1) { zudiff += 86400L; }
          else if (bd.tm_yday == utctime.tm_yday - 1) { zudiff -= 86400L; }
          else if (bd.tm_yday <  utctime.tm_yday)     { zudiff += 86400L; }
          else                                        { zudiff -= 86400L; }
          newest += zudiff;
        }
      }
    }
  }
  return newest;
}


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
  static char buffer[10+32+sizeof("v"CLIENT_VERSIONSTRING"-XXX-99071523-*dev* client for "CLIENT_OS_NAME_EXTENDED)];
  struct timeval tv; tv.tv_usec = 0;
  tv.tv_sec = CliGetNewestModuleTime();
  sprintf( buffer, "%s v" CLIENT_VERSIONSTRING "-"
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
         "-%s" /* date is in bugzilla format yymmddhh */
         "%s"  /* "-*dev*" or "" */
         " for "CLIENT_OS_NAME_EXTENDED,
         utilGetAppName(),
         ((ConIsGUI())?('G'):('C')),
         ((CliIsDevelVersion())?('L'):('R')),
         CliGetTimeString(&tv,4),
         ((CliIsDevelVersion())?("-*dev*"):("")) );
  return buffer;
}
