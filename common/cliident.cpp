/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
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
return "@(#)$Id: cliident.cpp,v 1.17.2.13 2000/05/08 11:11:34 cyp Exp $"; } 

#include "cputypes.h"
#include "baseincs.h"
#include "autobuff.h"
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
#include "clirate.h"
#include "clisrate.h"
#include "clitime.h"
#include "cmdline.h"
#include "confopt.h"
#include "confrwv.h"
#include "console.h"
#include "convdes.h"
#include "cpucheck.h"
#include "disphelp.h"
#include "iniread.h"
#include "logstuff.h"
#include "lurk.h"
#include "mail.h"
#include "memfile.h"
#include "modereq.h"
#include "network.h"
#include "pathwork.h"
#include "pollsys.h"
#include "probfill.h"
#include "problem.h"
#include "probman.h"
#include "random.h"
#include "rsadata.h"
#include "selcore.h"
#include "selftest.h"
#include "setprio.h"
#include "sleepdef.h"
#include "triggers.h"
#include "util.h"
#include "version.h"

static const char *h_ident_table[] = 
{
  (const char *)__AUTOBUFF_H__,
  (const char *)__BASE64_H__,
  (const char *)__BASEINCS_H__,
  (const char *)__BENCH_H__,
  (const char *)__BUFFUPD_H__,
  (const char *)__BUFFBASE_H__,
  (const char *)__CHECKPT_H__,
  (const char *)__CCOREIO_H__,
  (const char *)__CLICDATA_H__,
  (const char *)__CLIENT_H__,
  (const char *)__CLIEVENT_H__,
  (const char *)__CLIIDENT_H__,
  (const char *)__CLIRATE_H__,
  (const char *)__CLISRATE_H__,
  (const char *)__CLITIME_H__,
  (const char *)__CMDLINE_H__,
  (const char *)__CONFOPT_H__,
  (const char *)__CONFRWV_H__,
  (const char *)__CONSOLE_H__,
//(const char *)__CONVDES_H__,
  (const char *)__CPUCHECK_H__,
  (const char *)__CPUTYPES_H__,
  (const char *)__DISPHELP_H__,
  (const char *)__INIREAD_H__,
  (const char *)__LOGSTUFF_H__,
  (const char *)__LURK_H__,
  (const char *)__MAIL_H__,
//(const char *)__MEMFILE_H__,
  (const char *)__MODEREQ_H__,
  (const char *)__NETWORK_H__,
  (const char *)__PATHWORK_H__,
  (const char *)__POLLSYS_H__,
  (const char *)__PROBFILL_H__,
  (const char *)__PROBLEM_H__,
  (const char *)__PROBMAN_H__,
  (const char *)__RANDOM_H__,
  (const char *)__RSADATA_H__,
  (const char *)__SELCORE_H__,
  (const char *)__SELFTEST_H__,
  (const char *)__SETPRIO_H__,
  (const char *)__SLEEPDEF_H__,
  (const char *)__TRIGGERS_H__,
  (const char *)__UTIL_H__,
  (const char *)__VERSION_H__
};

extern const char *autobuff_cpp(void);
extern const char *base64_cpp(void);
extern const char *bench_cpp(void);
extern const char *buffbase_cpp(void);
extern const char *buffupd_cpp(void);
extern const char *buffpub_cpp(void);
extern const char *checkpt_cpp(void);
extern const char *clicdata_cpp(void);
extern const char *client_cpp(void);
extern const char *clievent_cpp(void);
//extern const char *cliident_cpp(void);
extern const char *clirate_cpp(void);
extern const char *clirun_cpp(void);
extern const char *clisrate_cpp(void);
extern const char *clitime_cpp(void);
extern const char *cmdline_cpp(void);
extern const char *confmenu_cpp(void);
extern const char *confopt_cpp(void);
extern const char *confrwv_cpp(void);
extern const char *console_cpp(void);
extern const char *convdes_cpp(void);
extern const char *cpucheck_cpp(void);
extern const char *disphelp_cpp(void);
extern const char *iniread_cpp(void);
extern const char *logstuff_cpp(void);
extern const char *lurk_cpp(void);
extern const char *mail_cpp(void);
extern const char *memfile_cpp(void);
extern const char *modereq_cpp(void);
extern const char *netinit_cpp(void);
extern const char *netres_cpp(void);
extern const char *network_cpp(void);
extern const char *pathwork_cpp(void);
extern const char *pollsys_cpp(void);
extern const char *probfill_cpp(void);
extern const char *problem_cpp(void);
extern const char *probman_cpp(void);
extern const char *selcore_cpp(void);
extern const char *selftest_cpp(void);
extern const char *setprio_cpp(void);
extern const char *triggers_cpp(void);
extern const char *util_cpp(void);

static const char * (*ident_table[])() = 
{
  autobuff_cpp,
  base64_cpp,
  bench_cpp,
  buffbase_cpp,
  buffupd_cpp,
  buffpub_cpp,
  checkpt_cpp,
  clicdata_cpp,
  client_cpp,
  clievent_cpp,
  cliident_cpp,
  clirate_cpp,
  clirun_cpp,
  clisrate_cpp,
  clitime_cpp,
  cmdline_cpp,
  confmenu_cpp,
  confopt_cpp,
  confrwv_cpp,
  console_cpp,
//convdes_cpp,
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
  netinit_cpp,
  netres_cpp,
  network_cpp,
  pathwork_cpp,
  pollsys_cpp,
  probfill_cpp,
  problem_cpp,
  probman_cpp,
  selcore_cpp,
  selftest_cpp,
  setprio_cpp,
  triggers_cpp,
  util_cpp
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
  unsigned int idline = sizeof(h_ident_table); /* squelch warning */
  for (idline = 0; idline < (sizeof(ident_table)/sizeof(ident_table[0])); idline++)
  {
    char buffer[80];
    if (split_line( buffer, (*ident_table[idline])(), sizeof(buffer)))
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
        p = (*ident_table[pos])();
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

int CliIsDevelVersion(void)
{
  #if (defined(BETA) || defined(BETA_PERIOD))
  return 1;
  #else
  static int isdevel = -1;
  if (isdevel == -1)
  {
    char *p = (char*)strchr( cliident_cpp(), ',' );
    isdevel = 0;
    if (p)
    {
      if (p[1] == 'v')
      { 
        char scratch[80]; 
        unsigned int len = 0;
        p+=2;
        while (*p && *p==' ')
          p++;
        while (*p && *p!=' ' && len<(sizeof(scratch)-1))
          scratch[len++] = *p++;
        scratch[len] = '\0';
        if (strchr(scratch, '.' ) == strrchr(scratch, '.'))
          isdevel = 1;
      }
    }
  }
  return isdevel;
  #endif
}

const char *CliGetFullVersionDescriptor(void)
{
  static char buffer[10+32+sizeof("v"CLIENT_VERSIONSTRING"-XXX-99071523-*dev* client for "CLIENT_OS_NAME)];
  struct timeval tv; tv.tv_usec = 0; 
  tv.tv_sec = CliGetNewestModuleTime();
  sprintf( buffer, "%s v" CLIENT_VERSIONSTRING "-"
         "%c"  /* GUI == "G", CLI == "C" */
         #ifdef CLIENT_SUPPORTS_SMP
         "T"   /* threads */
         #else
         "P"   /* polling */
         #endif
         "%c"  /* limited release or dev branch or public release */
         "-%s" /* date is in bugzilla format yymmddhh */ 
         "%s"  /* "-*dev*" or "" */
         " for "CLIENT_OS_NAME,
         utilGetAppName(),
         ((ConIsGUI())?('G'):('C')),  
         ((CliIsDevelVersion())?('L'):('R')),
         CliGetTimeString(&tv,4),
         ((CliIsDevelVersion())?("-*dev*"):("")) );
  return buffer;
}  
