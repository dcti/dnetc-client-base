/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * The file contains CliIdentifyModules() which lists the cvs id strings
 * to stdout. Users can assist us (when making bug reports) by telling us 
 * exactly which modules were actually in effect when the binary was made. 
 * Currently, starting the client with the '-ident' switch will exec the 
 * function.
 *
 * This file also contains CliGetNewestModuleTime() which returns time_t of  
 * the newest module in the list - useful for asserting beta expiry, that the
 * time obtained from proxies is sane etc.
 * ----------------------------------------------------------------------
*/ 
const char *cliident_cpp(void) { 
return "@(#)$Id: cliident.cpp,v 1.17 1999/05/06 17:49:46 cyp Exp $"; } 

#include "cputypes.h"
#include "baseincs.h"
#include "autobuff.h"
#include "base64.h"
#include "bench.h"
#include "buffupd.h"
#include "client.h"
#include "buffwork.h"
#include "ccoreio.h"
#include "checkpt.h"
#include "clicdata.h"
#include "clievent.h"
#include "cliident.h"
#include "clirate.h"
#include "clisrate.h"
#include "clitime.h"
#include "cmdline.h"
#include "cmpidefs.h"
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
//#include "packets.h"
#include "pathwork.h"
#include "pollsys.h"
#include "probfill.h"
#include "problem.h"
#include "probman.h"
#include "random.h"
#include "rsadata.h"
//#include "scram.h"
#include "selcore.h"
#include "selftest.h"
#include "setprio.h"
#include "sleepdef.h"
#include "threadcd.h"
#include "triggers.h"
//#include "u64class.h"
#include "util.h"
#include "version.h"

static const char *h_ident_table[] = {
(const char *)__AUTOBUFF_H__,
(const char *)__BASE64_H__,
(const char *)__BASEINCS_H__,
(const char *)__BENCH_H__,
(const char *)__BUFFUPD_H__,
(const char *)__BUFFWORK_H__,
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
(const char *)__CMPIDEFS_H__,
(const char *)__CONFOPT_H__,
(const char *)__CONFRWV_H__,
(const char *)__CONSOLE_H__,
(const char *)__CONVDES_H__,
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
//(const char *)__PACKETS_H__,
(const char *)__PATHWORK_H__,
(const char *)__POLLSYS_H__,
(const char *)__PROBFILL_H__,
(const char *)__PROBLEM_H__,
(const char *)__PROBMAN_H__,
(const char *)__RANDOM_H__,
(const char *)__RSADATA_H__,
//(const char *)__SCRAM_H__,
(const char *)__SELCORE_H__,
(const char *)__SELFTEST_H__,
(const char *)__SETPRIO_H__,
(const char *)__SLEEPDEF_H__,
(const char *)__THREADCD_H__,
//(const char *)__TRIGGERS_H__,
//(const char *)__U64CLASS_H__,
(const char *)__UTIL_H__,
(const char *)__VERSION_H__
};


extern const char *buffupd_cpp(void);
extern const char *clicdata_cpp(void);
extern const char *clirate_cpp(void);
extern const char *convdes_cpp(void);
extern const char *autobuff_cpp(void);
extern const char *buffwork_cpp(void);
extern const char *iniread_cpp(void);
//extern const char *scram_cpp(void);
extern const char *clitime_cpp(void);
extern const char *cliident_cpp(void);
extern const char *confopt_cpp(void);
extern const char *confrwv_cpp(void);
extern const char *client_cpp(void);
extern const char *disphelp_cpp(void);
extern const char *netinit_cpp(void);
extern const char *mail_cpp(void);
extern const char *pathwork_cpp(void);
extern const char *problem_cpp(void);
extern const char *logstuff_cpp(void);
extern const char *lurk_cpp(void);
extern const char *clisrate_cpp(void);
//extern const char *netres_cpp(void);
extern const char *triggers_cpp(void);
//extern const char *memfile_cpp(void);
extern const char *selcore_cpp(void);
extern const char *selftest_cpp(void);
extern const char *cpucheck_cpp(void);
extern const char *cmdline_cpp(void);
extern const char *probfill_cpp(void);
extern const char *pollsys_cpp(void);
extern const char *clirun_cpp(void);
extern const char *setprio_cpp(void);
extern const char *bench_cpp(void);
extern const char *network_cpp(void);
extern const char *probman_cpp(void);
extern const char *console_cpp(void);

static const char * (*ident_table[])() = {
buffupd_cpp,
clicdata_cpp,
clirate_cpp,
convdes_cpp,
autobuff_cpp,
buffwork_cpp,
iniread_cpp,
//scram_cpp,
clitime_cpp,
cliident_cpp,
confopt_cpp,
confrwv_cpp,
client_cpp,
disphelp_cpp,
netinit_cpp,
mail_cpp,
pathwork_cpp,
problem_cpp,
logstuff_cpp,
#ifdef LURK
lurk_cpp,
#endif
clisrate_cpp,
//netres_cpp,
triggers_cpp,
//memfile_cpp,
selcore_cpp,
selftest_cpp,
cpucheck_cpp,
cmdline_cpp,
probfill_cpp,
pollsys_cpp,
clirun_cpp,
setprio_cpp,
bench_cpp,
network_cpp,
probman_cpp,
console_cpp
};

static const char *split_line( char *buffer, const char *p1, unsigned int bufsize )
{
  if ( p1 != NULL )
  {
    unsigned int pos;
    char *p2 = buffer;
    char *p3 = &buffer[bufsize-2];
    p1 += 9;
    for ( pos = 0; pos < 4; pos++)
    {
      unsigned int len = 0;
      while ( *p1 != 0 && *p1 == ' ' )
        p1++;
      while ( p2<p3 && *p1 != 0 && *p1 != ' ' )
        { *p2++ = *p1++; len++; }
      if ( p2>=p3 || *p1 == 0 )
        break;
      if (pos != 0) 
        len+=10;
      do
      { *p2++ = ' ';
      } while (p2 < p3 && (++len) < 20);
    }
    *p2 = 0;
    if ( p2 != buffer )
      return buffer;
  }
  return (const char *)0;
}  


void CliIdentifyModules(void)
{
  unsigned int idline = sizeof(h_ident_table); /* squelch warning */
  for (idline = 0; idline < (sizeof(ident_table)/sizeof(ident_table[0])); idline++)
  {
    char buffer[80];
    if ((split_line( buffer, (*ident_table[idline])(), 
                            sizeof(buffer) )) != ((const char *)0))
      LogScreenRaw( "%s\n", buffer );
  }
  return;
}  

time_t CliGetNewestModuleTime(void)
{
  static time_t newest = (time_t)0;
  if (newest == (time_t)0)
  {
    unsigned int idline;
    unsigned int cppidcount = (sizeof(ident_table)/sizeof(ident_table[0]));
    unsigned int hidcount = (sizeof(h_ident_table)/sizeof(h_ident_table[0]));
    for (idline = 0; idline < (cppidcount + hidcount); idline++)
    {
      char buffer[80];
      const char *p = ((const char *)0);
      if (idline < cppidcount)
        p = (*ident_table[idline])();
      else
        p = h_ident_table[idline-cppidcount];
      if ((split_line( buffer, p, sizeof(buffer) )) != ((const char *)0))
      {
        struct tm bd;
        // 0         1         2         3         4
        // 01234567890123456789012345678901234567890123456789
        // cliident.cpp,v      1.14      1999/04/05 17:56:51  
        // cliident.h,v        1.4       1999/04/06 10:20:47  
        memset((void *)&bd,0,sizeof(bd));
        bd.tm_year = atoi(&buffer[30])-1900;
        bd.tm_mon  = atoi(&buffer[35])-1;
        bd.tm_mday = atoi(&buffer[38]);
        bd.tm_hour = atoi(&buffer[41]);
        bd.tm_min  = atoi(&buffer[44]);
        bd.tm_sec  = atoi(&buffer[47]);
        if (bd.tm_mon >= 0  && bd.tm_mon  <= 11 && 
           bd.tm_mday >= 1  && bd.tm_mday <= 31 &&
           bd.tm_year >= 99 && bd.tm_year <= 132 &&
           bd.tm_hour >= 0  && bd.tm_hour <= 23 &&
           bd.tm_min  >= 0  && bd.tm_min  <= 59 &&
           bd.tm_sec  >= 0  && bd.tm_sec  <= 61 )
        {
          time_t bdtime = mktime( &bd );
          if (bdtime != ((time_t)-1))
          {
            bdtime -= (CliTimeGetMinutesWest() * 60);
            if (bdtime > newest)
              newest = bdtime;
          }
        }
      }
    }
  }
  return newest;
}  
