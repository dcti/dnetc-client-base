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
 * ----------------------------------------------------------------------
*/ 
const char *cliident_cpp(void) { 
return "@(#)$Id: cliident.cpp,v 1.14 1999/04/05 17:56:51 cyp Exp $"; } 

/* --------------------------------------------------------------------- */

#include <stdio.h>
#include <string.h>
#include "logstuff.h" //LogScreen()
#include "cliident.h" //just to keep the prototypes in sync.

extern const char *buffupd_cpp(void);
extern const char *clicdata_cpp(void);
extern const char *clirate_cpp(void);
extern const char *convdes_cpp(void);
extern const char *autobuff_cpp(void);
extern const char *buffwork_cpp(void);
extern const char *iniread_cpp(void);
extern const char *scram_cpp(void);
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
scram_cpp,
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

//"@(#)$Id: cliident.cpp,v 1.14 1999/04/05 17:56:51 cyp Exp $"

void CliIdentifyModules(void)
{
  unsigned int i;
  for (i = 0; i < (sizeof(ident_table)/sizeof(ident_table[0])); i++)
  {
    //LogScreen( "%s\n", (*ident_table[i])() );
    const char *p1 = (*ident_table[i])();
    if ( p1 != NULL )
    {
      char buffer[76];
      char *p2 = &buffer[0];
      char *p3 = &buffer[sizeof(buffer)-2];
      p1 += 9;
      for (unsigned int pos = 0; pos < 4; pos++)
      {
        while ( *p1 != 0 && *p1 == ' ' )
          p1++;
        unsigned int len = 0;
        while ( p2<p3 && *p1 != 0 && *p1 != ' ' )
          { *p2++ = *p1++; len++; }
        if ( p2>=p3 || *p1 == 0 )
          break;
        if (pos != 0) 
          len+=10;
        do
        {
          *p2++ = ' ';
        } while (p2 < p3 && (++len) < 20);
      }
      *p2 = 0;
      if ( p2 != &buffer[0] )
        LogScreen( "%s\n", buffer );
    }
  }
  return;
}

