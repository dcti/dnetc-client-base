// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// The file contains CliIdentifyModules() which lists the cvs id strings
// to stdout. Users can assist us (when making bug reports) by telling us 
// exactly which modules were actually in effect when the binary was made. 
// Currently, starting the client with the '-ident' switch will exec the 
// function.
// ----------------------------------------------------------------------
//
// $Log: cliident.cpp,v $
// Revision 1.13  1999/01/29 19:17:28  jlawson
// fixed formatting.
//
// Revision 1.12  1999/01/01 02:45:14  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.11  1998/12/29 22:35:42  cyp
// removed guistuff.cpp
//
// Revision 1.10  1998/12/26 21:23:05  cyp
// Removed threadcd.
//
// Revision 1.9  1998/11/22 14:56:25  cyp
// Removed cliconfig.cpp; Added confopt.cpp, confrwv.cpp
//
// Revision 1.8  1998/11/02 04:41:25  cyp
// Removed references to netres_cpp.
//
// Revision 1.7  1998/10/04 20:46:11  remi
// LogScreen -> LogScreenRaw
//
// Revision 1.6  1998/10/04 19:44:41  remi
// Added guistuff, console and probman.
//
// Revision 1.5  1998/09/28 12:52:11  cyp
// updated. woo-hoo.
//
// Revision 1.4  1998/08/02 16:17:52  cyruspatel
// Completed support for logging.
//
// Revision 1.3  1998/07/13 23:39:32  cyruspatel
// Added cpucheck.cpp
//
// Revision 1.2  1998/07/09 09:43:31  remi
// Give an error message when the user ask for '-ident' and there is no support
// for it in the client.
//
// Revision 1.1  1998/07/07 21:55:20  cyruspatel
// Created.
//

#if (!defined(lint) && defined(__showids__))
const char *cliident_cpp(void) { 
return "@(#)$Id: cliident.cpp,v 1.13 1999/01/29 19:17:28 jlawson Exp $"; } 
#endif

//-----------------------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include "logstuff.h" //LogScreen()
#include "cliident.h" //just to keep the prototypes in sync.

#if defined(__showids__) //not needed if we're not showing ids anyway

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
//extern const char *lurk_cpp(void);
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
//lurk_cpp,
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

//"@(#)$Id: cliident.cpp,v 1.13 1999/01/29 19:17:28 jlawson Exp $"

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

#else //#if defined(__showids__)

void CliIdentifyModules(void)
{
  LogScreen( "No support for -ident in this client.\n" );
}
  
#endif //#if defined(__showids__)


