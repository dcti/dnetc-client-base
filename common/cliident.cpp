// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// The file contains CliIdentifyModules() which lists the cvs id strings
// to stdout. Users can assist us (when making bug reports) by telling us 
// exactly which modules were actually in effect when the binary was made. 
// Currently, starting the client with the '-ident' switch will exec the 
// function.
//
// $Log: cliident.cpp,v $
// Revision 1.1  1998/07/07 21:55:20  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
//

#if (!defined(lint) && defined(__showids__))
const char *cliident_cpp(void) { 
static const char *id="@(#)$Id: cliident.cpp,v 1.1 1998/07/07 21:55:20 cyruspatel Exp $";
return id; } 
#endif

//-----------------------------------------------------------------------

#if defined(__showids__) //not needed if we're not showing ids anyway

#include <stdio.h>
#include "cliident.h" //just to keep the prototypes in sync.

extern const char *disphelp_cpp(void);
extern const char *cliconfig_cpp(void);
extern const char *buffwork_cpp(void);
extern const char *clitime_cpp(void);
extern const char *cpucheck_cpp(void);
extern const char *scram_cpp(void);
extern const char *cliident_cpp(void);
extern const char *problem_cpp(void);
extern const char *client_cpp(void);
extern const char *pathwork_cpp(void);
extern const char *threadcd_cpp(void);
extern const char *iniread_cpp(void);
extern const char *autobuff_cpp(void);
extern const char *network_cpp(void);
extern const char *convdes_cpp(void);
extern const char *clirate_cpp(void);
extern const char *clicdata_cpp(void);
extern const char *mail_cpp(void);
extern const char *clisrate_cpp(void);

static const char * (*ident_table[])() = { 
   disphelp_cpp, cliconfig_cpp, buffwork_cpp, clitime_cpp, cpucheck_cpp,
   scram_cpp, cliident_cpp, problem_cpp, client_cpp, pathwork_cpp,
   threadcd_cpp, iniread_cpp, autobuff_cpp, network_cpp, convdes_cpp,
   clirate_cpp, clicdata_cpp, mail_cpp, clisrate_cpp };

void CliIdentifyModules(void)
{
  unsigned int i;
  for (i=0;i<(sizeof(ident_table)/sizeof(ident_table[0]));i++)
    printf( "%s\n", (*ident_table[i])() );
}
  
#endif //#if defined(__showids__)
