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
// Revision 1.4  1998/08/02 16:17:52  cyruspatel
// Completed support for logging.
//
// Revision 1.3  1998/07/13 23:39:32  cyruspatel
// Added functions to format and display raw cpu info for better management
// of the processor detection functions and tables. Well, not totally raw,
// but still less cooked than SelectCore(). All platforms are supported, but
// the data may not be meaningful on all. The info is accessible to the user
// though the -cpuinfo switch.
//
// Revision 1.2  1998/07/09 09:43:31  remi
// Give an error message when the user ask for '-ident' and there is no support
// for it in the client.
//
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
return "@(#)$Id: cliident.cpp,v 1.4 1998/08/02 16:17:52 cyruspatel Exp $"; } 
#endif

//-----------------------------------------------------------------------

#include <stdio.h>
#include <string.h>
#include "logstuff.h" //LogScreen()
#include "cliident.h" //just to keep the prototypes in sync.

#if defined(__showids__) //not needed if we're not showing ids anyway

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
extern const char *logstuff_cpp(void);

static const char * (*ident_table[])() = { 
   disphelp_cpp, cliconfig_cpp, buffwork_cpp, clitime_cpp, cpucheck_cpp,
   scram_cpp, cliident_cpp, problem_cpp, client_cpp, pathwork_cpp,
   threadcd_cpp, iniread_cpp, autobuff_cpp, network_cpp, convdes_cpp,
   clirate_cpp, clicdata_cpp, mail_cpp, clisrate_cpp, logstuff_cpp };

//"@(#)$Id: cliident.cpp,v 1.4 1998/08/02 16:17:52 cyruspatel Exp $"

void CliIdentifyModules(void)
{
  unsigned int i;
  for (i=0;i<(sizeof(ident_table)/sizeof(ident_table[0]));i++)
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
        do{ *p2++ = ' ';
          } while (p2<p3 && (++len)<20);
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
