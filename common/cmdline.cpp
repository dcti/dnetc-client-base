// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: cmdline.cpp,v $
// Revision 1.92.2.4  1998/11/15 11:10:52  remi
// Synced with :
//  Revision 1.98  1998/11/15 11:00:16  remi
//  Moved client->SelectCore() for -test and -benchmark* from cmdline.cpp to
//  modereq.cpp and told it to not be quiet.
//
// Revision 1.92.2.3  1998/11/11 03:09:41  remi
// Synced with :
//
//  Revision 1.96  1998/11/10 21:45:59  cyp
//  ParseCommandLine() is now one-pass. (a second pass is available so that
//  lusers can see the override messages on the screen)
//
//  Revision 1.93  1998/11/08 19:03:22  cyp
//  -help (and invalid command line options) are now treated as "mode" requests.
//
// Revision 1.92.2.2  1998/11/08 11:50:51  remi
// Lots of $Log tags.
//

#if (!defined(lint) && defined(__showids__))
const char *cmdline_cpp(void) {
return "@(#)$Id: cmdline.cpp,v 1.92.2.4 1998/11/15 11:10:52 remi Exp $"; }
#endif

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "modereq.h"   // get/set/clear mode request bits

/* -------------------------------------- */

int Client::ParseCommandline( int run_level, int argc, const char *argv[], 
                              int *retcodeP, int logging_is_initialized )
{
  int inimissing, pos, skip_next, terminate_app, havemode;
  const char *thisarg, *nextarg;

  terminate_app = 0;
  inimissing = 0;
  havemode = 0;

  //-----------------------------------
  // In the next loop we parse the other options
  //-----------------------------------

  if (!terminate_app && ((run_level == 0) || (logging_is_initialized)))
    {
    for (pos = 1;pos<argc; pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;

      if ( thisarg == NULL )
        { //nothing
        }
      else if (*thisarg == 0)
        { //nothing
        }
      else if (
          ( strcmp( thisarg, "-cpuinfo"     ) == 0 ) ||
          ( strcmp( thisarg, "-test"        ) == 0 ) ||
          ( strncmp( thisarg, "-benchmark", 10 ) == 0))
        {
        ; //nothing - handled in next loop
        havemode = 1;
        }
      else if (run_level==0)
        {
        ModeReqClear(-1); /* clear all */
        ModeReqSet( MODEREQ_CMDLINE_HELP );
        ModeReqSetArg(MODEREQ_CMDLINE_HELP,(void *)thisarg);
        inimissing = 0; // don't need an .ini file if we just want help
        havemode = 0;
        break;
        }
      }
    ValidateConfig();
    }
        
  //-----------------------------------
  // In the final loop we parse the "modes".
  //-----------------------------------

  if (!terminate_app && havemode && run_level == 0)
    {
    for (pos = 1;pos<argc;pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;
  
      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
      else if ( strcmp( thisarg, "-cpuinfo" ) == 0 )
        {
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -cpuinfo
        ModeReqSet( MODEREQ_CPUINFO );
        break;
        }
      else if ( strcmp( thisarg, "-test" ) == 0 )
        {
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -test
        ModeReqSet( MODEREQ_TEST );
        break;
        }
      else if (strncmp( thisarg, "-benchmark", 10 ) == 0)
        {
        int do_mode = 0;
        thisarg += 10;

        if (*thisarg == '2')
          {
          do_mode |= MODEREQ_BENCHMARK_QUICK;
          thisarg++;
          }
        if ( strcmp( thisarg, "rc5" ) == 0 )  
          do_mode |= MODEREQ_BENCHMARK_RC5;
        else if ( strcmp( thisarg, "des" ) == 0 )
           do_mode |= MODEREQ_BENCHMARK_DES;
        else 
          do_mode |= (MODEREQ_BENCHMARK_DES | MODEREQ_BENCHMARK_RC5);

        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do benchmark
        ModeReqSet( do_mode );
        break;
        }
      }
    }

  if (retcodeP) 
    *retcodeP = 0;
  return terminate_app;
}
