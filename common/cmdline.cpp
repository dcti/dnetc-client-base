// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "modereq.h"   // get/set/clear mode request bits

/* ------------------------------------------------------------------------
 * runlevel == 0 = pre-anything    (-quiet, -ini, -guistart etc done here)
 *          >= 1 = post-readconfig (override ini options)
 *          == 2 = run "modes"
 *
 * Sequence of events:
 *
 *   ParseCommandLine( 0, argc, argv, NULL, NULL, 0 );
 *   if ( InitializeLogging() == 0 ) //let -quiet take affect
 *     {
 *     inimissing = ReadConfig();
 *     if (ParseCommandLine( X, argc, argv, &inimissing, &retcode, x )==0)
 *       {                   |                                     
 *       if (inimissing)     `-- X==2 for OS_xxx that do "modes", 1 for others
 *         { 
 *         Configure() ...  
 *         }
 *       else 
 *         {
 *         if ( RunStartup() == 0 )
 *           {
 *           ValidateConfig()
 *           Run();
 *           RunShutdown();
 *           }
 *         }
 *       }
 *     DeinitializeLogging();
 *     }
 *   
 *------------------------------------------------------------------------ */

int Client::ParseCommandline( int runlevel, int argc, const char *argv[], 
                              int *retcodeP, int logging_is_initialized )
{
  int pos, skip_next = 0, do_break = 0, retcode = 0;
  const char *thisarg, *nextarg;

  //---------------------------------------
  //first handle the options that affect option handling
  //--------------------------------------

  if (runlevel >= 0) // this is only to protect against an invalid runlevel
    {
    if (runlevel == 0) 
      {    
      ModeReqClear( -1 ); // clear all mode request bits
      }
    } //if (runlevel >= 0)

  //---------------------------------------
  // handle the other options 
  //--------------------------------------

  static const char *ignoreX[]={ /* options handled in other loops */
  "-cpuinfo","-test", "-help", "--help", "-h", "-?",
  "-benchmark2rc5","-benchmark2des","-benchmark2","-benchmarkrc5","-benchmarkdes",
  "-benchmark","" };

  if (runlevel >= 1 && !do_break)
    {
    for (pos = 1;((!do_break) && (pos<argc)); pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
  
      skip_next = -1;
      for (unsigned int i=0;i<(sizeof(ignoreX)/sizeof(ignoreX[1]));i++)
        {
        if (*thisarg == '-' && strcmp( thisarg+1, ignoreX[i]+1 )==0 )
          {
          skip_next++;
          break;
          }
        }
      if (skip_next != -1) //not found in ignoreX
        continue;
      skip_next = 0;
      
      if ( strcmp( thisarg, "-c" ) == 0)      // set cpu type
        {
        if (nextarg)
          {
          skip_next = 1;
          cputype = (s32) atoi( nextarg );
          }
        }
      }
    if (!do_break)
      ValidateConfig();
    }
      
  //---------------------------------------
  // handle the run modes
  //--------------------------------------
  
  if (runlevel >= 2 && !do_break)
    {
    for (pos = 1;((!do_break) && (pos<argc)); pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;

      if ( strcmp( thisarg, "-cpuinfo" ) == 0 )
        {
        do_break = 1;
        ModeReqClear(-1); //clear all - only do -cpuinfo
        ModeReqSet( MODEREQ_CPUINFO );
        retcode = 0; //and break out of loop
        }
      else if ( strcmp( thisarg, "-test" ) == 0 )
        {
        do_break = 1;
        ModeReqClear(-1); //clear all - only do -test
        ModeReqSet( MODEREQ_TEST );
        SelectCore( 0 /* not quietly */ );
        }
      else if (strncmp( thisarg, "-benchmark", 10 ) == 0)
        {
        do_break = 1;
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

        ModeReqClear(-1); //clear all - only do benchmark
        ModeReqSet( do_mode );
        SelectCore( 0 /* not quietly */ );
        }
      else
	{
	ModeReqSet( MODEREQ_HELP ); // display help text if no other "modes" requested
	retcode = 1;
	do_break = 1;
	}
      }
    }

  if (retcodeP && do_break) 
    *retcodeP = retcode;
  return do_break;
}
