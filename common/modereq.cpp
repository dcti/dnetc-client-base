// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// $Log: modereq.cpp,v $
// Revision 1.6.2.6  1998/12/29 19:58:59  remi
// A small fix, and synced with :
//
//  Revision 1.16  1998/12/28 21:05:55  cyp
//  Removed CLIENT_OS specific stuff. MacOS! Get in line!
//
// Revision 1.6.2.5  1998/12/28 15:11:49  remi
// Synced with :
//  Revision 1.13  1998/12/08 05:48:59  dicamillo
//  For MacOS GUI client, add calls to create and destroy benchmark display.
//
// Revision 1.6.2.4  1998/11/15 11:07:20  remi
// Synced with :
//  Revision 1.10  1998/11/15 11:00:17  remi
//  Moved client->SelectCore() for -test and -benchmark* from cmdline.cpp to
//  modereq.cpp and told it to not be quiet.
//
// Revision 1.6.2.3  1998/11/11 03:11:08  remi
// Synced with :
//  Revision 1.7  1998/11/08 19:03:21  cyp
//  -help (and invalid command line options) are now treated as "mode" requests.
//
// Revision 1.6.2.2  1998/11/08 11:51:32  remi
// Lots of $Log tags.
//

#if (!defined(lint) && defined(__showids__))
const char *modereq_cpp(void) {
return "@(#)$Id: modereq.cpp,v 1.6.2.6 1998/12/29 19:58:59 remi Exp $"; }
#endif

#include "client.h"    //client class
#include "triggers.h"  //CheckExitRequestTrigger() [used by bench stuff]
#include "logstuff.h"  //LogScreen() [used by update/fetch/flush stuff]
#include "modereq.h"   //our constants

#include "cpucheck.h"  //"mode" DisplayProcessorInformation()
#include "selftest.h"  //"mode" SelfTest()
#include "bench.h"     //"mode" Benchmark()

/* --------------------------------------------------------------- */

static struct
{
  int isrunning;
  int reqbits;
  const char *helpoption;
} modereq = {0,0,(const char *)0};

/* --------------------------------------------------------------- */
 
int ModeReqSetArg(int mode, void *arg )
{
  if (mode == MODEREQ_CMDLINE_HELP)
    {
    ModeReqSet(MODEREQ_CMDLINE_HELP);
    modereq.helpoption = (const char *)arg;
    return 0;
    }
  return -1;
}  
  
/* --------------------------------------------------------------- */

int ModeReqIsSet(int modemask)
{
  return ((modereq.reqbits & modemask) != 0);
}

/* --------------------------------------------------------------- */

int ModeReqSet(int modemask)
{
  if (modemask == -1)
    modemask = MODEREQ_ALL;
  int oldmask = (modereq.reqbits & modemask);
  modereq.reqbits |= modemask;
  return oldmask;
}

/* --------------------------------------------------------------- */

int ModeReqClear(int modemask)
{
  int oldmask;
  if (modemask == -1)
    {
    oldmask = modereq.reqbits;
    modereq.reqbits = 0;
    }
  else
    {
    modemask &= MODEREQ_ALL;
    oldmask = (modereq.reqbits & modemask);
    modereq.reqbits ^= (modereq.reqbits & modemask);
    }
  return oldmask;
}

/* --------------------------------------------------------------- */

int ModeReqIsRunning(void)
{
  return (modereq.isrunning != 0);
}

/* --------------------------------------------------------------- */

int ModeReqRun(Client *client)
{
  int retval = 0;
  
  if (++modereq.isrunning == 1)
    {
    while ((modereq.reqbits & MODEREQ_ALL)!=0)
      {
      unsigned int bits = modereq.reqbits;
      
      if ((bits & MODEREQ_CMDLINE_HELP) && client)
	{
	client->DisplayHelp();
        modereq.reqbits &= ~(MODEREQ_CMDLINE_HELP);
        retval |= (MODEREQ_CMDLINE_HELP);
	}

      if ((bits & (MODEREQ_BENCHMARK_DES | MODEREQ_BENCHMARK_RC5)) != 0)
        {
        if (client)
          {
          client->SelectCore( 0 /* not quietly */ );
          u32 benchsize = (1L<<23); /* long bench: 8388608 instead of 100000000 */
          if ((bits & (MODEREQ_BENCHMARK_QUICK))!=0)
            benchsize = (1L<<20); /* short bench: 1048576 instead of 10000000 */
          if ( !CheckExitRequestTriggerNoIO() && (bits&MODEREQ_BENCHMARK_RC5)!=0) 
            Benchmark( 0, benchsize, client->cputype );
          if ( !CheckExitRequestTriggerNoIO() && (bits&MODEREQ_BENCHMARK_DES)!=0) 
            Benchmark( 1, benchsize, client->cputype );
          }
        retval |= (modereq.reqbits & (MODEREQ_BENCHMARK_DES | 
                 MODEREQ_BENCHMARK_RC5 | MODEREQ_BENCHMARK_QUICK ));
        modereq.reqbits &= ~(MODEREQ_BENCHMARK_DES | 
                 MODEREQ_BENCHMARK_RC5 | MODEREQ_BENCHMARK_QUICK );
        }
      if ((bits & MODEREQ_CPUINFO)!=0)
        {
        DisplayProcessorInformation(); 
        modereq.reqbits &= ~(MODEREQ_CPUINFO);
        retval |= (MODEREQ_CPUINFO);
        }
      if ((bits & MODEREQ_TEST)!=0)
        {
        if (client)
          {
          client->SelectCore( 0 /* not quietly */ );
          if ( SelfTest(0, client->cputype ) > 0 ) 
            SelfTest(1, client->cputype );
          }
        retval |= (MODEREQ_TEST);
        modereq.reqbits &= ~(MODEREQ_TEST);
        }
      } //end while
    } //if (++isrunning == 1)

  modereq.isrunning--;
  return retval;
}

