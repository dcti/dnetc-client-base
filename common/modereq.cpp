// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

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
} modereq = {0,0};

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
      
      if ((bits & MODEREQ_HELP) && client)
	{
	client->DisplayHelp();
        modereq.reqbits &= ~(MODEREQ_HELP);
	}

      if ((bits & (MODEREQ_BENCHMARK_DES | MODEREQ_BENCHMARK_RC5)) != 0)
        {
        if (client)
          {
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

