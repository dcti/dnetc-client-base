/* Created by Cyrus Patel <cyp@fb14.uni-mainz.de> 
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ---------------------------------------------------------------
 * This file contains functions for getting/setting/clearing
 * "mode" requests (--flush,--fetch etc) and the like. Client::Run() will 
 * clear/run the modes when appropriate.
 *
 * This is a bridge module. Do not muck with the prototypes.
 * ---------------------------------------------------------------
*/    
const char *modereq_cpp(void) {
return "@(#)$Id: modereq.cpp,v 1.29 1999/06/09 15:06:17 cyp Exp $"; }

#include "client.h"   //client class + CONTEST_COUNT
#include "baseincs.h" //basic #includes
#include "triggers.h" //CheckExitRequestTrigger() [used by bench stuff]
#include "logstuff.h" //LogScreen() [used by update/fetch/flush stuff]
#include "modereq.h"  //our constants
#include "triggers.h" //RaiseRestartRequestTrigger/CheckExitRequestTriggerNoIO
#include "console.h"  //Clear the screen after config if restarting

#include "disphelp.h" //"mode" DisplayHelp()
#include "cpucheck.h" //"mode" DisplayProcessorInformation()
#include "cliident.h" //"mode" CliIdentifyModules();
#include "selftest.h" //"mode" SelfTest()
#include "bench.h"    //"mode" Benchmark()
#include "buffwork.h" //"mode" UnlockBuffer(), ImportBuffer()
#include "buffupd.h"  //"mode" BufferUpdate() flags
#include "confrwv.h"  //"mode" (actually needed to update config)

/* --------------------------------------------------------------- */

static struct
{
  int isrunning;
  int reqbits;
  const void *unlock_arg;
  const void *import_arg; /* struct { const char *fn,count; } ptr */
  const void *help_arg;
} modereq = {0,0, (const void *)0,(const void *)0,(const void *)0};

/* --------------------------------------------------------------- */

int ModeReqSetArg(int mode, void *arg )
{
  if (mode == MODEREQ_UNLOCK)
    modereq.unlock_arg = arg;
  else if (mode == MODEREQ_IMPORT)
    modereq.import_arg = arg;
  else if (mode == MODEREQ_CMDLINE_HELP)
    modereq.help_arg = arg;
  else
    return -1;
  ModeReqSet(mode);
  return 0;
}  
  
/* --------------------------------------------------------------- */

int ModeReqIsSet(int modemask)
{
  if (modemask == -1)
    modemask = MODEREQ_ALL;
  return (modereq.reqbits & modemask);
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
    int restart = ((modereq.reqbits & MODEREQ_RESTART) != 0);
    modereq.reqbits &= ~MODEREQ_RESTART;
    
    while ((modereq.reqbits & MODEREQ_ALL)!=0)
    {
      unsigned int bits = modereq.reqbits;
      if ((bits & (MODEREQ_BENCHMARK_DES | MODEREQ_BENCHMARK_RC5 | 
                                           MODEREQ_BENCHMARK_ALL)) != 0)
      {
        if (client)
        {
          client->SelectCore( 0 /* not quietly */ );
          u32 benchsize = (1L<<23); /* long bench: 8388608 instead of 100000000 */
          if ((bits & (MODEREQ_BENCHMARK_QUICK))!=0)
            benchsize = (1L<<20); /* short bench: 1048576 instead of 10000000 */
          if ((bits & MODEREQ_BENCHMARK_ALL) == MODEREQ_BENCHMARK_ALL)
          {
            unsigned int contest;
            for (contest = 0; contest < CONTEST_COUNT; contest++)
            {
              if (CheckExitRequestTriggerNoIO())
                break;
              Benchmark( contest, benchsize, client->cputype, NULL );
            }
          }
          else
          {
            if ( !CheckExitRequestTriggerNoIO() && (bits&MODEREQ_BENCHMARK_RC5)!=0) 
              Benchmark( 0, benchsize, client->cputype, NULL );
            if ( !CheckExitRequestTriggerNoIO() && (bits&MODEREQ_BENCHMARK_DES)!=0) 
              Benchmark( 1, benchsize, client->cputype, NULL );
          }
        }
        retval |= (modereq.reqbits & (MODEREQ_BENCHMARK_DES | 
                 MODEREQ_BENCHMARK_RC5 | MODEREQ_BENCHMARK_ALL | 
                 MODEREQ_BENCHMARK_QUICK ));
        modereq.reqbits &= ~(MODEREQ_BENCHMARK_DES | MODEREQ_BENCHMARK_RC5 | 
                 MODEREQ_BENCHMARK_ALL | MODEREQ_BENCHMARK_QUICK );
      }
      if ((bits & MODEREQ_CMDLINE_HELP) != 0)
      {
        DisplayHelp((const char *)modereq.help_arg);
        modereq.help_arg = (const void *)0;
        modereq.reqbits &= ~(MODEREQ_CMDLINE_HELP);        
        retval |= (MODEREQ_CMDLINE_HELP);
      }
      if ((bits & (MODEREQ_CONFIG | MODEREQ_CONFRESTART)) != 0)
      {
        if (client)
        {
          Client *newclient = new Client;
          if (!newclient)
            LogScreen("Unable to configure. (Insufficient memory)");
          else
          {
            strcpy(newclient->inifilename, client->inifilename );
            if ( ReadConfig(newclient) >= 0 ) /* no or non-fatal error */
            {
              if ( newclient->Configure() == 1 )
              {
                WriteConfig(newclient,1); //full new build
                retval |= (bits & (MODEREQ_CONFIG | MODEREQ_CONFRESTART));
              }
              if ((bits & MODEREQ_CONFRESTART) != 0)
                restart = 1;
            }
            delete newclient;
          }
        }
        modereq.reqbits &= ~(MODEREQ_CONFIG | MODEREQ_CONFRESTART);
      }
      if ((bits & (MODEREQ_FETCH | MODEREQ_FLUSH)) != 0)
      {
        if (client)
        {
          int domode = 0;
          int interactive = ((bits & MODEREQ_FQUIET) == 0);
          domode  = ((bits & MODEREQ_FETCH) ? BUFFERUPDATE_FETCH : 0);
          domode |= ((bits & MODEREQ_FLUSH) ? BUFFERUPDATE_FLUSH : 0);
          domode = client->BufferUpdate( domode, interactive );
          if (domode & BUFFERUPDATE_FETCH)
            retval |= MODEREQ_FETCH;
          if (domode & BUFFERUPDATE_FLUSH)
            retval |= MODEREQ_FLUSH;
          if (domode!=0 && (bits & MODEREQ_FQUIET) != 0)
            retval |= MODEREQ_FQUIET;
        }
        modereq.reqbits &= ~(MODEREQ_FETCH | MODEREQ_FLUSH | MODEREQ_FQUIET);
      }
      if ((bits & MODEREQ_IDENT) != 0)    
      {
        CliIdentifyModules();
        modereq.reqbits &= ~(MODEREQ_IDENT);
        retval |= (MODEREQ_IDENT);
      }
      if ((bits & MODEREQ_UNLOCK)!=0)
      {
        if (modereq.unlock_arg)
        {
          UnlockBuffer((const char *)modereq.unlock_arg);
          modereq.unlock_arg = (const void *)0;
          retval |= (MODEREQ_UNLOCK);
        }
        modereq.reqbits &= ~(MODEREQ_UNLOCK);
      }
      if ((bits & MODEREQ_IMPORT)!=0)
      {
        if (modereq.import_arg && client)
        {
          long count = -1; /* assume all */
          struct _argstack { const char *fn, *count; } *argstack;
          argstack = (struct _argstack *)modereq.import_arg;
          modereq.import_arg = (const void *)0;
          if (argstack->count && isdigit(*(argstack->count)))
            count = (long)atoi(argstack->count);
          BufferImportFileRecords(client, argstack->fn, 1/*interactive*/,count);
          retval |= (MODEREQ_IMPORT);
        }
        modereq.reqbits &= ~(MODEREQ_IMPORT);
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
          unsigned int contestid = 0;
          client->SelectCore( 0 /* not quietly */ );
          for (contestid = 0; contestid < CONTEST_COUNT; contestid++ ) 
          {
            if ( SelfTest(contestid, client->cputype ) < 0 ) 
              break;
          }
        }
        retval |= (MODEREQ_TEST);
        modereq.reqbits &= ~(MODEREQ_TEST);
      }
      if (CheckExitRequestTriggerNoIO())
      {
        restart = 0;
        break;
      }
    } //end while
    
    if (restart)
      RaiseRestartRequestTrigger();
  } //if (++isrunning == 1)

  modereq.isrunning--;
  return retval;
}

