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
 * ---------------------------------------------------------------
*/    
const char *modereq_cpp(void) {
return "@(#)$Id: modereq.cpp,v 1.28.2.4 1999/10/07 18:38:58 cyp Exp $"; }

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
  const char *filetounlock;
  const char *filetoimport;
  const char *helpoption;
} modereq = {0,0,(const char *)0,(const char *)0,(const char *)0};

/* --------------------------------------------------------------- */

int ModeReqSetArg(int mode, void *arg)
{
  if (mode == MODEREQ_UNLOCK)
    modereq.filetounlock = (const char *)arg;
  else if (mode == MODEREQ_IMPORT)
    modereq.filetoimport = (const char *)arg;
  else if (mode == MODEREQ_CMDLINE_HELP)
    modereq.helpoption = (const char *)arg;
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
      if ((bits & (MODEREQ_BENCHMARK_QUICK | MODEREQ_BENCHMARK_ALL)) != 0)
      {
        if (client)
        {
          #if (CONTEST_COUNT != 4)
          #error This needs fixing
          #endif        
          static int bmask2cid[CONTEST_COUNT] = 
          {
            MODEREQ_BENCHMARK_RC5,
            MODEREQ_BENCHMARK_DES,
            MODEREQ_BENCHMARK_OGR,
            MODEREQ_BENCHMARK_CSC
          };
          unsigned int contest, benchsecs = 16;
          if ((bits & (MODEREQ_BENCHMARK_QUICK))!=0)
            benchsecs = 8;
          for (contest = 0; contest < CONTEST_COUNT; contest++)
          {
            if (CheckExitRequestTriggerNoIO())
              break;
            if ((bits & MODEREQ_BENCHMARK_ALL)==0 || /*none set==all set*/
                (bits & bmask2cid[contest]) != 0)
            {
              retval |= bmask2cid[contest];
              TBenchmark( contest, benchsecs, -1, 0 );
            }
          }
        }    
        modereq.reqbits &= ~(MODEREQ_BENCHMARK_QUICK | MODEREQ_BENCHMARK_ALL);
      }
      if ((bits & MODEREQ_CMDLINE_HELP) != 0)
      {
        DisplayHelp(modereq.helpoption);
        modereq.helpoption = (const char *)0;
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
          domode = BufferUpdate( client, domode, interactive );
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
        if (modereq.filetounlock)
        {
          UnlockBuffer(modereq.filetounlock);
          modereq.filetounlock = (const char *)0;
        }
        modereq.reqbits &= ~(MODEREQ_UNLOCK);
        retval |= (MODEREQ_UNLOCK);
      }
      if ((bits & MODEREQ_IMPORT)!=0)
      {
        if (modereq.filetoimport && client)
        {
          BufferImportFileRecords(client, modereq.filetoimport, 1 /* interactive */);
          modereq.filetoimport = (const char *)0;
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
          for (contestid = 0; contestid < CONTEST_COUNT; contestid++ ) 
          {
            if ( SelfTest(contestid, -1 ) < 0 ) 
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

