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
return "@(#)$Id: modereq.cpp,v 1.34 1999/10/16 16:48:11 cyp Exp $"; }

#include "client.h"   //client class + CONTEST_COUNT
#include "baseincs.h" //basic #includes
#include "triggers.h" //CheckExitRequestTrigger() [used by bench stuff]
#include "logstuff.h" //LogScreen() [used by update/fetch/flush stuff]
#include "modereq.h"  //our constants
#include "triggers.h" //RaiseRestartRequestTrigger/CheckExitRequestTriggerNoIO
#include "console.h"  //Clear the screen after config if restarting
#include "confrwv.h"  //load/save after successful Configure()

#include "disphelp.h" //"mode" DisplayHelp()
#include "cpucheck.h" //"mode" DisplayProcessorInformation()
#include "cliident.h" //"mode" CliIdentifyModules();
#include "selcore.h"  //"mode" selcoreSelftest(), selcoreBenchmark()
#include "selftest.h" //"mode" SelfTest()
#include "bench.h"    //"mode" Benchmark()
#include "buffwork.h" //"mode" UnlockBuffer(), ImportBuffer()
#include "buffupd.h"  //"mode" BufferUpdate()
#include "confmenu.h" //"mode" Configure()

/* --------------------------------------------------------------- */

static struct
{
  int isrunning;
  int reqbits;
  const char *filetounlock;
  const char *filetoimport;
  const char *helpoption;
  unsigned long bench_projbits;
  unsigned long test_projbits;
  int cmdline_config; /* user passed --config opt, so don't do tty check */
} modereq = {0,0,(const char *)0,(const char *)0,(const char *)0,0,0,0};

/* --------------------------------------------------------------- */

int ModeReqIsProjectLimited(int mode, unsigned int contest_i)
{
  unsigned long l = 0;
  if (contest_i < CONTEST_COUNT)
  {
    if (mode == MODEREQ_BENCHMARK || 
        mode == MODEREQ_BENCHMARK_QUICK ||
        mode == MODEREQ_BENCHMARK_ALLCORE )
      l = modereq.bench_projbits;
    else if (mode == MODEREQ_TEST ||
          mode == MODEREQ_TEST_ALLCORE )
      l = modereq.test_projbits;
    l &= (1L<<contest_i);
  }
  return (l != 0);
}

/* --------------------------------------------------------------- */

int ModeReqLimitProject(int mode, unsigned int contest_i)
{
  unsigned long l;
  if (contest_i > CONTEST_COUNT) 
    return -1;
  l = 1<<contest_i;
  if (l == 0) /* wrapped */
    return -1;
  if (mode == MODEREQ_BENCHMARK || 
      mode == MODEREQ_BENCHMARK_QUICK ||
      mode == MODEREQ_BENCHMARK_ALLCORE )
    modereq.bench_projbits |= l;
  else if (mode == MODEREQ_TEST ||
           mode == MODEREQ_TEST_ALLCORE )
    modereq.test_projbits |= l;
  else
    return -1;
  return 0;
}

/* --------------------------------------------------------------- */

int ModeReqSetArg(int mode, void *arg)
{
  if (mode == MODEREQ_UNLOCK)
    modereq.filetounlock = (const char *)arg;
  else if (mode == MODEREQ_IMPORT)
    modereq.filetoimport = (const char *)arg;
  else if (mode == MODEREQ_CMDLINE_HELP)
    modereq.helpoption = (const char *)arg;
  else if (mode == MODEREQ_CONFIG)
    modereq.cmdline_config = 1;
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
    memset( &modereq, 0, sizeof(modereq));
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
      if ((bits & (MODEREQ_BENCHMARK | 
                   MODEREQ_BENCHMARK_QUICK |
                   MODEREQ_BENCHMARK_ALLCORE )) != 0)
      {
        do
        {
          unsigned int contest, benchsecs = 16;
          unsigned long sel_contests = modereq.bench_projbits;
          modereq.bench_projbits = 0;

          if ((bits & (MODEREQ_BENCHMARK_QUICK))!=0)
            benchsecs = 8;
          for (contest = 0; contest < CONTEST_COUNT; contest++)
          {
            if (CheckExitRequestTriggerNoIO())
              break;
            if (sel_contests == 0 /*none set==all set*/
             || (sel_contests & (1L<<contest)) != 0)
            {
              if ((bits & (MODEREQ_BENCHMARK_ALLCORE))!=0)
                selcoreBenchmark( contest, benchsecs );
              else
                TBenchmark( contest, benchsecs, 0 );
            }
          }
        } while (!CheckExitRequestTriggerNoIO() && modereq.bench_projbits);
        retval |= (bits & (MODEREQ_BENCHMARK_QUICK | MODEREQ_BENCHMARK |
                           MODEREQ_BENCHMARK_ALLCORE));
        modereq.reqbits &= ~(MODEREQ_BENCHMARK_QUICK | MODEREQ_BENCHMARK |
                             MODEREQ_BENCHMARK_ALLCORE);
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
        /* <BovineMoo> okay, so if -config is 
        //   explicitly specified, then do not check isatty().
        */
        int ttycheckok = 1;
        if (!modereq.cmdline_config) /* not started by cmdline --config */
        {
          if (!ConIsScreen())
          {
            ConOutErr("Screen output is redirected/not available. Please use --config\n");
            ttycheckok = 0;
          }
        }
        if (ttycheckok && client)
        {
          Client *newclient = new Client;
          if (!newclient)
            LogScreen("Unable to configure. (Insufficient memory)");
          else
          {
            strcpy(newclient->inifilename, client->inifilename );
            if ( ReadConfig(newclient) >= 0 ) /* no or non-fatal error */
            {
              if ( Configure(newclient) > 0 ) /* success */
              {
                WriteConfig(newclient,1);
                retval |= (bits & (MODEREQ_CONFIG | MODEREQ_CONFRESTART));
              }
              if ((bits & MODEREQ_CONFRESTART) != 0)
                restart = 1;
            }
            delete newclient;
          }
        }
        modereq.cmdline_config = 0;
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
      if ((bits & (MODEREQ_TEST | MODEREQ_TEST_ALLCORE))!=0)
      {
        int testfailed = 0;
        do
        {
          unsigned int contest; 
          unsigned long sel_contests = modereq.test_projbits;
          modereq.test_projbits = 0;

          for (contest = 0; !testfailed && contest < CONTEST_COUNT; contest++)
          {
            if (CheckExitRequestTriggerNoIO())
            {
              testfailed = 1;
              break;
            }
            if (sel_contests == 0 /*none set==all set*/
             || (sel_contests & (1L<<contest)) != 0)
            {
              if ((bits & (MODEREQ_TEST_ALLCORE))!=0)
              {
                if (selcoreSelfTest( contest ) < 0)
                  testfailed = 1;
              }
              else if ( SelfTest( contest ) < 0 ) 
                 testfailed = 1;
            }
          }
        } while (!testfailed && modereq.test_projbits);
        retval |= (MODEREQ_TEST|MODEREQ_TEST_ALLCORE);
        modereq.reqbits &= ~(MODEREQ_TEST|MODEREQ_TEST_ALLCORE);
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

