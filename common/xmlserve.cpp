/* Created by Michael Feiri (michael.feiri@mfeiri.lake.de)
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/

#include "cputypes.h"
#include "netio.h"
#include "xmlserve.h"
#include "minihttp.h"
#include "cliident.h"
#include "version.h"
#include "cpucheck.h"
#include "modereq.h"
#include "problem.h"
#include "probman.h"
#include "client.h"
#include "clicdata.h"
#include "clitime.h"
#include "util.h"
#include "problem.h"
#include "clirate.h"
#include "triggers.h"

const char *FillActivity(void)
{ 
  if (ModeReqIsRunning())
  {
  /*
    int mode = ModeReqIsSet(-1);
    if (((mode & MODEREQ_CONFIG) != 0))
    
    
    if (ModeReqIsSet(MODEREQ_CONFIG))
      return "config";
    else if (ModeReqIsSet(MODEREQ_CMDLINE_HELP))
      return "help";
    else if (ModeReqIsSet(MODEREQ_RESTART))
      return "restarting";
    else if (ModeReqIsSet(MODEREQ_BENCHMARK_MASK))
      return "benchmark";
    else if (ModeReqIsSet(MODEREQ_TEST_MASK))
      return "testing";
    else if (ModeReqIsSet(MODEREQ_FLUSH))
      return "flushing";
    else if (ModeReqIsSet(MODEREQ_FETCH))
      return "fetching";
    else*/
      return "event";
  }
  else if (CheckPauseRequestTrigger())
    return "paused";
  else
    return "running";
}

bool HandleRequest_Version(MiniHttpRequest *request, MiniHttpResponse *response)
{
  char buffer[1024];

  sprintf( buffer,"<?xml version=\"1.0\" ?>\r"
                  "\r"
                  "<dcti>\r"
                  "<versioninfo>\r"
                  "<organization>Distributed Computing Technologies, Inc.</organization>\r"
                  "<client-name>distributed computing client</client-name>\r"
                  "<version>%s</version>\r"
                  "<activities-supported>\r"
                  "<activity id=\"rc5\">RC5-64</activity>\r" // always supported?
#ifdef HAVE_DES_CORES
                  "<activity id=\"des\">DES II</activity>\r"
#endif
#ifdef HAVE_CSC_CORES
                  "<activity id=\"csc\">CS Cipher</activity>\r"
#endif
#ifdef HAVE_OGR_CORES
                  "<activity id=\"ogr\">Optimal Golomb Rulers</activity>\r"
#endif
                  "</activities-supported>\r"
                  "<subprotocols-supported>\r"
                  "<subprotocol id=\"Status-1\"/>\r"
                  "<subprotocol id=\"Event-1\"/>\r"
//                  "<subprotocol id=\"Config-1\"/>\r"  // not yet
                  "</subprotocols-supported>\r"
                  "</versioninfo>\r"
                  "</dcti>",
                  CliGetFullVersionDescriptor() // alternatively use CLIENT_VERSIONSTRING

                  // %s request->RequestMethod          GET or POST
                  // %s request->RequestUri.GetHead()   complete file location (including ?etc)
                  // %s request->Headers.GetHead()      the entire header
                  // %s request->ContentBody.GetHead()  entire content only used by POST?
                  // %i request->ContentLength          contentlength  only used by POST?

                  // request->GetDocumentUri((AutoBuff)&value);                file location (without ?etc)
                  // request->GetFormVariable("something",(AutoBuff)&value);   single form inputs
                  // request->GetHeader("user-agEnt",(AutoBuff)&value);        single header info pieces
                        /*
                  request->RequestUri.GetHead(),
                  ((testserv)?("TEST:"):("")), msgptr */);


  response->ContentBody = buffer;
  response->SetHeader("Content-Type", "text/xml");
  return true;
}

bool HandleRequest_Event(MiniHttpRequest *request, MiniHttpResponse *response)
{
  char buffer[512];

  sprintf( buffer,"<?xml version=\"1.0\" ?>\r"
                  "\r"
                  "<dcti>\r"
                  "<eventlist>\r"
                  "<event name=\"restart\">Restart the client</event>\r"
                  "<event name=\"quit\">Quits the client</event>\r"
                  "<event name=\"pause\">Pauses the client</event>\r"
                  "<event name=\"unpause\">Unpauses the client</event>\r"
                  "\r"
                  "<event name=\"fetch\">Fetch Work Blocks</event>\r"
                  "<event name=\"flush\">Flush Completed Blocks</event>\r"
                  "<event name=\"update\">Update all buffer</event>\r"
                  "</eventlist>\r"
                  "</dcti>");

  response->ContentBody = buffer;
  response->SetHeader("Content-Type", "text/xml");
  return true;
}

int FillStruct(char *buffer, Problem *thisprob, int prob_i)
{
  char info [36];
  ContestWork work;
  static struct timeval tv_cpu = {0,0};
  static struct timeval tv_real = {0,0};

  // <cpu-time-elapsed>
    if (CliGetThreadUserTime( &tv_cpu )!=0)
    {
      tv_cpu.tv_sec=thisprob->runtime_sec;
      tv_cpu.tv_usec=thisprob->runtime_usec;
    }
  // </cpu-time-elapsed>

  // <real-time-elapsed>
    //tv_real.tv_usec = thisprob->completion_timelo; /* real clock time between start/finish */
    //tv_real.tv_sec = thisprob->completion_timehi; /* including suspended/paused time */
    // why does the above not work?!? it returns 0.00:00:00.00
/*    
    //thisprob->GetKeysDone(&tv_real);
    tv_real.tv_usec = thisprob->timehi;
    tv_real.tv_usec = thisprob->timelo;
    //CliTimerDiff( &tv_real, NULL,  &tv_real); //get time difference as tv

    // borrowed from clitime.cpp since calling CliGetTimeString 2 times in a rows triggers a bug!
    unsigned long longtime = (unsigned long)tv_cpu.tv_sec;
    static char hourstring[8+1 +2+1 +2+1+2 +1 +2];
    int days = (longtime / 86400UL);
    sprintf( hourstring,  "%d.%02d:%02d:%02d.%02d", (int) (longtime / 86400UL),
      (int) ((longtime % 86400L) / 3600UL), (int) ((longtime % 3600UL)/60),
      (int) (longtime % 60), (int) ((tv_cpu.tv_usec/10000L)%100) );
*/  // </real-time-elapsed>

  // <loaded-info>
    if (thisprob->RetrieveState(&work, NULL, 0) >= 0)
    {
      if (thisprob->contest == OGR)
        sprintf(info,ogr_stubstr(&work.ogr.workstub.stub));
      else
        sprintf(info,"%08lX:%08lX",(unsigned long)(work.crypto.key.hi),(unsigned long)(work.crypto.key.lo));
    }
    else
        sprintf(info,"error");
  // </loaded-info>


  sprintf(buffer,"<thread id=\"%i\">\r"
                 "<activity>%s</activity>\r"
                 "<percentage-complete>%i</percentage-complete>\r"
                 "<real-time-elapsed>%s</real-time-elapsed>\r"
                 "<cpu-time-elapsed>%s</cpu-time-elapsed>\r"
                 "<current-rate units=\"kkeys/sec\">---.--</current-rate>\r"
                 "<loaded-size units=\"work units\">--</loaded-size>\r"
                 "<loaded-info>%s</loaded-info>\r"
                 "</thread>\r",
                 prob_i,//thisprob->threadindex,
                 CliGetContestNameFromID( thisprob->contest ), // __getprojsectname(uint) as in confrwv.cpp
                 (thisprob->CalcPermille()/10),
                 /*CliGetTimeString( &tv_real, 2 ),hourstring,*/"-.--:--:--.--",
                 CliGetTimeString( &tv_cpu, 2 ),
                 info
                 );
   return 0;
}

bool HandleRequest_Status(MiniHttpRequest *request, MiniHttpResponse *response)
{
  char buffer[3052];

  unsigned int prob_i = 0;
  char buf2[256];
  char cusbuff[1024];
  Problem *thisprob;
  struct timeval tv;

  thisprob = GetProblemPointerFromIndex(prob_i); // is counting on at least one thread a safe thing?
  FillStruct(cusbuff,thisprob,prob_i);
  
  CliClock(&tv);
  
  prob_i++;

  while (((thisprob = GetProblemPointerFromIndex(prob_i)) != NULL) && prob_i <= 26)
  {
    FillStruct(buf2,thisprob,prob_i);
    strcat(cusbuff,buf2);
    prob_i++;
  }

  sprintf( buffer,"<?xml version=\"1.0\" ?>\r"
                  "\r"
                  "<dcti>\r"
                  "<status>%s</status>\r"
                  "<numprocessors>%i</numprocessors>\r"
                  "<time-elapsed>%s</time-elapsed>\r"
                  "\r"
                  "<threadlist>\r"
                  "%s\r"
                  "</threadlist>\r"
                  "\r"
                  "<activitylist>\r"
                  "<activity id=\"csc\">\r"
                  "<contest-open> true </contest-open>\r"
                  "<multi-real-time-elapsed> 1.03:45:21.25 </multi-real-time-elapsed>\r"
                  "<average-rate units=\"kkeys/sec\"> 704.17 </average-rate>\r"
                  "<total-completed units=\"work units\"> 82 </total-completed>\r"
                  "<buffers type=\"in\">\r"
                  "<amount units=\"packets\"> 6 </amount>\r"
                  "<amount units=\"work units\"> 24 </amount>\r"
                  "<amount units=\"time\"> 0.02:12:00.00 </amount>\r"
                  "</buffers>\r"
                  "<buffers type=\"out\">\r"
                  "<amount units=\"packets\"> 7 </amount>\r"
                  "<amount units=\"work units\"> 28 </amount>\r"
                  "</buffers>\r"
                  "</activity>\r"
                  "</activitylist>\r"
                  "\r"
                  "</dcti>",
                  FillActivity(),
                  GetNumberOfDetectedProcessors(),
                  CliGetTimeString( &tv, 2 ),               // totaltime by thread #0
                  cusbuff

                  );

  response->ContentBody = buffer;
  response->SetHeader("Content-Type", "text/xml");
  return true;
}

// Called when the specified MiniHttpDaemonConnection has a fully formed
// request to be processed.  Returns false if the request was invalid,
// or true if the request was successfully executed.

bool ProcessClientPacket(MiniHttpDaemonConnection *client)
{
  MiniHttpRequest request;

  if (client->FetchRequest(&request))
  {
#if 0
    printf("headers follow :\n------\n%s\n------\n\n",
        request.Headers.GetHead() );
    printf("body follows (length=%d):\n------\n%s\n------\n\n",
        request.ContentLength, request.ContentBody.GetHead());

    AutoBuffer value;
    if (request.GetHeader("user-agEnt", &value))
      printf("User-Agent: \"%s\"\n", value.GetHead() );
    else
      printf("Could not find user-agent\n");

    MiniHttpResponse response;
    response.ContentBody = "<h1>Howdy</h1>";
    response.SetHeader("Content-Type", "text/html");
    client->QueueResponse(&response);
    return true;
#else
    bool haveResponse = false;
    MiniHttpResponse response;
    AutoBuffer DocumentUri;
    if (request.GetDocumentUri(&DocumentUri))
    {
      if (strcmp(request.RequestMethod, "GET") == 0)
      {
        if (strcmp(DocumentUri.GetHead(), "/version.xml") == 0)
        {
          haveResponse = HandleRequest_Version(&request, &response);
        }
        else if (strcmp(DocumentUri.GetHead(), "/Status-1/status.xml") == 0)
        {
          haveResponse = HandleRequest_Status(&request, &response);
        }
        else if (strcmp(DocumentUri.GetHead(), "/Event-1/eventlist.xml") == 0)
        {
          haveResponse = HandleRequest_Event(&request, &response);
        }
        else if (strcmp(DocumentUri.GetHead(), "/Event-1/set.xml") == 0)
        {

        }
        else if (strcmp(DocumentUri.GetHead(), "/config/optionlist.xml") == 0)
        {

        }
        else if (strcmp(DocumentUri.GetHead(), "/config/option.xml") == 0)
        {

        }
        else
        {
        MiniHttpResponse response;

        response.ContentBody = "<h1>What are you looking for?</h1>";
        response.SetHeader("Content-Type", "text/html");
        haveResponse = true;
        }
      }
      else if (strcmp(request.RequestMethod, "POST") == 0)
      {
        if (strcmp(DocumentUri.GetHead(), "/Event-1/set.xml") == 0)
        {
          haveResponse = HandleRequest_Status(&request, &response);
        }
        else if (strcmp(DocumentUri.GetHead(), "/config/option.xml") == 0)
        {

        }
      }
    }

    if (haveResponse) {
      if (client->QueueResponse(&response))
        return true;
    }
#endif
  }
  return false;     // error, close connection.
}


void __fdsetsockadd(SOCKET sock, fd_set *setname, int *sockmax, int *writecnt)
{
  if (*writecnt < FD_SETSIZE)
  {
    *writecnt++;
    if ((int) sock > *sockmax) *sockmax = (int) sock;
    FD_SET(sock, setname);
  }
}

