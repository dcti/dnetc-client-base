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
#include <Sound.h>
#include "client.h"
#include "clicdata.h"
#include "clitime.h"

bool HandleRequest_Version(MiniHttpRequest *request, MiniHttpResponse *response)
{
  char buffer[1024];

  sprintf( buffer,"<?xml version=\"1.0\"?>\r"
                  "\r"
                  "<dcti>\r"
                  "  <versioninfo>\r"
                  "    <organization>Distributed Computing Technologies, Inc.</organization>\r"
                  "    <client-name>distributed computing client</client-name>\r"
                  "    <version>%s</version>\r"
                  "    <activities-supported>\r"
                  "      <activity id=\"rc5\">RC5-64</activity>\r" // always supported?
#ifdef HAVE_DES_CORES
                  "      <activity id=\"des\">DES II</activity>\r"
#endif
#ifdef HAVE_CSC_CORES
                  "      <activity id=\"csc\">CS Cipher</activity>\r"
#endif
#ifdef HAVE_OGR_CORES
                  "      <activity id=\"ogr\">Optimal Golomb Rulers</activity>\r"
#endif
                  "    </activities-supported>\r"
                  "    <subprotocols-supported>\r"
                  "      <subprotocol id=\"Status-1\"/>\r"
                  "      <subprotocol id=\"Event-1\"/>\r"
//                  "      <subprotocol id=\"Config-1\"/>\r"  // not yet
                  "    </subprotocols-supported>\r"
                  "  </versioninfo>\r"
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

int FillStruct(char *buffer, Problem *thisprob, int prob_i)
{
   sprintf(buffer,"    <thread id=\"%i\">\r"
                  "      <activity>%s</activity>\r"
                  "      <percentage-complete>%i</percentage-complete>\r"
                  "      <real-time-elapsed> 0.00:13:11.27 </real-time-elapsed>\r"
                  "      <cpu-time-elapsed> 0.00:13:11.27 </cpu-time-elapsed>\r"
                  "      <current-rate units=\"kkeys/sec\"> 771.77 </current-rate>\r"
                  "      <loaded-size units=\"work units\"> 16 </loaded-size>\r"
                  "      <loaded-info>00B28625:8000000</loaded-info> \r"
                  "    </thread>\r",
                  prob_i,//thisprob->threadindex,
                  CliGetContestNameFromID( thisprob->contest ), // __getprojsectname( unsigned int ci ) as in confrwv.cpp
                  (thisprob->CalcPermille()/10));
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
  tv.tv_sec = thisprob->completion_timehi;  //wall clock time
  tv.tv_usec = thisprob->completion_timelo;
  printf("\ntime:%d,%d,%s\n", tv.tv_sec, tv.tv_usec, CliGetTimeString( &tv, 2 ));
  prob_i++;

  while (((thisprob = GetProblemPointerFromIndex(prob_i)) != NULL) && prob_i <= 26)
  {
   FillStruct(buf2,thisprob,prob_i);
   strcat(cusbuff,buf2);
   prob_i++;
  }

  sprintf( buffer,"<?xml version=\"1.0\"?>\r"
                  "\r"
                  "<dcti>\r"
                  "  <status>%s</status>\r"
                  "  <numprocessors>%i</numprocessors>\r"
                  "  <time-elapsed>%s</time-elapsed>\r"
                  "\r"
                  "  <threadlist>\r"
                  "%s\r"
                  "  </threadlist>\r"
                  "\r"
                  "  <activitylist>\r"
                  "    <activity id=\"csc\">\r"
                  "      <contest-open> true </contest-open>\r"
                  "      <multi-real-time-elapsed> 1.03:45:21.25 </multi-real-time-elapsed>\r"
                  "      <average-rate units=\"kkeys/sec\"> 704.17 </average-rate>\r"
                  "      <total-completed units=\"work units\"> 82 </total-completed>\r"
                  "      <buffers type=\"in\">\r"
                  "        <amount units=\"packets\"> 6 </amount>\r"
                  "        <amount units=\"work units\"> 24 </amount>\r"
                  "        <amount units=\"time\"> 0.02:12:00.00 </amount>\r"
                  "      </buffers>\r"
                  "      <buffers type=\"out\">\r"
                  "        <amount units=\"packets\"> 7 </amount>\r"
                  "        <amount units=\"work units\"> 28 </amount>\r"
                  "      </buffers>\r"
                  "    </activity>\r"
                  "  </activitylist>\r"
                  "\r"
                  "</dcti>",
                  (ModeReqIsSet(-1)?("event"):("running")), // useless
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

