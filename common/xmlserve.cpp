/* Created by Michael Feiri (michael.feiri@mfeiri.lake.de)
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/

#include "client.h"
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
#include "clisrate.h"
#include "buffbase.h"

double gKeyRate;

/* as seen in confmenu.cpp */
char *__have_xxx_cores(unsigned int cont_i)
{
  switch (cont_i)
  {
    case RC5: return "RC5-64";
    #if defined(HAVE_DES_CORES)
    case DES: return "DES";
    #endif
    #if defined(HAVE_OGR_CORES)
    case OGR: return "Optimal Golomb Rulers";
    #endif
    #if defined(HAVE_CSC_CORES)
    case CSC: return "CS Cipher";
    #endif
  }
  return NULL;
}

static char *GetKeyrateAsString2( char *buffer, double rate, double limit, int round )
{
  if (rate<=((double)(0)))  // unfinished (-2) or error (-1) or impossible (0)
    strcpy( buffer, "---.-- " );
  else
  {
    unsigned int rateint, index = 0;
    const char *multiplier[]={"","k","M","G","T"}; // "", "kilo", "mega", "giga", "tera"
    while (index<=5 && (((double)(rate))>=((double)(limit))) )
    {
      index++;
      rate = ((double)(rate)) / ((double)(1000));
    }
    if (index > 4)
      strcpy( buffer, "***.** " ); //overflow (rate>1.0 TKeys. Str>25 chars)
    else
    {
      rateint = (unsigned int)(rate);
      if (round) {
        sprintf( buffer, "%u.%02u %s", rateint,
           ((unsigned int)((((double)(rate-((double)(rateint)))))*((double)(100)))),
           multiplier[index] );
      }
      else {
        sprintf( buffer, "%u %s", rateint, multiplier[index] );
      }
    }
  }
  return buffer;
}

static const char *num_sep(const char *number)
{
  static char num_string[(sizeof(long)*3*2) + 1];

  char *rp, *wp, *xp;             /* read ptr, write ptr, aux ptr */
  register unsigned int digits;
  register unsigned int i, j;

  /*
   * just in case somebody passes a pointer to a long string which
   * then obviously holds much more than just a number. In this case
   * we simply return what we got and don't do anything.
   */

  digits = strlen(number);

  if (digits >= (sizeof(num_string)-1))
    return number;

  strcpy(num_string, number);
  rp = num_string + digits - 1;
  wp = num_string + (sizeof(num_string)-1);
  *wp-- = '\0';
  if ((xp = strchr(num_string, '.')) != NULL) {
    while (rp >= xp) {
      *wp-- = *rp--;
      digits--;
    }
  }
  for (i = digits, j = 1; i; i--, j++) {
    *wp-- = *rp--;
    if (j && ((j % 3) == 0))
      *wp-- = ',';
  }
  if (wp[1] == ',')
    wp++;
  return (wp + 1);
}


const char *FillStatus(void)
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

int FillActivityID(char * buffer)
{
   int contestnum = 0;
   char *contestname;
   char helper[128];
   
   while (CliIsContestIDValid( contestnum ))
   {
      if ((contestname = __have_xxx_cores(contestnum)) != NULL)
      {
         sprintf(helper,"<activity id=\"%s\">%s</activity>\r", CliGetContestNameFromID(contestnum),contestname);
         strcat(buffer,helper);
      }
      contestnum++;
   }
   return 0;
}

int FillThreadList(char *buffer, Problem *thisprob, int prob_i)
{
  char info [36];
  ContestWork work;
  static struct timeval tv_cpu = {0,0};
  static struct timeval tv_real = {0,0};
  char keyrate[64];
  const char *keyrateP = "---.-- ";
    
  // <cpu-time-elapsed>
    if (CliGetThreadUserTime( &tv_cpu )!=0)
    {
      tv_cpu.tv_sec=thisprob->runtime_sec;
      tv_cpu.tv_usec=thisprob->runtime_usec;
    }
  // </cpu-time-elapsed>

  // <real-time-elapsed>
    tv_real.tv_usec = thisprob->completion_timelo; /* real clock time between start/finish */
    tv_real.tv_sec = thisprob->completion_timehi; /* including suspended/paused time */
    // why does the above not work?!? it returns 0.00:00:00.00
/*    
    //thisprob->GetKeysDone(&tv_real);
    tv_real.tv_usec = thisprob->timehi;
    tv_real.tv_usec = thisprob->timelo;
    //CliTimerDiff( &tv_real, NULL,  &tv_real); //get time difference as tv
*/
    // borrowed from clitime.cpp since calling CliGetTimeString 2 times in a rows triggers a bug!
    unsigned long longtime = (unsigned long)tv_real.tv_sec;
    static char hourstring[8+1 +2+1 +2+1+2 +1 +2];
    int days = (longtime / 86400UL);
    sprintf( hourstring,  "%d.%02d:%02d:%02d.%02d", (int) (longtime / 86400UL),
      (int) ((longtime % 86400L) / 3600UL), (int) ((longtime % 3600UL)/60),
      (int) (longtime % 60), (int) ((tv_real.tv_usec/10000L)%100) );
  // </real-time-elapsed>

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

 //   keyrateP=GetKeyrateAsString2(keyrate,CliGetKeyrateForProblemNoSave(thisprob),((double)(1000)), 1);
   gKeyRate += CliGetKeyrateForProblemNoSave(thisprob);

  sprintf(buffer,"<thread id=\"%i\">\r"
                 "<activity>%s</activity>\r"
                 "<percentage-complete>%i</percentage-complete>\r"
                 "<real-time-elapsed>%s</real-time-elapsed>\r"
                 "<cpu-time-elapsed>%s</cpu-time-elapsed>\r"
                 "<current-rate units=\"keys/sec\">%.0f</current-rate>\r"
                 "<loaded-size units=\"work units\">%d</loaded-size>\r"
                 "<loaded-info>%s</loaded-info>\r"
                 "</thread>\r",
                 prob_i,//thisprob->threadindex,
                 CliGetContestNameFromID( thisprob->contest ), // __getprojsectname(uint) as in confrwv.cpp
                 (thisprob->CalcPermille()/10),
                 /*CliGetTimeString( &tv_real, 2 ),*/hourstring,
                 CliGetTimeString( &tv_cpu, 2 ),
                 CliGetKeyrateForProblemNoSave(thisprob),//keyrateP,
                 (thisprob->contest==OGR) ? 1 : (__iter2norm(work.crypto.iterations.lo,work.crypto.iterations.hi)),
                 info
                 );
   return 0;
}

int FillActivityList(char *buffer,Client *client)
{

  unsigned int contestnum = 0;
   char helper[2048], keyrate[32];
  unsigned int packets, units;
  struct timeval ttime,tv;
  double totaliter;
  const char *keyrateP;
  unsigned long units_in, units_out;
  long buff_in, buff_out;
  
   while (CliIsContestIDValid( contestnum ))
   {
      if (__have_xxx_cores(contestnum) != NULL)
      {
      
      // as seen in CliGetSummaryStringForContest(int)
      CliGetContestInfoSummaryData( contestnum, &packets, &totaliter, &ttime, &units );

      buff_out = GetBufferCount( client, contestnum, 1, &units_out );
      buff_in = GetBufferCount( client, contestnum, 0, &units_in );
      
      //keyrateP=GetKeyrateAsString2(keyrate,CliGetKeyrateForContest(contestnum),((double)(1000)), 1);

      // extended function in client.cpp!!!
      extern int ClientGetNumberOfProcessorsInUse(void);
      tv.tv_usec = 0;
      tv.tv_sec = units_in * (CliGetContestWorkUnitSpeed( contestnum, 0 )) / (ClientGetNumberOfProcessorsInUse());


      sprintf(helper,"<activity id=\"%s\">\r"
                     "<contest-open>----</contest-open>\r"
                     "<multi-real-time-elapsed>%s</multi-real-time-elapsed>\r"
                     "<average-rate units=\"keys/sec\">%.0f</average-rate>\r"
                     "<total-completed units=\"work units\">%i</total-completed>\r"
                     "<buffers type=\"in\">\r"
                     "<amount units=\"packets\">%d</amount>\r"
                     "<amount units=\"work units\">%d</amount>\r"
                     "<amount units=\"time\">%s</amount>\r"
                     "</buffers>\r"
                     "<buffers type=\"out\">\r"
                     "<amount units=\"packets\">%d</amount>\r"
                     "<amount units=\"work units\">%d</amount>\r"
                     "</buffers>\r"
                     "</activity>\r",
                     CliGetContestNameFromID(contestnum),
                     CliGetTimeString( &ttime, 2 ),
                     gKeyRate,//CliGetKeyrateForContest(contestnum),//keyrateP,
                     units,
                     buff_in,
                     units_in,
                     CliGetTimeString( &tv, 2),
                     buff_out,
                     units_out
                     );

      if (contestnum==0) // can I depend on RC5 being always there/first?
      strcpy(buffer,helper);
      else
      strcat(buffer,helper);
      
      }
      contestnum++;
   }
   return 0;
   
   
   
}

bool HandleRequest_Version(MiniHttpRequest *request, MiniHttpResponse *response)
{
  char buffer[1024];
  char contestsupport[1024];

  FillActivityID(contestsupport);

  sprintf( buffer,"<?xml version=\"1.0\" ?>\r"
                  "\r"
                  "<dcti>\r"
                  "<versioninfo>\r"
                  "<organization>Distributed Computing Technologies, Inc.</organization>\r"
                  "<client-name>distributed computing client</client-name>\r"
                  "<version>%s</version>\r"
                  "<activities-supported>\r"
                  "%s\r"
                  "</activities-supported>\r"
                  "<subprotocols-supported>\r"
                  "<subprotocol id=\"Status-1\"/>\r"
                  "<subprotocol id=\"Event-1\"/>\r"
//                  "<subprotocol id=\"Config-1\"/>\r"  // not yet
                  "</subprotocols-supported>\r"
                  "</versioninfo>\r"
                  "</dcti>",
                  CliGetFullVersionDescriptor(), // alternatively use CLIENT_VERSIONSTRING
                  contestsupport
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

bool HandleRequest_Status(MiniHttpRequest *request, MiniHttpResponse *response, Client *client)
{
  char buffer[3052];

  unsigned int prob_i = 0;
  char buf2[256];
  char threadlist[2048];
  char activitylist[2048];
  Problem *thisprob;
  struct timeval tv;

  gKeyRate=0;

  thisprob = GetProblemPointerFromIndex(prob_i); // is counting on at least one thread a safe thing?
  FillThreadList(threadlist,thisprob,prob_i);
  
  CliClock(&tv);
  
  prob_i++;

  while (((thisprob = GetProblemPointerFromIndex(prob_i)) != NULL) && prob_i <= 26)
  {
    FillThreadList(buf2,thisprob,prob_i);
    strcat(threadlist,buf2);
    prob_i++;
  }

  FillActivityList(activitylist,client);

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
                  "%s\r"
                  "</activitylist>\r"
                  "\r"
                  "</dcti>",
                  FillStatus(),
                  GetNumberOfDetectedProcessors(),
                  CliGetTimeString( &tv, 2 ),               // totaltime by thread #0
                  threadlist,
                  activitylist
                  );

  response->ContentBody = buffer;
  response->SetHeader("Content-Type", "text/xml");
  return true;
}

// Called when the specified MiniHttpDaemonConnection has a fully formed
// request to be processed.  Returns false if the request was invalid,
// or true if the request was successfully executed.

bool ProcessClientPacket(MiniHttpDaemonConnection *httpcon, Client *client)
{
  MiniHttpRequest request;

  if (httpcon->FetchRequest(&request))
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
    httpcon->QueueResponse(&response);
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
          haveResponse = HandleRequest_Status(&request, &response, client);
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
          haveResponse = HandleRequest_Status(&request, &response, client);
        }
        else if (strcmp(DocumentUri.GetHead(), "/config/option.xml") == 0)
        {

        }
      }
    }

    if (haveResponse) {
      if (httpcon->QueueResponse(&response))
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

