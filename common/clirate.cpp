// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for calculating the keyrate for a completed
// problem and for obtaining the total/average keyrate for an entire contest.
//
// $Log: clirate.cpp,v $
// Revision 1.15  1998/10/04 11:35:25  remi
// Id tags fun.
//
// Revision 1.14  1998/07/07 21:55:22  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.13  1998/06/29 08:44:02  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.12  1998/06/29 06:57:45  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.11  1998/06/24 19:25:51  cyruspatel
// Created function CliGetKeyrateForProblemNoSave(). Same as
// CliGetKeyrateForProblem() but does not affect cumulative stats.
//
// Revision 1.10  1998/06/15 12:03:52  kbracey
// Lots of consts.
//
// Revision 1.9  1998/06/14 08:26:41  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.8  1998/06/14 08:12:38  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.7  1998/06/09 09:03:07  jlawson
// Cast warning with Borland removed.
//
// Revision 1.6  1998/06/09 08:54:26  jlawson
// changes from Cyrus Patel.  
// 
// Revision 1.5  1998/06/08 15:47:05  kbracey
// added lots of "const"s and "static"s to reduce compiler warnings, and
// hopefully improve output code, too.
// 
// Revision 1.4  1998/05/29 08:01:04  bovine
// copyright update, indents
// 
// Revision 1.3  1998/05/28 14:09:37  daa
// fix for 2 contest benchmarking
//
// Revision 1.2  1998/05/25 02:54:16  bovine
// fixed indents
// 
// Revision 1.1  1998/05/24 14:25:49  daa
// Import 5/23/98 client tree
//
// Revision 0.1  1998/05/23 23:05:08  cyruspatel
// Corrected CliGetKeyrateForProblem so that blocks with the same block ID
// but different contest IDs are recognized as being different problems.
//
// Revision 0.0  1998/05/01 05:01:08  cyruspatel
// Created

// ======================================================================


#if (!defined(lint) && defined(__showids__))
const char *clirate_cpp(void) {
return "@(#)$Id: clirate.cpp,v 1.15 1998/10/04 11:35:25 remi Exp $"; }
#endif

#include "cputypes.h" //for u64 define
#include "problem.h"  //uses Problem and RC5Result class definitions 
#include "clicdata.h" //Cli[Add|Get]ContestInfoSummaryData, CliGetContestInfoBaseData
#include "clitime.h"  //CliTimerDiff
#include "clirate.h"  //keep the prototypes in sync.
#include "network.h"  // for ntohl and timeval

// ---------------------------------------------------------------------------

#define LASTDONE_LIST_SIZE (16) //the number of block id's we cache 
                                //to see if we've already added them

// ---------------------------------------------------------------------------

// return (cumulative) keyrate for a particular contest
double CliGetKeyrateForContest( int contestid )
{
  struct timeval totaltime;
  double totaliter;

  if (CliGetContestInfoSummaryData( contestid, NULL, &totaliter, &totaltime))
    return ((double)(0));  //clicdata.cpp says no such contest
  if (!totaltime.tv_sec && !totaltime.tv_usec)
    return ((double)(0));

  return ((double)(totaliter))/
      (((double)(totaltime.tv_sec))+
     (((double)(totaltime.tv_usec))/((double)(1000000L))));
}

// ---------------------------------------------------------------------------

//internal - see CliGetKeyrateForProblem() for description
static double __CliGetKeyrateForProblem( Problem *prob, int doSave )
{
  static struct { u64 key; signed char contest; } last_done_list[LASTDONE_LIST_SIZE];
  static int last_done_pos = -1;

  RC5Result rc5result;
  unsigned int count;
  struct timeval tv;
  int contestid, additive;
  double keys;

  if (!prob)
    return ((double)(-1));
  if ((!(prob->finished)) || (!(prob->started)) || (!(prob->IsInitialized())))
    return ((double)(-2));

  tv.tv_usec = prob->timelo;
  tv.tv_sec = prob->timehi;
  CliTimerDiff( &tv, &tv, NULL ); //get time difference as tv
  if (!tv.tv_sec && !tv.tv_usec)
    tv.tv_usec = 1; //don't divide by zero

  contestid = prob->GetResult( &rc5result );
  if (CliGetContestInfoBaseData( contestid, NULL, &count )) //clicdata.cpp
    return ((double)(0));   //clicdata.cpp says no such contest

  keys = U64TODOUBLE(ntohl(rc5result.keysdone.hi),ntohl(rc5result.keysdone.lo));
  if (count>1) //iteration-to-keycount-multiplication-factor
    keys = (keys)*((double)(count));
  if (prob->startpercent) //slight misnomer. factor is *100000 not *100
    keys = (keys)*(((double)(100000L-(prob->startpercent)))/((double)(100000L)));
  if (keys==((double)(0))) //no keys done? should never happen.
    return ((double)(0));

  additive = 1;
  for (int i = 0; i < (LASTDONE_LIST_SIZE); i++)
  {
    if (last_done_pos==-1)
    {
      last_done_list[i].key.lo = last_done_list[i].key.hi = 0;
      last_done_list[i].contest = -1;
      if (i == ((LASTDONE_LIST_SIZE)-1)) last_done_pos = 0;
    }
    else if (last_done_list[i].key.hi == rc5result.key.hi &&
        last_done_list[i].key.lo == rc5result.key.lo &&
        last_done_list[i].contest == contestid )
    {
      additive=0;
      break;
    }
  }

  if (additive && doSave)
  {
    last_done_list[last_done_pos].key.hi = rc5result.key.hi;
    last_done_list[last_done_pos].key.lo = rc5result.key.lo;
    last_done_list[last_done_pos].contest = (u8) contestid;
    if ((++last_done_pos) >= (LASTDONE_LIST_SIZE))
      last_done_pos=0;

    count = 1; //number of blocks to add to clicdata.cpp information
    CliAddContestInfoSummaryData( contestid, &count, &keys, &tv );
  }

  return ((double)(keys))/
       (((double)(tv.tv_sec))+(((double)(tv.tv_usec))/((double)(1000000L))));
}

// return keyrate for a single problem. Problem must be finished.
double CliGetKeyrateForProblem( Problem *prob )
{  return __CliGetKeyrateForProblem( prob, 1 ); }

double CliGetKeyrateForProblemNoSave( Problem *prob )
{  return __CliGetKeyrateForProblem( prob, 0 ); }

// ---------------------------------------------------------------------------

