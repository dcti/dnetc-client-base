// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for calculating the keyrate for a completed
// problem and for obtaining the total/average keyrate for an entire contest.

/* module history:
   01 May 1998 - created - Cyrus Patel <cyp@fb14.uni-mainz.de>

   revisions:
   23 May 1998 - corrected CliGetKeyrateForProblem so that blocks
                 with the same key but different contest ids
                 are recognized as unique problems.
*/   

#include "clirate.h" //includes client.h, clicdata.h, clitime.h

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
     (((double)(totaltime.tv_usec))/((double)(1000000))));
}

// ---------------------------------------------------------------------------


// return keyrate for a single problem. Problem must be finished.
double CliGetKeyrateForProblem( Problem *prob )
{
  static struct { u64 key; char contest; } addedqueue[MAXCPUS*2];
  static int addedqpos = -1;

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

  keys= U64TODOUBLE(ntohl(rc5result.keysdone.hi),ntohl(rc5result.keysdone.lo));
  if (count>1) //iteration-to-keycount-multiplication-factor
    keys = (keys)*((double)(count));
  if (prob->startpercent) //slight misnomer. factor is *100000 not *100
    keys = (keys)*(((double)(100000-(prob->startpercent)))/((double)(100000)));
  if (keys==((double)(0))) //no keys done? should never happen.
    return ((double)(0));

  additive = 1;
  for (int i=0;i<(MAXCPUS*2);i++)
    {
    if (addedqpos==-1)
      { addedqueue[i].key.lo = addedqueue[i].key.hi = 0;
        addedqueue[i].contest = -1;
        if (i==((MAXCPUS*2)-1)) addedqpos = 0;
      }
    else if (addedqueue[i].key.hi==rc5result.key.hi && 
        addedqueue[i].key.lo==rc5result.key.lo && 
        addedqueue[i].contest == contestid )
      {
      additive=0;
      break;
      }
    }

  if (additive)
    {
    addedqueue[addedqpos].key.hi=rc5result.key.hi; 
    addedqueue[addedqpos].key.lo=rc5result.key.lo;
    addedqueue[addedqpos].contest=contestid;
    if ((++addedqpos)>=(MAXCPUS*2))
      addedqpos=0;

    count = 1; //number of blocks to add to clicdata.cpp information
    CliAddContestInfoSummaryData( contestid, &count, &keys, &tv );
    }
    
  return ((double)(keys))/
       (((double)(tv.tv_sec))+(((double)(tv.tv_usec))/((double)(1000000))));
}

// ---------------------------------------------------------------------------

