/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for calculating the keyrate for a completed
 * problem and for obtaining the total/average keyrate for an entire contest.
 * ----------------------------------------------------------------------
*/
const char *clirate_cpp(void) {
return "@(#)$Id: clirate.cpp,v 1.22 1999/04/09 13:31:59 cyp Exp $"; }

#include "cputypes.h" //for u64 define
#include "problem.h"  //uses Problem::RetrieveState()
#include "baseincs.h" //timeval
#include "clicdata.h" //Cli[Add|Get]ContestInfoSummaryData, CliGetContestInfoBaseData
#include "clitime.h"  //CliTimerDiff
#include "clirate.h"  //keep the prototypes in sync.

/* -------------------------------------------------------------------- */

// return (cumulative) keyrate for a particular contest
double CliGetKeyrateForContest( int contestid )
{
  struct timeval totaltime;
  double totaliter;

  if (CliGetContestInfoSummaryData( contestid, NULL, &totaliter, &totaltime))
    return ((double)(0));  //clicdata.cpp says no such contest
  if (!totaltime.tv_sec && !totaltime.tv_usec)
    return ((double)(0));

  return ((double)(totaliter)) /
      (((double)(totaltime.tv_sec))+
     (((double)(totaltime.tv_usec))/((double)(1000000L))));
}

/* ---------------------------------------------------------------------- */

//internal - see CliGetKeyrateForProblem() for description
static double __CliGetKeyrateForProblem( Problem *prob, int doSave )
{
  unsigned int count, contestid;
  struct timeval tv;
  ContestWork work;
  int resultcode;
  double keys;

  if (!prob)
    return ((double)(-1));

  resultcode = prob->RetrieveState( &work, &contestid, 0 ); 
  if (resultcode < 0)
    return ((double)(-2));   // not initialized or core error
  if (resultcode != RESULT_NOTHING && resultcode != RESULT_FOUND)
    return ((double)(-2));   // not finished
    
  /*
  tv.tv_usec = prob->timelo;
  tv.tv_sec = prob->timehi;
  CliTimerDiff( &tv, &tv, NULL ); //get time difference as tv
  */

  tv.tv_usec = prob->runtime_usec; /* actual core run time */
  tv.tv_sec = prob->runtime_sec;
  
  if (CliGetContestInfoBaseData( contestid, NULL, &count )) //clicdata.cpp
    return ((double)(0));   //clicdata.cpp says no such contest

  keys = U64TODOUBLE(work.crypto.keysdone.hi,work.crypto.keysdone.lo);
  if (count>1) //iteration-to-keycount-multiplication-factor
    keys = (keys)*((double)(count));
  if (prob->startpermille) //0-1000
    keys = (keys)*(((double)(1000L-(prob->startpermille)))/((double)(1000L)));
  if (keys==((double)(0))) //no keys done? should never happen.
    return ((double)(0));

  if (!tv.tv_sec && !tv.tv_usec)
    tv.tv_usec = 1; //don't divide by zero
    
  if (doSave)
  {
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

