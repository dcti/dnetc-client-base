/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * ----------------------------------------------------------------------
 * This file contains functions for calculating the keyrate for a completed
 * problem and for obtaining the total/average keyrate for an entire contest.
 * ----------------------------------------------------------------------
*/
const char *clirate_cpp(void) {
return "@(#)$Id: clirate.cpp,v 1.24.2.6 2000/09/24 13:36:31 andreasb Exp $"; }

#include "baseincs.h" //timeval
#include "client.h"   //for project constants
#include "problem.h"  //uses Problem::RetrieveState()
#include "clicdata.h" //Cli[Get|Add]ContestInfo[Summary|Base]Data
#include "clitime.h"  //CliTimerDiff
#include "clirate.h"  //keep the prototypes in sync.

/* -------------------------------------------------------------------- */

// return (cumulative) keyrate for a particular contest
double CliGetKeyrateForContest( int contestid )
{
  struct timeval totaltime;
  double totaliter;

  if (CliGetContestInfoSummaryData( contestid, NULL, &totaliter, &totaltime, NULL ))
    return ((double)(0));  //clicdata.cpp says no such contest
  if (!totaltime.tv_sec && !totaltime.tv_usec)
    return ((double)(0));

  return ((double)(totaliter)) /
      (((double)(totaltime.tv_sec))+
     (((double)(totaltime.tv_usec))/((double)(1000000L))));
}

/* ---------------------------------------------------------------------- */

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_ ((double)(0xFFFFFFFFul))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
          (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

// return keyrate for a single problem. Problem must be finished.
double CliGetKeyrateForProblem( Problem *prob /*, int doSave */ )
{
  unsigned int count, contestid, units;
  struct timeval tv;
  ContestWork work;
  int resultcode;
  double keys, rate;

  if (!prob)
    return ((double)(-1));

  resultcode = prob->RetrieveState( &work, &contestid, 0 ); 
  if (resultcode != RESULT_NOTHING && resultcode != RESULT_FOUND)
    return ((double)(-2)); // not initialized or core error or not finished

  if (CliGetContestInfoBaseData( contestid, NULL, &count )) //clicdata.cpp
    return ((double)(0));   //clicdata.cpp says no such contest
    
  /*
  tv.tv_usec = prob->timelo;
  tv.tv_sec = prob->timehi;
  CliTimerDiff( &tv, &tv, NULL ); //get time difference as tv
  */

  //tv.tv_usec = prob->runtime_usec; /* actual core crunch time */
  //tv.tv_sec = prob->runtime_sec;
  //tv.tv_usec = prob->elapsed_time_usec; /* real clock time between start/finish */
  //tv.tv_sec  = prob->elapsed_time_sec;  /* including suspended/paused time */
  prob->GetElapsedTime(&tv);              // wall clock time

  if (!tv.tv_sec && !tv.tv_usec)
    tv.tv_usec = 1; //don't divide by zero

  switch (contestid) {
    case RC5:
    case DES:
    case CSC:
      units = (((work.crypto.iterations.lo) >> 28) +
               ((work.crypto.iterations.hi) <<  4) );
      keys = U64TODOUBLE(work.crypto.keysdone.hi,work.crypto.keysdone.lo);
      break;
    case OGR:
      units = 1;
      keys = U64TODOUBLE(work.ogr.nodes.hi,work.ogr.nodes.lo);
      break;
  }

  if (prob->startkeys.hi || prob->startkeys.lo) 
    keys = (keys)-U64TODOUBLE(prob->startkeys.hi,prob->startkeys.lo);
  if (count>1) //iteration-to-keycount-multiplication-factor
    keys = (keys)*((double)(count));
  if (keys==((double)(0))) //no keys done? should never happen.
    return ((double)(0));

  /* if (doSave) */
  {
    count = 1; //number of blocks to add to clicdata.cpp information
    CliAddContestInfoSummaryData( contestid, &count, &keys, &tv, &units );
  }

  rate = ((double)(keys))/
       (((double)(tv.tv_sec))+(((double)(tv.tv_usec))/((double)(1000000L))));
       
  if (contestid != OGR)       
    CliSetContestWorkUnitSpeed(contestid, (unsigned int)((1<<28)/rate + 0.5));

  return rate;
}
/* ---------------------------------------------------------------------- */

