// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for calculating the keyrate for a completed
// problem and for obtaining the total/average keyrate for an entire contest.

/* module history:
   01 May 1998 - created - Cyrus Patel <cyp@fb14.uni-mainz.de>
*/   

#ifndef _CLIRATE_H_
#define _CLIRATE_H_

#include "client.h" //uses Problem and RC5Result class definitions 
#include "clicdata.h" //Cli[Add|Get]ContestInfoSummaryData, CliGetContestInfoBaseData
#include "clitime.h" //CliTimerDiff

// return (cumulative) keyrate for a particular contest
double CliGetKeyrateForContest( int contestid );

// return keyrate for a single problem. Problem must be finished.
// Sets bit 0x80 in problem->finished to prevent repeated additions to total
double CliGetKeyrateForProblem( Problem *problem );

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_ ((double)(0xFFFFFFFF))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
          (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

#endif //ifdef _CLIRATE_H_

