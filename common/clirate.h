// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for calculating the keyrate for a completed
// problem and for obtaining the total/average keyrate for an entire contest.
// 
// $Log: clirate.h,v $
// Revision 1.6  1998/06/24 19:25:53  cyruspatel
// Created function CliGetKeyrateForProblemNoSave(). Same as
// CliGetKeyrateForProblem() but does not affect cumulative stats.
//
// Revision 1.5  1998/06/14 08:12:39  friedbait
// 'Log' keywords added to maintain automatic change history
// 
// Revision 1.4  1998/06/08 18:11:31  remi
// Added a short line to tell [X]Emacs to use C++ mode instead 
// of C mode for these files.
//
// Revision 1.3  1998/05/29 08:01:06  bovine
// Copyright update, indents
//
// Revision 1.2  1998/05/25 02:54:17  bovine
// fixed indents
// 
// Revision 1.0  1998/05/24 14:25:49  daa
// Import 5/23/98 client tree
//
// Revision 0.0  1998/05/01 05:01:08  cyruspatel
// Created
//
// =====================================================================

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
double CliGetKeyrateForProblem( Problem *problem );

//same as CliGetKeyrateForProblem() but doesn't add stats to contest totals
double CliGetKeyrateForProblemNoSave( Problem *problem );

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_ ((double)(0xFFFFFFFF))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
          (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

#endif //ifdef _CLIRATE_H_

