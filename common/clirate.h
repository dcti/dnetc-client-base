/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for calculating the keyrate for a completed
 * problem and for obtaining the total/average keyrate for an entire contest.
 * ----------------------------------------------------------------------
*/ 
#ifndef __CLIRATE_H__
#define __CLIRATE_H__ "@(#)$Id: clirate.h,v 1.13 1999/04/06 11:55:43 cyp Exp $"

//#include "problem.h" //uses Problem and RC5Result class definitions 

// return (cumulative) keyrate for a particular contest
double CliGetKeyrateForContest( int contestid );

// return keyrate for a single problem. Problem must be finished.
double CliGetKeyrateForProblem( Problem *problem );

//same as CliGetKeyrateForProblem() but doesn't add stats to contest totals
double CliGetKeyrateForProblemNoSave( Problem *problem );

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_ ((double)(0xFFFFFFFFul))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
          (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

#endif /* __CLIRATE_H__ */
