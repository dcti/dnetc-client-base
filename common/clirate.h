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
#define __CLIRATE_H__ "@(#)$Id: clirate.h,v 1.13.2.1 2000/09/21 18:07:37 cyp Exp $"

// return (cumulative) keyrate for a particular contest
double CliGetKeyrateForContest( int contestid );

// return keyrate for a single problem. Problem must be finished.
double CliGetKeyrateForProblem( Problem *problem );

#endif /* __CLIRATE_H__ */
