/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for formatting keyrate/time/summary data
 * statistics obtained from clirate.cpp into strings suitable for display.
 * ----------------------------------------------------------------------
*/ 
#ifndef __CLISRATE_H__
#define __CLISRATE_H__ "@(#)$Id: clisrate.h,v 1.20.2.2 1999/12/12 15:34:04 cyp Exp $"

//#include "cputypes.h" // struct fake_u64
//#include "problem.h"  // Problem class

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_  ((double)(0xFFFFFFFFul))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
           (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

/* returns keyrate as string (maxlen=26) "nnnn.nn ['k'|'M'|'G'|'T']" */
/* return value is a pointer to buffer.                           */
const char *CliGetKeyrateAsString( char *buffer, double rate );

/* return iter/keysdone/whatever as string.                       */
/* set contestID = -1 to have the ID ignored                      */
const char *CliGetU64AsString( struct fake_u64 *u, int /*inNetOrder*/, int contestID );

/* combines CliGetKeyrateForProblem() and CliGetKeyrateAsString() */
const char *CliGetKeyrateStringForProblem( Problem *prob );

/* combines CliGetKeyrateForContest() and CliGetKeyrateAsString() */
const char *CliGetKeyrateStringForContest( int contestid );

/* "4 RC5 Blocks 12:34:56.78 - [123456789 keys/s]"               */
const char *CliGetSummaryStringForContest( int contestid );

/* Completed RC5 block 68E0D85A:A0000000 (123456789 keys)         */
/*           123:45:67:89 - [987654321 keys/s]                    */
const char *CliGetMessageForProblemCompleted( Problem *problem );

/* same as above, but does not affect cumulative stats            */
const char *CliGetMessageForProblemCompletedNoSave( Problem *problem );

#endif /* __CLISRATE_H__ */
