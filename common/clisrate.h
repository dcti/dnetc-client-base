// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for formatting keyrate/time/summary data
// statistics obtained from clirate.cpp into strings suitable for display.

/* module history:
   01 May 1998 - created - Cyrus Patel <cyp@fb14.uni-mainz.de>
*/   

#ifndef _CLICSTAT_H_
#define _CLICSTAT_H_

#include "client.h" //timeval, sprintf(), u64, Problem/Fileentry/RC5Result
#include "clitime.h" // for CliTimer(), CliTimerDiff(), CliGetTimeString()
#include "clirate.h"  // for CliGetKeyrateFor[Problem|Contest]()
#include "clicdata.h" // for CliGetContestInfo[Base|Summary]Data()

#ifndef _U32LimitDouble_
  #define _U32LimitDouble_  ((double)(0xFFFFFFFF))
  #define U64TODOUBLE( hi, lo ) ((double)((((double)(hi))* \
           (((double)(_U32LimitDouble_))+((double)(1))))+((double)(lo))))
#endif

// returns keyrate as string (maxlen=26) "nnnn.nn ['K'|'M'|'G'|'T']"
// return value is a pointer to buffer.
char *CliGetKeyrateAsString( char *buffer, double rate );

// return iter/keysdone/whatever as string. set inNetOrder if 'u' 
// needs ntohl()ing first, set contestID = -1 to have the ID ignored
const char *CliGetU64AsString( u64 *u, int inNetOrder, int contestID );

// combines CliGetKeyrateForProblem() and CliGetKeyrateAsString()
const char *CliGetKeyrateStringForProblem( Problem *prob );

// combines CliGetKeyrateForContest() and CliGetKeyrateAsString()
const char *CliGetKeyrateStringForContest( int contestid );

// "4 RC5 Blocks 12:34:56.78 - [123456789 kps]"
const char *CliGetSummaryStringForContest( int contestid );

// [time] Queued RC5 1*2^30 block 68E0D85A:A0000000 (10.25% done)
const char *CliGetMessageForFileentryLoaded( FileEntry *fileentry );

// [time] Completed RC5 block 68E0D85A:A0000000 (123456789 keys)
//           123:45:67:89 - [987654321 kps]
const char *CliGetMessageForProblemCompleted( Problem *problem );

// breaks 'message' into (max) two lines with correct word wrap
// forced newlines ('\n') and non-breaking space ('\xFF') are supported
const char *CliReformatMessage( char *header, char *message );

#endif // ifdef _CLICSTAT_H_

