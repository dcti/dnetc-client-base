// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: clicdata.h,v $
// Revision 1.11  1998/07/28 11:44:50  blast
// Amiga specific changes
//
// Revision 1.10  1998/07/15 05:49:11  ziggyb
// included the header baseincs.h because that's where timeval is and it won't compile without it being defined
//
// Revision 1.9  1998/07/07 21:55:08  cyruspatel
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
// Revision 1.8  1998/06/29 06:57:29  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.7  1998/06/22 11:25:46  cyruspatel
// Created new function in clicdata.cpp: CliClearContestSummaryData(int c)
// Needed to flush/clear accumulated statistics for a particular contest.
// Inserted into all ::SelectCore() sections that use a benchmark to select
// the fastest core. Would otherwise skew the statistics for any subsequent
// completed problem.
//
// Revision 1.6  1998/06/14 08:12:31  friedbait
// 'Log' keywords added to maintain automatic change history
//

// This file contains functions for obtaining contest constants (name, id,
// iteration-to-keycount-multiplication-factor) or obtaining/adding to
// contest summary data (totalblocks, totaliterations, totaltime).
// The data itself is hidden from other modules to protect integrity and
// ease maintenance. 

#ifndef _CLICDATA_H_
#define _CLICDATA_H_
#include "baseincs.h"   // for timeval
#if (CLIENT_OS == OS_AMIGAOS)
#include <sys/time.h> // This is to make it compile, there was a define in there
                      // that was needed 'cause someone changed the source tree :)
#endif

// return 0 if contestID is invalid, non-zero if valid.
int CliIsContestIDValid(int contestID);

// obtain the contestID for a contest identified by name.
// returns -1 if invalid name (contest not found).
int CliGetContestIDFromName( char *name );

// obtain constant data for a contest. name/iter2key may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoBaseData( int contestid, const char **name, unsigned int *iter2key );

// obtain summary data for a contest. unrequired args may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoSummaryData( int contestid, unsigned int *totalblocks,
                                double *totaliter, struct timeval *totaltime);

// clear summary data for a contest.
// returns 0 if success, !0 if error (bad contestID).
int CliClearContestInfoSummaryData( int contestid );

// add data to the summary data for a contest.
// returns 0 if added successfully, !0 if error (bad contestID).
int CliAddContestInfoSummaryData( int contestid, unsigned int *addblocks,
                                double *aditer, struct timeval *addtime );

// Return a usable contest name, returns "???" if bad id.
const char *CliGetContestNameFromID(int contestid);

#endif //ifndef _CLICDATA_H_

