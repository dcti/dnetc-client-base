/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for obtaining contest constants as 
 * well as name, id, iteration-to-keycount-multiplication-factor or 
 * obtaining/adding to contest summary data (totalblocks, totaliterations, 
 * totaltime. The data itself is hidden from other modules to protect 
 * integrity and ease maintenance. 
 * ----------------------------------------------------------------------
*/ 
#ifndef __CLICDATA_H__
#define __CLICDATA_H__ "@(#)$Id: clicdata.h,v 1.19.2.2 2000/01/08 23:18:01 cyp Exp $"

// return 0 if contestID is invalid, non-zero if valid.
int CliIsContestIDValid(int contestID);

// obtain the contestID for a contest identified by name.
// returns -1 if invalid name (contest not found).
int CliGetContestIDFromName( char *name );

// obtain constant data for a contest. name/iter2key may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoBaseData( int contestid, const char **name, 
                                               unsigned int *iter2key );

struct timeval; /* forward ref */

// obtain summary data for a contest. unrequired args may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoSummaryData( int contestid, unsigned int *totalblocks,
    double *totaliter, struct timeval *totaltime, unsigned int *totalunits);

// clear summary data for a contest.
// returns 0 if success, !0 if error (bad contestID).
int CliClearContestInfoSummaryData( int contestid );

// add data to the summary data for a contest.
// returns 0 if added successfully, !0 if error (bad contestID).
int CliAddContestInfoSummaryData( int contestid, unsigned int *addblocks,
    double *aditer, struct timeval *addtime, unsigned int *addunits );

// Return a usable contest name, returns "???" if bad id.
const char *CliGetContestNameFromID(int contestid);

// returns the expected time to complete a work unit, in seconds
// if force is true, then a microbenchmark will be done to get the
// rate if no work on this contest has been completed yet.
int CliGetContestWorkUnitSpeed( int contestid, int force = 0 );

// sets a possible new value for best time; returns true
// if this speed was a new record
int CliSetContestWorkUnitSpeed( int contestid, unsigned int sec);

#endif /* __CLICDATA_H__ */

