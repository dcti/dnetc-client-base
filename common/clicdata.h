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
#define __CLICDATA_H__ "@(#)$Id: clicdata.h,v 1.19.2.6.2.1 2001/03/22 22:03:56 sampo Exp $"

// return 0 if contestID is invalid, non-zero if valid.
int CliIsContestIDValid(int contestID);

// obtain the contestID for a contest identified by name.
// returns -1 if invalid name (contest not found).
int CliGetContestIDFromName( char *name );

#if 0
// obtain constant data for a contest. name/iter2key may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoBaseData( int contestid, const char **name, 
                                               unsigned int *iter2key );
#endif

// clear summary data for a contest.
// returns 0 if success, !0 if error (bad contestID).
int CliClearContestInfoSummaryData( int contestid );

// obtain summary data for a contest. unrequired args may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoSummaryData( int contestid, unsigned int *totalblocks,
                                  u32 *donei_hi, u32 *done_ilo, 
                                  struct timeval *totaltime, 
                                  unsigned int *totalunits );

// add data to the summary data for a contest.
// returns 0 if added successfully, !0 if error (bad contestID).
int CliAddContestInfoSummaryData( int contestid, 
                                  u32 iter_hi, u32 iter_lo, 
                                  const struct timeval *addtime, 
                                  unsigned int addunits );

// Return a usable contest name, returns "???" if bad id.
const char *CliGetContestNameFromID(int contestid);

// Return a usable contest unit name, returns "???" if bad id.
const char *CliGetContestUnitFromID(int contestid);

// returns the expected time to complete a work unit, in seconds
// if force is true, then a microbenchmark will be done to get the
// rate if no work on this contest has been completed yet.
int CliGetContestWorkUnitSpeed( int contestid, int do_force, int *was_forced );

// sets a possible new value for best time; returns true
// if this speed was a new record
int CliSetContestWorkUnitSpeed( int contestid, unsigned int sec);

#endif /* __CLICDATA_H__ */

