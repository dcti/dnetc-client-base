// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for obtaining contest constants (name, id,
// iteration-to-keycount-multiplication-factor) or obtaining/adding to
// contest summary data (totalblocks, totaliterations, totaltime). 
// The data itself is hidden from other modules to protect integrity and
// ease maintenance.

/* module history:
   01 May 1998 - created - Cyrus Patel <cyp@fb14.uni-mainz.de>
*/   


#ifndef _CLICDATA_H_
#define _CLICDATA_H_

#include "client.h" //required for struct timeval and NULL definition
#include "clitime.h" //required for CliTimerDiff()

    //return 0 if contestID is invalid, non-zero if valid.
int CliIsContestIDValid(int contestID);

    //obtain the contestID for a contest identified by name.
    //returns -1 if invalid name (contest not found).
int CliGetContestIDFromName( char *name );

    //obtain constant data for a contest. name/iter2key may be NULL
    //returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoBaseData( int contestid, char **name, unsigned int *iter2key );

    //obtain summary data for a contest. unrequired args may be NULL
    //returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoSummaryData( int contestid, unsigned int *totalblocks, 
                                double *totaliter, struct timeval *totaltime);

    //add data to the summary data for a contest. 
    //returns 0 if added successfully, !0 if error (bad contestID).
int CliAddContestInfoSummaryData( int contestid, unsigned int *addblocks, 
                                double *aditer, struct timeval *addtime );

  //Return a usable contest name, returns "???" if bad id.
const char *CliGetContestNameFromID(int contestid);

#endif //ifndef _CLICDATA_H_
