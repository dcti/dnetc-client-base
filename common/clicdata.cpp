// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// This file contains functions for obtaining contest constants (name, id,
// iteration-to-keycount-multiplication-factor) or obtaining/adding to
// contest summary data (totalblocks, totaliterations, totaltime).
// The data itself is hidden from other modules to protect integrity and
// ease maintenance.
//
//
// $Log: clicdata.cpp,v $
// Revision 1.12  1998/10/04 11:35:23  remi
// Id tags fun.
//
// Revision 1.11  1998/07/07 21:55:07  cyruspatel
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
// Revision 1.10  1998/06/29 08:43:45  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.9  1998/06/29 06:57:28  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.8  1998/06/22 11:25:44  cyruspatel
// Created new function in clicdata.cpp: CliClearContestSummaryData(int c)
// Needed to flush/clear accumulated statistics for a particular contest.
// Inserted into all ::SelectCore() sections that use a benchmark to select
// the fastest core. Would otherwise skew the statistics for any subsequent
// completed problem.
//
// Revision 1.7  1998/06/15 12:03:45  kbracey
// Lots of consts.
//
// Revision 1.6  1998/06/14 08:26:37  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.5  1998/06/14 08:12:30  friedbait
// 'Log' keywords added to maintain automatic change history
//

#if (!defined(lint) && defined(__showids__))
const char *clicdata_cpp(void) {
return "@(#)$Id: clicdata.cpp,v 1.12 1998/10/04 11:35:23 remi Exp $"; }
#endif

#include "baseincs.h" //for timeval
#include "clitime.h" //required for CliTimerDiff() and CliClock()

// ---------------------------------------------------------------------------

static struct contestInfo
{
  const char *ContestName;
  int ContestID;
  unsigned int Iter2KeyFactor; /* by how much must iterations/keysdone
                        be multiplied to get the number of keys checked. */
  unsigned int BlocksDone;
  double IterDone;
  struct timeval TimeDone;
  struct timeval TimeStart;
} conStats[] = {  { "RC5", 0, 1, 0, 0, {0,0}, {0,0} },
                  { "DES", 1, 2, 0, 0, {0,0}, {0,0} },
                  {  NULL,-1, 0, 0, 0, {0,0}, {0,0} }  };

// ---------------------------------------------------------------------------

static struct contestInfo *__internalCliGetContestInfoVectorForID( int contestid )
{
  for (int i = 0; conStats[i].ContestName != NULL; i++)
  {
    if (conStats[i].ContestID == contestid)
      return (&conStats[i]);
  }
  return ((struct contestInfo *)(NULL));
}

// ---------------------------------------------------------------------------

// obtain the contestID for a contest identified by name.
// returns -1 if invalid name (contest not found).
int CliGetContestIDFromName( char *name )
{
  for (int i = 0; conStats[i].ContestName != NULL; i++)
  {
    int n;
    for (n = 0; conStats[i].ContestName[n] != 0; n++)
    {
      if (conStats[i].ContestName[n] != name[n])
        return -1;
    }
    if (!name[n])
      return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------

// obtain constant data for a contest. name/iter2key may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoBaseData( int contestid, const char **name, unsigned int *iter2key )
{
  struct contestInfo *conInfo =
                       __internalCliGetContestInfoVectorForID( contestid );
  if (!conInfo)
    return -1;
  if (name)     *name = conInfo->ContestName;
  if (iter2key) *iter2key = (conInfo->Iter2KeyFactor<=1)?(1):(conInfo->Iter2KeyFactor);
  return 0;
}

// ---------------------------------------------------------------------------

// reset the contest summary data for a contest
int CliClearContestInfoSummaryData( int contestid )
{
  struct contestInfo *conInfo =
                      __internalCliGetContestInfoVectorForID( contestid );
  if (!conInfo)
    return -1;
  conInfo->BlocksDone = 0;
  conInfo->IterDone = (double)(0);
  conInfo->TimeDone.tv_sec = conInfo->TimeDone.tv_usec = 0;
  conInfo->TimeStart.tv_sec = conInfo->TimeStart.tv_usec = 0;
  return 0;
}  

// ---------------------------------------------------------------------------

    //obtain summary data for a contest. unrequired args may be NULL
    //returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoSummaryData( int contestid, unsigned int *totalblocks,
                               double *totaliter, struct timeval *totaltime)
{
  struct contestInfo *conInfo =
                      __internalCliGetContestInfoVectorForID( contestid );
  if (!conInfo)
    return -1;
  if (totalblocks) *totalblocks = conInfo->BlocksDone;
  if (totaliter)   *totaliter   = conInfo->IterDone;
  if (totaltime)
  {
    if (conInfo->BlocksDone <= 1)
    {
      totaltime->tv_sec = conInfo->TimeDone.tv_sec;
      totaltime->tv_usec = conInfo->TimeDone.tv_usec;
    }
    else
    {
      //get time since first call to CliTimer() (time when 1st prob started)
      CliClock(totaltime);
      if (totaltime->tv_sec >= conInfo->TimeDone.tv_sec)
      {
        //no overlap means non-mt or only single thread
        totaltime->tv_sec = conInfo->TimeDone.tv_sec;
        totaltime->tv_usec = conInfo->TimeDone.tv_usec;
      }
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------

// add data to the summary data for a contest.
// returns 0 if added successfully, !0 if error (bad contestID).
int CliAddContestInfoSummaryData( int contestid, unsigned int *addblocks,
                                double *additer, struct timeval *addtime)
{
  struct contestInfo *conInfo =
                       __internalCliGetContestInfoVectorForID( contestid );
  if (!conInfo)
    return -1;
  if (addblocks) conInfo->BlocksDone += (*addblocks);
  if (additer)   conInfo->IterDone = conInfo->IterDone + (*additer);
  if (addtime)
  {
    conInfo->TimeDone.tv_sec += addtime->tv_sec;
    if ((conInfo->TimeDone.tv_usec += addtime->tv_usec) > 1000000L)
    {
      conInfo->TimeDone.tv_sec += (conInfo->TimeDone.tv_usec / 1000000L);
      conInfo->TimeDone.tv_usec %= 1000000L;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------

// return 0 if contestID is invalid, non-zero if valid.
int CliIsContestIDValid(int contestid)
{
  return (__internalCliGetContestInfoVectorForID(contestid) != NULL);
}

// ---------------------------------------------------------------------------

// Return a usable contest name.
const char *CliGetContestNameFromID(int contestid)
{
  struct contestInfo *conInfo =
                     __internalCliGetContestInfoVectorForID( contestid );
  if (conInfo)
    return conInfo->ContestName;
  return ((const char *)("???"));
}

// ---------------------------------------------------------------------------

