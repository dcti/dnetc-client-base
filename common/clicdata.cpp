/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for obtaining contest constants (name, id,
 * iteration-to-keycount-multiplication-factor) or obtaining/adding to
 * contest summary data (totalblocks, totaliterations, totaltime).
 * The data itself is hidden from other modules to protect integrity and
 * ease maintenance.
 * ----------------------------------------------------------------------
*/ 
const char *clicdata_cpp(void) {
return "@(#)$Id: clicdata.cpp,v 1.18.2.9 2000/10/27 17:58:42 cyp Exp $"; }

#include "baseincs.h" //for timeval
#include "clitime.h" //required for CliTimerDiff() and CliClock()
#include "bench.h"    // for TBenchmark
#include "selcore.h"  // for selcoreGetSelectedCoreForContest()

/* ------------------------------------------------------------------------ */

static struct contestInfo
{
  const char *ContestName;
  int ContestID;
  unsigned int Iter2KeyFactor; /* by how much must iterations/keysdone
                        be multiplied to get the number of keys checked. */
  unsigned int BlocksDone;
  double IterDone;
  struct timeval TimeDone;
  unsigned int UnitsDone;
  unsigned int BestTime;  /* in seconds */
} conStats[] = {  { "RC5", 0,  1, 0, 0, {0,0}, 0, 0 },
                  { "DES", 1,  2, 0, 0, {0,0}, 0, 0 },
                  { "OGR", 2,  1, 0, 0, {0,0}, 0, 0 },
                  { "CSC", 3,  1, 0, 0, {0,0}, 0, 0 },
                  {  NULL,-1,  0, 0, 0, {0,0}, 0, 0 }  };

/* ----------------------------------------------------------------------- */

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

// returns the expected time to complete a work unit, in seconds
// if force is true, then a microbenchmark will be done to get the
// rate if no work on this contest has been completed yet.
int CliGetContestWorkUnitSpeed( int contestid, int force)
{
  struct contestInfo *conInfo =
                       __internalCliGetContestInfoVectorForID( contestid );
  if (conInfo)
  {
    if ((conInfo->BestTime == 0) && force)
    {
      // This may trigger a mini-benchmark, which will get the speed
      // we need and not waste time.
      selcoreGetSelectedCoreForContest( contestid );

      if (conInfo->BestTime == 0)
        TBenchmark(contestid, 2, TBENCHMARK_QUIET | TBENCHMARK_IGNBRK);
    }
    return conInfo->BestTime;
  }
  return 0;  
}

// ---------------------------------------------------------------------------

// set new record speed for a contest, returns !0 on success
int CliSetContestWorkUnitSpeed( int contestid, unsigned int sec)
{
  struct contestInfo *conInfo =
                       __internalCliGetContestInfoVectorForID( contestid );
  if (conInfo && sec)
  {
    if ((conInfo->BestTime == 0) || (conInfo->BestTime > sec))
    {
      conInfo->BestTime = sec;
      return 1; /* success */
    }
  }  
  return 0; /* failed */
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
  conInfo->UnitsDone = 0;
  conInfo->BestTime = 0;
  return 0;
}  

// ---------------------------------------------------------------------------

// obtain summary data for a contest. unrequired args may be NULL
// returns 0 if success, !0 if error (bad contestID).
int CliGetContestInfoSummaryData( int contestid, unsigned int *totalblocks,
                                  double *totaliter, struct timeval *totaltime, 
                                  unsigned int *totalunits, double *avgrate)
{
  struct contestInfo *conInfo =
                      __internalCliGetContestInfoVectorForID( contestid );
  if (!conInfo)
    return -1;
  if (totalblocks) *totalblocks = conInfo->BlocksDone;
  if (totaliter)   *totaliter   = conInfo->IterDone;
  if (totalunits)  *totalunits  = conInfo->UnitsDone;
  if (totaltime || avgrate)
  {
    struct timeval tv;
    tv.tv_sec = conInfo->TimeDone.tv_sec;
    tv.tv_usec = conInfo->TimeDone.tv_usec;
    if (conInfo->BlocksDone > 1)
    {
      //get time since first call to CliTimer() (time when 1st prob started)
      struct timeval tv2;
      CliClock(&tv2);
      if (tv2.tv_sec < tv.tv_sec) 
      {   //no overlap means non-mt or only single thread
        tv.tv_sec = tv2.tv_sec;
        tv.tv_usec = tv2.tv_usec;
      }
    }
    if (totaltime)
    {
      totaltime->tv_sec = tv.tv_sec;
      totaltime->tv_usec = tv.tv_usec;
    }
    if (avgrate)
    {
      double r = ((double)0); 
      if (conInfo->BlocksDone)
      {
        r = conInfo->IterDone;
        if (tv.tv_sec || tv.tv_usec)
          r=r/(((double)tv.tv_sec)+(((double)tv.tv_usec)/((double)(1000000L))));
      }
      *avgrate = r;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------

// add data to the summary data for a contest.
// returns 0 if added successfully, !0 if error (bad contestID).
int CliAddContestInfoSummaryData( int contestid, 
                                  u32 iter_hi, u32 iter_lo, 
                                  const struct timeval *addtime,
                                  unsigned int addunits,
                                  double addrate )
{
  struct contestInfo *conInfo =
                       __internalCliGetContestInfoVectorForID( contestid );
  if (!conInfo || !addtime)
    return -1;
  conInfo->BlocksDone++;
  if (contestid != 2) /* OGR */
  {
    unsigned int sec = (unsigned int)((1<<28)/addrate + 0.5);
    CliSetContestWorkUnitSpeed(contestid, sec);
  }
  conInfo->IterDone = conInfo->IterDone + 
                       (((double)iter_hi)*4294967296.0)+((double)iter_lo);
  conInfo->UnitsDone += addunits;
  conInfo->TimeDone.tv_sec += addtime->tv_sec;
  conInfo->TimeDone.tv_usec += addtime->tv_usec;
  if (conInfo->TimeDone.tv_usec > 1000000L)
  {
    conInfo->TimeDone.tv_sec += (conInfo->TimeDone.tv_usec / 1000000L);
    conInfo->TimeDone.tv_usec %= 1000000L;
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
